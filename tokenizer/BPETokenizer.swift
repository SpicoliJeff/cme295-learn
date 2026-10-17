// =============================================================================
//  BPETokenizer.swift
//
//  A complete, byte-level Byte-Pair-Encoding (BPE) tokenizer in one Swift file,
//  in the style of GPT-2 / tiktoken. It combines and fixes the four earlier
//  versions (ByteLevelTokenizer → AdvancedByteLevelTokenizer →
//  FinalByteLevelTokenizer → ProductionBPETokenizer).
//
//  Build & run (macOS or Linux, needs Foundation):
//      swiftc -O BPETokenizer.swift -o tok && ./tok
//
//  ---------------------------------------------------------------------------
//  THE BIG PICTURE (why a tokenizer exists at all)
//  ---------------------------------------------------------------------------
//  A language model cannot read text. It reads a list of integers ("token IDs")
//  and each ID indexes a row in an embedding table. The tokenizer is the
//  translator between the two worlds:
//
//      "Hello world"  --encode-->  [15496, 995]  --model-->  ...  --decode-->  text
//
//  Design pressures that shape every decision in this file:
//    * Never fail on any input. Emoji, Chinese, typos, binary junk: everything
//      must become *some* list of IDs. Byte-level BPE guarantees this, because
//      every possible byte (0-255) is already a token before training starts.
//    * Short sequences. Attention cost grows with sequence length (roughly
//      quadratically), and the context window is fixed, so we want common
//      chunks like " the" or "ing" to be ONE token instead of 4 or 3 bytes.
//      BPE learns these chunks from data by repeatedly merging the most
//      frequent adjacent pair.
//    * Exact round trip. decode(encode(x)) must equal x byte-for-byte,
//      otherwise the model literally sees different text than you gave it.
//    * Determinism. The same corpus must always produce the same merges and
//      the same IDs, or a saved model and its tokenizer silently disagree.
//
//  VOCABULARY LAYOUT (the ID-numbering rule this file uses everywhere):
//
//      0 ..< 256                   base byte tokens (ID n means the byte n)
//      256 ..< 256 + merges.count  learned merges, in rank order
//                                  (merge rank r always has ID 256 + r)
//      after that                  special tokens ([BOS], [EOS], [PAD], ...)
//
//  `targetVocabSize` passed to train() counts ONLY base + merges. Special tokens
//  are registered AFTER training and get the next free IDs, so they can never
//  collide with a merge ID. (Registering specials and then re-training is
//  refused with an error, because it would make the IDs collide.)
// =============================================================================

import Foundation

// MARK: - Errors

/// Everything that can go wrong, with a readable message.
/// Using real errors (instead of `fatalError` or silently ignoring problems)
/// means a bad JSON file or a typo'd ID is reported, not swallowed.
public enum TokenizerError: Error, CustomStringConvertible {
    case invalidVocabSize(requested: Int, minimum: Int)
    case trainingAfterSpecialTokens
    case emptySpecialToken
    case unknownTokenId(Int)
    case invalidModel(String)

    public var description: String {
        switch self {
        case let .invalidVocabSize(requested, minimum):
            return "targetVocabSize \(requested) is smaller than the \(minimum) base byte tokens"
        case .trainingAfterSpecialTokens:
            return "train() called after special tokens were registered; their IDs would collide with new merge IDs. Train first, then register specials."
        case .emptySpecialToken:
            return "a special token cannot be the empty string"
        case let .unknownTokenId(id):
            return "token id \(id) is not in the vocabulary"
        case let .invalidModel(reason):
            return "invalid saved model: \(reason)"
        }
    }
}

// MARK: - Small value types

/// Two adjacent token IDs, e.g. (104 'h', 101 'e').
///
/// WHY token IDs instead of byte arrays (the old `PairWrapper` held `[UInt8]`)?
/// Comparing/hashing two Ints is much cheaper than hashing two arrays, and IDs
/// are unambiguous: the bytes "abc" could come from ("a","bc") or ("ab","c"),
/// but each pair of IDs names exactly one merge.
public struct TokenPair: Hashable, Codable {
    public let left: Int
    public let right: Int
}

/// One learned merge rule: "whenever `left` is directly followed by `right`,
/// replace both with `result`". Its position in the `merges` array is its
/// RANK: lower rank = learned earlier = more frequent in the training data
/// = applied first when encoding.
public struct Merge: Equatable {
    public let left: Int
    public let right: Int
    public let result: Int
    public var pair: TokenPair { TokenPair(left: left, right: right) }
}

/// Numbers reported by `benchmarkCompression`.
public struct CompressionStats {
    public let characters: Int      // Swift Characters (what a human counts as "letters")
    public let bytes: Int           // UTF-8 bytes (what a byte-level model would see untokenized)
    public let tokens: Int          // token IDs after BPE
    public let charsPerToken: Double
    public let bytesPerToken: Double
    public let savingsPercent: Double // how many fewer IDs than raw bytes, in %
}

// MARK: - The tokenizer

public final class BPETokenizer {

    // ---------------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------------

    /// Number of possible byte values. These IDs exist before any training,
    /// which is what makes the tokenizer unable to hit an "unknown token".
    public static let baseVocabSize = 256

    /// Version tag written into saved JSON so a future format change can be
    /// detected instead of misread.
    public static let fileFormatVersion = 2

    /// GPT-2's pre-tokenization regex. Read the alternatives left to right:
    ///
    ///   's|'t|'re|'ve|'m|'ll|'d   English contractions become their own piece
    ///                             ("world's" → "world" + "'s")
    ///    ?\p{L}+                  a run of letters, optionally with ONE leading space
    ///    ?\p{N}+                  a run of digits,  optionally with ONE leading space
    ///    ?[^\s\p{L}\p{N}]+        a run of punctuation/symbols/emoji, same idea
    ///   \s+(?!\S)                 whitespace, but leave the last space for the next word
    ///   \s+                       any remaining whitespace (e.g. at the end of text)
    ///
    /// WHY pre-tokenize at all? Without it, BPE would happily learn merges
    /// that span word boundaries like "e t" or "., T", wasting vocabulary on
    /// accidental neighbours. Splitting first means merges only happen INSIDE
    /// a piece, so learned tokens look like real word parts.
    ///
    /// WHY the optional leading space (" ?")? Your AdvancedByteLevelTokenizer
    /// used the same regex without it, which made every space its own piece.
    /// With it, " world" is a single piece and can become a single token, which
    /// is how GPT-2 gets ~4 characters per token on English. `\s+(?!\S)` is the
    /// partner trick: in "a   b" the first two spaces form one piece and the
    /// last space attaches to " b".
    ///
    /// Every character is a letter (\p{L}), a number (\p{N}), whitespace (\s)
    /// or "something else", so the alternatives cover ALL input. That matters:
    /// any text the regex skipped would be lost. (preTokenize() also keeps any
    /// gap defensively, just in case.)
    public static let gpt2SplitPattern =
        #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

    // ---------------------------------------------------------------------
    // State
    // ---------------------------------------------------------------------

    /// ID → the exact bytes that token stands for. Decoding is just "look up
    /// every ID and concatenate the bytes".
    public private(set) var vocab: [Int: [UInt8]] = [:]

    /// Learned merges IN ORDER. Index = rank. This order is the heart of BPE:
    /// the old versions stored merges in a Dictionary, which has no order, so
    /// encoding could apply merges in a different order than training learned
    /// them and produce different (worse, non-reproducible) tokens.
    public private(set) var merges: [Merge] = []

    /// Fast lookup: pair → rank. Encoding asks "of all adjacent pairs in this
    /// piece, which has the lowest rank?" thousands of times, so it must be O(1).
    private var mergeRanks: [TokenPair: Int] = [:]

    /// Special token text → ID, and the reverse map for decoding.
    public private(set) var specialTokens: [String: Int] = [:]
    private var specialTokensById: [Int: String] = [:]

    /// The ID the next merge or special token will receive. Always equals
    /// (largest ID in use) + 1, so IDs are dense and never reused.
    public private(set) var nextTokenId: Int = BPETokenizer.baseVocabSize

    /// Compiled once: NSRegularExpression compilation is expensive.
    private let splitRegex: NSRegularExpression

    /// Cache: pre-token piece → its token IDs. Natural text repeats the same
    /// pieces (" the", " of", ".") constantly, so caching skips the merge loop
    /// for most of them. Cleared whenever merges change.
    private var fragmentCache: [String: [Int]] = [:]
    private let maxCacheEntries = 50_000

    /// Cache of compiled "split on these special tokens" regexes, keyed by
    /// the joined pattern, so repeated encode calls don't recompile.
    private var specialRegexCache: [String: NSRegularExpression] = [:]

    // ---------------------------------------------------------------------
    // Init & simple properties
    // ---------------------------------------------------------------------

    public init() {
        // The pattern is a compile-time constant we control, so failure here
        // would be a programming error, not a user error.
        splitRegex = try! NSRegularExpression(pattern: Self.gpt2SplitPattern)
        resetToBaseVocabulary()
    }

    /// Base bytes + merges + specials: the size of the model's embedding table.
    public var vocabSize: Int { vocab.count + specialTokens.count }

    /// Convenience: pass this as `allowedSpecial` to allow every registered special.
    public var allSpecialTokens: Set<String> { Set(specialTokens.keys) }

    /// Forget everything learned; keep only the 256 byte tokens.
    private func resetToBaseVocabulary() {
        vocab = Dictionary(uniqueKeysWithValues: (0..<Self.baseVocabSize).map { ($0, [UInt8($0)]) })
        merges = []
        mergeRanks = [:]
        specialTokens = [:]
        specialTokensById = [:]
        nextTokenId = Self.baseVocabSize
        fragmentCache = [:]
        specialRegexCache = [:]
    }

    // =====================================================================
    // MARK: Step 1 — Pre-tokenization
    // =====================================================================

    /// Split text into pieces with the GPT-2 regex. Concatenating the pieces
    /// gives back the original text exactly (nothing is thrown away; this is
    /// the fix for ByteLevelTokenizer, which split on whitespace and lost the
    /// spaces, so decode could never reproduce the input).
    public func preTokenize(_ text: String) -> [String] {
        if text.isEmpty { return [] }
        var pieces: [String] = []
        var cursor = text.startIndex
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        for match in splitRegex.matches(in: text, options: [], range: fullRange) {
            // NSRegularExpression speaks UTF-16 offsets (NSRange); convert
            // back to Swift String indices before slicing.
            guard let range = Range(match.range, in: text) else { continue }
            if cursor < range.lowerBound {
                // Defensive: keep any text the regex did not match.
                pieces.append(String(text[cursor..<range.lowerBound]))
            }
            pieces.append(String(text[range]))
            cursor = range.upperBound
        }
        if cursor < text.endIndex { pieces.append(String(text[cursor...])) }
        return pieces
    }

    // =====================================================================
    // MARK: Step 2 — Training (learning the merges)
    // =====================================================================

    /// Learn merges from `corpus` until the vocabulary (base + merges) reaches
    /// `targetVocabSize`, or until no pair occurs at least `minPairFrequency`
    /// times (merging a pair seen once teaches nothing general).
    ///
    /// Algorithm (the classic BPE loop):
    ///   1. Pre-tokenize the corpus and count how often each distinct piece
    ///      occurs. We train on (piece, count) instead of the raw stream, so
    ///      " the" appearing 500 times is processed once with weight 500.
    ///   2. Write every piece as its UTF-8 bytes = base token IDs.
    ///   3. Repeat: count all adjacent pairs (weighted), pick the most
    ///      frequent, give it a new ID, and replace it everywhere.
    ///
    /// Returns the number of merges actually learned.
    @discardableResult
    public func train(corpus: String, targetVocabSize: Int, minPairFrequency: Int = 2) throws -> Int {
        guard specialTokens.isEmpty else { throw TokenizerError.trainingAfterSpecialTokens }
        guard targetVocabSize >= Self.baseVocabSize else {
            throw TokenizerError.invalidVocabSize(requested: targetVocabSize, minimum: Self.baseVocabSize)
        }
        resetToBaseVocabulary()

        // 1. Count distinct pieces.
        var pieceCounts: [String: Int] = [:]
        for piece in preTokenize(corpus) { pieceCounts[piece, default: 0] += 1 }

        // 2. Pieces as arrays of byte IDs. Sorting the keys makes the
        //    working arrays identical on every run (Dictionary order is
        //    randomized per process in Swift).
        var words: [[Int]] = []
        var weights: [Int] = []
        for (piece, count) in pieceCounts.sorted(by: { $0.key < $1.key }) {
            words.append(piece.utf8.map { Int($0) })
            weights.append(count)
        }

        // 3. The merge loop.
        let mergesWanted = targetVocabSize - Self.baseVocabSize
        for _ in 0..<mergesWanted {
            // 3a. Count adjacent pairs, weighted by how often each piece occurs.
            var pairCounts: [TokenPair: Int] = [:]
            for (word, weight) in zip(words, weights) where word.count >= 2 {
                for i in 0..<(word.count - 1) {
                    pairCounts[TokenPair(left: word[i], right: word[i + 1]), default: 0] += weight
                }
            }

            // 3b. Pick the winner deterministically (see bestPair).
            guard let (pair, count) = bestPair(in: pairCounts), count >= minPairFrequency else { break }

            // 3c. Record the merge. Its rank is its index; its ID is 256 + rank.
            let newId = nextTokenId
            vocab[newId] = vocab[pair.left]! + vocab[pair.right]!
            mergeRanks[pair] = merges.count
            merges.append(Merge(left: pair.left, right: pair.right, result: newId))
            nextTokenId += 1

            // 3d. Apply it everywhere so the next round counts the new token.
            for i in words.indices where words[i].count >= 2 {
                words[i] = Self.applyMerge(words[i], pair: pair, newId: newId)
            }
        }

        fragmentCache.removeAll()
        return merges.count
    }

    /// The most frequent pair, with a DETERMINISTIC tie-break.
    ///
    /// WHY: when two pairs have the same count, the old code picked whichever
    /// the Dictionary happened to visit first, and Swift randomizes Dictionary
    /// order per run. So training the same corpus twice could give different
    /// merges and different IDs. Here ties are broken by comparing the pairs'
    /// bytes lexicographically (left token's bytes first, then right's), and
    /// finally by the IDs themselves, so the answer never depends on luck.
    private func bestPair(in counts: [TokenPair: Int]) -> (TokenPair, Int)? {
        var best: (pair: TokenPair, count: Int)? = nil
        for (pair, count) in counts {
            guard let current = best else { best = (pair, count); continue }
            if count > current.count || (count == current.count && precedes(pair, current.pair)) {
                best = (pair, count)
            }
        }
        return best.map { ($0.pair, $0.count) }
    }

    /// Total order on pairs used only for tie-breaking.
    private func precedes(_ a: TokenPair, _ b: TokenPair) -> Bool {
        let aL = vocab[a.left]!, bL = vocab[b.left]!
        if aL != bL { return aL.lexicographicallyPrecedes(bL) }
        let aR = vocab[a.right]!, bR = vocab[b.right]!
        if aR != bR { return aR.lexicographicallyPrecedes(bR) }
        // Different IDs can spell the same bytes (e.g. "ab"+"c" and "a"+"bc"
        // both make "abc"); fall back to IDs so the order is still total.
        return (a.left, a.right) < (b.left, b.right)
    }

    /// Replace every non-overlapping occurrence of `pair` (scanning left to
    /// right) with `newId`. "aaa" with pair (a,a) → [aa, a], same as GPT-2.
    static func applyMerge(_ ids: [Int], pair: TokenPair, newId: Int) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(ids.count)
        var i = 0
        while i < ids.count {
            if i + 1 < ids.count && ids[i] == pair.left && ids[i + 1] == pair.right {
                out.append(newId)
                i += 2
            } else {
                out.append(ids[i])
                i += 1
            }
        }
        return out
    }

    // =====================================================================
    // MARK: Step 3 — Special tokens
    // =====================================================================

    /// Register a special token such as "[BOS]" and return its ID.
    ///
    /// WHY special tokens? The model needs control signals that can never be
    /// confused with real text: "a document starts here" ([BOS]), "stop
    /// generating" ([EOS]), "this slot is padding, ignore it" ([PAD]). Each
    /// gets ONE dedicated ID that ordinary text can never produce (unless you
    /// explicitly allow it in encode).
    ///
    /// IDs come from `nextTokenId`, which is always past the last merge, so a
    /// special can never collide with a merge. Registering the same text again
    /// just returns the existing ID.
    @discardableResult
    public func registerSpecialToken(_ text: String) throws -> Int {
        guard !text.isEmpty else { throw TokenizerError.emptySpecialToken }
        if let existing = specialTokens[text] { return existing }
        let id = nextTokenId
        specialTokens[text] = id
        specialTokensById[id] = text
        nextTokenId += 1
        specialRegexCache.removeAll()
        return id
    }

    // =====================================================================
    // MARK: Step 4 — Encoding
    // =====================================================================

    /// Encode text to token IDs.
    ///
    /// `allowedSpecial` lists which special-token strings should become their
    /// special ID when they appear in the text. Anything else is encoded as
    /// ordinary bytes. WHY opt-in? If user-supplied text containing "[EOS]"
    /// were turned into the real [EOS] ID, a user could make the model think a
    /// document ended (a classic prompt-injection hole). tiktoken works the same.
    ///
    /// A string in `allowedSpecial` that was never registered has no ID, so it
    /// is encoded as normal text (and still round-trips exactly). It is never
    /// silently dropped.
    public func encode(text: String, allowedSpecial: Set<String> = []) -> [Int] {
        let active = allowedSpecial.filter { specialTokens[$0] != nil }
        guard !active.isEmpty, !text.isEmpty else { return encodeOrdinary(text) }

        let regex = specialSplitRegex(for: active)
        var ids: [Int] = []
        var cursor = text.startIndex
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        for match in regex.matches(in: text, options: [], range: fullRange) {
            guard let range = Range(match.range, in: text) else { continue }
            // Ordinary text before the special token.
            ids += encodeOrdinary(String(text[cursor..<range.lowerBound]))
            // The special token itself → its single reserved ID.
            ids.append(specialTokens[String(text[range])]!)
            cursor = range.upperBound
        }
        ids += encodeOrdinary(String(text[cursor...]))
        return ids
    }

    /// Build (or fetch from cache) a regex matching any of the given specials.
    /// LONGEST FIRST: regex alternation takes the first alternative that
    /// matches, so with "<|end" and "<|end|>" both registered, putting the
    /// short one first would match "<|end" and leave "|>" as junk text.
    private func specialSplitRegex(for specials: Set<String>) -> NSRegularExpression {
        let ordered = specials.sorted { a, b in
            a.count != b.count ? a.count > b.count : a < b   // longest first, then alphabetical
        }
        let pattern = ordered.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")
        if let cached = specialRegexCache[pattern] { return cached }
        // Escaped literals always form a valid pattern.
        let regex = try! NSRegularExpression(pattern: pattern)
        specialRegexCache[pattern] = regex
        return regex
    }

    /// Encode text with NO special-token handling: pre-tokenize, then BPE
    /// each piece independently and concatenate.
    public func encodeOrdinary(_ text: String) -> [Int] {
        var ids: [Int] = []
        for piece in preTokenize(text) { ids += encodePiece(piece) }
        return ids
    }

    /// BPE on one pre-token piece, GPT-2 / tiktoken style:
    ///   start from raw bytes; repeatedly find the adjacent pair with the
    ///   LOWEST merge rank and merge it; stop when no adjacent pair has a rank.
    ///
    /// WHY lowest rank first? That replays the merges in exactly the order
    /// training learned them. Training created "th" before "the", so "the" can
    /// only form if "th" is formed first. Applying merges in arbitrary
    /// (dictionary) order, as the old versions did, can block those chains
    /// and give a different, longer encoding than training intended.
    private func encodePiece(_ piece: String) -> [Int] {
        if let cached = fragmentCache[piece] { return cached }

        var ids = piece.utf8.map { Int($0) }   // step: text → bytes → base IDs
        while ids.count >= 2 {
            var bestRank = Int.max
            var bestPair: TokenPair? = nil
            for i in 0..<(ids.count - 1) {
                let pair = TokenPair(left: ids[i], right: ids[i + 1])
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }
            guard let pair = bestPair else { break }   // nothing left to merge
            ids = Self.applyMerge(ids, pair: pair, newId: merges[bestRank].result)
        }

        if fragmentCache.count >= maxCacheEntries { fragmentCache.removeAll(keepingCapacity: true) }
        fragmentCache[piece] = ids
        return ids
    }

    // =====================================================================
    // MARK: Step 5 — Decoding
    // =====================================================================

    /// IDs → raw bytes. Special IDs emit their text (e.g. "[BOS]").
    public func decodeBytes(_ ids: [Int]) throws -> [UInt8] {
        var bytes: [UInt8] = []
        for id in ids {
            if let b = vocab[id] {
                bytes += b
            } else if let special = specialTokensById[id] {
                bytes += Array(special.utf8)
            } else {
                throw TokenizerError.unknownTokenId(id)
            }
        }
        return bytes
    }

    /// IDs → String.
    ///
    /// WHY join ALL bytes before converting to text? A single token may hold
    /// only part of a multi-byte character (👋 is 4 bytes; a token might hold
    /// 2 of them). Converting token by token would produce garbage; converting
    /// the whole byte stream at once reassembles the characters. If the bytes
    /// are genuinely invalid UTF-8 (e.g. a model generated a stray partial
    /// character), String(decoding:) substitutes U+FFFD "�" instead of
    /// crashing: "lossy" decoding.
    public func decode(_ ids: [Int]) throws -> String {
        String(decoding: try decodeBytes(ids), as: UTF8.self)
    }

    /// Human-readable view of one token, for debugging/teaching output.
    /// Newlines/tabs are escaped; bytes that are not valid UTF-8 on their own
    /// (half of an emoji) are shown as hex like <F0><9F>.
    public func tokenDisplay(_ id: Int) -> String {
        if let special = specialTokensById[id] { return special }
        guard let bytes = vocab[id] else { return "<unknown \(id)>" }
        if let s = String(bytes: bytes, encoding: .utf8) {
            let escaped = s.replacingOccurrences(of: "\n", with: "\\n")
                           .replacingOccurrences(of: "\t", with: "\\t")
            return "\"\(escaped)\""
        }
        return bytes.map { String(format: "<%02X>", $0) }.joined()
    }

    // =====================================================================
    // MARK: Step 6 — Saving and loading (JSON via Codable)
    // =====================================================================

    // On-disk format. Arrays are written SORTED (vocab and specials by ID,
    // merges by rank) and keys sorted, so saving the same tokenizer always
    // produces byte-identical JSON: clean diffs and easy checksums. (Your
    // FinalByteLevelTokenizer used a [String: ...] dictionary, whose order
    // changes between runs.)
    private struct SavedModel: Codable {
        let format: String
        let version: Int
        let baseVocabSize: Int
        let nextTokenId: Int
        let vocab: [SavedVocabEntry]
        let merges: [SavedMerge]
        let specialTokens: [SavedSpecial]
    }
    private struct SavedVocabEntry: Codable { let id: Int; let bytes: [UInt8] }
    private struct SavedMerge: Codable { let rank: Int; let left: Int; let right: Int; let result: Int }
    private struct SavedSpecial: Codable { let id: Int; let text: String }

    /// Serialize the tokenizer to stable, pretty JSON.
    public func jsonData() throws -> Data {
        let model = SavedModel(
            format: "byte-level-bpe",
            version: Self.fileFormatVersion,
            baseVocabSize: Self.baseVocabSize,
            nextTokenId: nextTokenId,
            vocab: vocab.keys.sorted().map { SavedVocabEntry(id: $0, bytes: vocab[$0]!) },
            merges: merges.enumerated().map { SavedMerge(rank: $0.offset, left: $0.element.left,
                                                         right: $0.element.right, result: $0.element.result) },
            specialTokens: specialTokensById.keys.sorted().map { SavedSpecial(id: $0, text: specialTokensById[$0]!) }
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(model)
    }

    public func saveToJSON(url: URL) throws {
        try jsonData().write(to: url, options: .atomic)
    }

    public func loadFromJSON(url: URL) throws {
        try load(jsonData: try Data(contentsOf: url))
    }

    /// Restore vocab, ordered merges + ranks, specials and nextTokenId.
    ///
    /// Everything is VALIDATED into local variables first and only then
    /// committed, so a corrupt file leaves the current tokenizer untouched
    /// instead of half-loaded. Validation rebuilds each merged token from its
    /// two parents and checks it against the stored bytes, so a hand-edited or
    /// truncated file is caught here rather than producing wrong IDs later.
    public func load(jsonData: Data) throws {
        let model: SavedModel
        do { model = try JSONDecoder().decode(SavedModel.self, from: jsonData) }
        catch { throw TokenizerError.invalidModel("JSON could not be decoded (\(error))") }

        func fail(_ why: String) -> TokenizerError { .invalidModel(why) }
        guard model.format == "byte-level-bpe" else { throw fail("unknown format '\(model.format)'") }
        guard model.version == Self.fileFormatVersion else { throw fail("unsupported version \(model.version)") }
        guard model.baseVocabSize == Self.baseVocabSize else { throw fail("baseVocabSize must be 256") }

        // Stored vocab → dictionary (reject duplicate IDs).
        var storedVocab: [Int: [UInt8]] = [:]
        for entry in model.vocab {
            guard storedVocab[entry.id] == nil else { throw fail("duplicate vocab id \(entry.id)") }
            storedVocab[entry.id] = entry.bytes
        }

        // Rebuild the vocab from scratch: bytes, then merges in rank order.
        var newVocab = Dictionary(uniqueKeysWithValues: (0..<Self.baseVocabSize).map { ($0, [UInt8($0)]) })
        var newMerges: [Merge] = []
        var newRanks: [TokenPair: Int] = [:]
        for (expectedRank, m) in model.merges.enumerated() {
            guard m.rank == expectedRank else { throw fail("merge ranks must be 0,1,2,... (found \(m.rank) at position \(expectedRank))") }
            guard m.result == Self.baseVocabSize + expectedRank else { throw fail("merge rank \(m.rank) must have id \(Self.baseVocabSize + expectedRank), found \(m.result)") }
            guard let l = newVocab[m.left], let r = newVocab[m.right] else { throw fail("merge rank \(m.rank) uses an id not defined earlier") }
            let pair = TokenPair(left: m.left, right: m.right)
            guard newRanks[pair] == nil else { throw fail("duplicate merge pair (\(m.left), \(m.right))") }
            newVocab[m.result] = l + r
            newRanks[pair] = expectedRank
            newMerges.append(Merge(left: m.left, right: m.right, result: m.result))
        }
        guard newVocab == storedVocab else { throw fail("stored vocab does not match the vocab rebuilt from merges") }

        // Specials: IDs must sit after all merges, be unique, non-empty.
        let firstSpecialId = Self.baseVocabSize + newMerges.count
        var newSpecials: [String: Int] = [:]
        var newSpecialsById: [Int: String] = [:]
        for s in model.specialTokens {
            guard !s.text.isEmpty else { throw fail("empty special token") }
            guard s.id >= firstSpecialId else { throw fail("special '\(s.text)' id \(s.id) collides with base/merge ids") }
            guard newSpecials[s.text] == nil, newSpecialsById[s.id] == nil else { throw fail("duplicate special token '\(s.text)' / id \(s.id)") }
            newSpecials[s.text] = s.id
            newSpecialsById[s.id] = s.text
        }
        let maxId = max(firstSpecialId - 1, newSpecialsById.keys.max() ?? -1)
        guard model.nextTokenId == maxId + 1 else { throw fail("nextTokenId \(model.nextTokenId) should be \(maxId + 1)") }

        // All good: commit.
        vocab = newVocab
        merges = newMerges
        mergeRanks = newRanks
        specialTokens = newSpecials
        specialTokensById = newSpecialsById
        nextTokenId = model.nextTokenId
        fragmentCache.removeAll()
        specialRegexCache.removeAll()
    }

    // =====================================================================
    // MARK: Step 7 — Compression benchmark
    // =====================================================================

    /// How much shorter is the token sequence than the raw byte sequence?
    /// Higher chars/token = more text fits in the context window and each
    /// forward pass covers more text. Byte-level with no merges would score
    /// exactly 1.0 bytes/token and 0% savings.
    @discardableResult
    public func benchmarkCompression(text: String, label: String = "text",
                                     allowedSpecial: Set<String> = []) -> CompressionStats {
        let chars = text.count
        let bytes = text.utf8.count
        let tokens = encode(text: text, allowedSpecial: allowedSpecial).count
        // Guard: empty input would divide by zero (and give NaN/inf).
        let stats = CompressionStats(
            characters: chars,
            bytes: bytes,
            tokens: tokens,
            charsPerToken: tokens > 0 ? Double(chars) / Double(tokens) : 0,
            bytesPerToken: tokens > 0 ? Double(bytes) / Double(tokens) : 0,
            savingsPercent: bytes > 0 ? (1 - Double(tokens) / Double(bytes)) * 100 : 0
        )
        print("  [\(label)] chars=\(chars) bytes=\(bytes) tokens=\(tokens) "
              + String(format: "chars/token=%.2f bytes/token=%.2f savings=%.1f%%",
                       stats.charsPerToken, stats.bytesPerToken, stats.savingsPercent)
              + (tokens == 0 ? " (empty input: ratios reported as 0)" : ""))
        return stats
    }
}

// =============================================================================
// MARK: - Demo & self-test
// =============================================================================

/// A few paragraphs of English to train on. Real tokenizers train on
/// gigabytes; a small corpus is enough to watch BPE discover " the", "ing",
/// " token", etc.
let trainingCorpus = """
Language models do not read letters. They read numbers, and the tokenizer is the \
piece that turns text into those numbers. A good tokenizer keeps common words \
whole, splits rare words into smaller pieces, and never fails on strange input. \
The model's vocabulary is fixed before training starts, so the tokenizer's \
choices shape everything the model will learn.

Byte pair encoding starts with the two hundred and fifty six possible bytes. It \
counts every pair of neighbouring tokens in the training text, finds the most \
frequent pair, and merges that pair into a new token. Then it counts again and \
merges again. After thousands of merges, the tokens look like real pieces of \
words: the, ing, tion, token, and model. Because every byte is already a token, \
there is no such thing as an unknown word.

When we encode new text, we split it into words and spaces, turn each piece into \
bytes, and replay the merges in the same order that training learned them. When \
we decode, we look up the bytes for every token and join them together. If the \
tokenizer is correct, decoding the encoded text gives back exactly the original \
text, including spaces, new lines, numbers like 2026 and 295, and emoji.

Special tokens are different. They mark the beginning of a sequence, the end of \
a sequence, or padding that the model should ignore. They are added after the \
merges, so their numbers never collide with the numbers of the learned tokens. \
The tokenizer's job is simple to describe and easy to get subtly wrong, which is \
why we test it: we encode, we decode, we compare, and we check that it's exact.
"""

/// Held-out text (NOT in the training corpus) for an honest benchmark.
let heldOutText = """
Transformers process tokens in parallel, and attention lets every token look at \
every other token. Longer sequences cost more compute, so a tokenizer that packs \
more text into each token makes training and inference cheaper.
"""

var passCount = 0
var failCount = 0

/// Print one PASS/FAIL line and keep score.
func check(_ condition: Bool, _ name: String, _ detail: String = "") {
    if condition { passCount += 1 } else { failCount += 1 }
    print("\(condition ? "PASS" : "FAIL"): \(name)\(detail.isEmpty ? "" : "  (\(detail))")")
}

/// Reference implementation for cross-checking: apply EVERY merge in rank
/// order to the whole byte sequence. Lowest-rank-first greedy merging (what
/// encodePiece does) must give the same answer; if it doesn't, encodePiece
/// has a bug.
func referenceEncodePiece(_ piece: String, merges: [Merge]) -> [Int] {
    var ids = piece.utf8.map { Int($0) }
    for m in merges { ids = BPETokenizer.applyMerge(ids, pair: m.pair, newId: m.result) }
    return ids
}

func runDemo() throws {
    print("=== 1. Training ===")
    let tok = BPETokenizer()
    let target = 512
    let learned = try tok.train(corpus: trainingCorpus, targetVocabSize: target)
    print("  corpus: \(trainingCorpus.count) chars, \(trainingCorpus.utf8.count) bytes")
    print("  targetVocabSize=\(target) (base 256 + merges); merges learned=\(learned) (stops early when no pair occurs >= 2 times)")
    print("  first 12 merges (rank: left + right -> new id):")
    for (rank, m) in tok.merges.prefix(12).enumerated() {
        print("    \(rank): \(tok.tokenDisplay(m.left)) + \(tok.tokenDisplay(m.right)) -> \(m.result) \(tok.tokenDisplay(m.result))")
    }
    check(tok.merges.enumerated().allSatisfy { $0.element.result == 256 + $0.offset }, "merge rank r has id 256 + r")

    let tok2 = BPETokenizer()
    try tok2.train(corpus: trainingCorpus, targetVocabSize: target)
    check(tok2.merges == tok.merges, "training is deterministic (two runs, identical merges)")

    print("\n=== 2. Special tokens ===")
    let bos = try tok.registerSpecialToken("[BOS]")
    let eos = try tok.registerSpecialToken("[EOS]")
    let pad = try tok.registerSpecialToken("[PAD]")
    print("  [BOS]=\(bos) [EOS]=\(eos) [PAD]=\(pad)  vocabSize=\(tok.vocabSize)")
    let firstSpecial = 256 + tok.merges.count
    check(bos == firstSpecial && eos == firstSpecial + 1 && pad == firstSpecial + 2,
          "special ids come right after the merges, no collision", "first special id \(firstSpecial)")
    check(try tok.registerSpecialToken("[EOS]") == eos, "re-registering a special returns the same id")
    var refused = false
    do { try tok.train(corpus: "abc", targetVocabSize: 300) } catch TokenizerError.trainingAfterSpecialTokens { refused = true }
    check(refused, "train() after registering specials is refused (protects ids)")

    print("\n=== 3. Encode / decode with specials ===")
    let sample = "[BOS]Hello world! The tokenizer's merges are replayed in order.\n  Emoji 👋 too.[EOS][PAD]"
    let ids = tok.encode(text: sample, allowedSpecial: tok.allSpecialTokens)
    print("  text: \(sample.debugDescription)")
    print("  ids (\(ids.count)): \(ids)")
    print("  tokens: " + ids.map { tok.tokenDisplay($0) }.joined(separator: " | "))
    let decoded = try tok.decode(ids)
    check(decoded == sample, "round trip with specials is exact")
    check(ids.first == bos && ids.suffix(2) == [eos, pad], "allowed specials map to single reserved ids")

    let idsNoSpecial = tok.encode(text: sample)   // specials NOT allowed
    check(!idsNoSpecial.contains(bos) && !idsNoSpecial.contains(eos) && !idsNoSpecial.contains(pad),
          "disallowed specials are encoded as plain text", "\(idsNoSpecial.count) ids")
    check(try tok.decode(idsNoSpecial) == sample, "round trip with specials disallowed is exact")

    let unk = "a [UNK] b"
    let unkIds = tok.encode(text: unk, allowedSpecial: ["[UNK]"])   // allowed but never registered
    let unkDecoded = try tok.decode(unkIds)
    check(unkIds == tok.encodeOrdinary(unk) && unkDecoded == unk,
          "allowed-but-unregistered special is encoded as text, not dropped")

    var threwUnknown = false
    do { _ = try tok.decode([999_999]) } catch TokenizerError.unknownTokenId { threwUnknown = true }
    check(threwUnknown, "decode of an unknown id throws instead of silently skipping")

    print("\n=== 4. Ranked encoding matches reference ===")
    var mismatches = 0
    let probe = tok.preTokenize(trainingCorpus + heldOutText + " unbelievable tokenization 1234567")
    for piece in Set(probe) where tok.encodeOrdinary(piece) != referenceEncodePiece(piece, merges: tok.merges) {
        mismatches += 1
    }
    check(mismatches == 0, "lowest-rank-first encoding == applying all merges in order",
          "\(Set(probe).count) distinct pieces checked")

    print("\n=== 5. Save / load JSON ===")
    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("bpe_tokenizer.json")
    try tok.saveToJSON(url: url)
    let fresh = BPETokenizer()
    try fresh.loadFromJSON(url: url)
    print("  saved to \(url.path) (\((try Data(contentsOf: url)).count) bytes)")
    check(fresh.merges == tok.merges, "loaded merges identical and in the same order")
    check(fresh.specialTokens == tok.specialTokens && fresh.nextTokenId == tok.nextTokenId && fresh.vocab == tok.vocab,
          "loaded vocab, specials and nextTokenId identical")
    check(fresh.encode(text: sample, allowedSpecial: fresh.allSpecialTokens) == ids, "loaded tokenizer produces identical ids")
    check(try fresh.jsonData() == tok.jsonData(), "saving is stable (byte-identical JSON on re-save)")
    let nextSpecial = try fresh.registerSpecialToken("[SEP]")
    check(nextSpecial == tok.nextTokenId, "after load, new specials continue from nextTokenId", "[SEP]=\(nextSpecial)")

    // Corrupt the file: change one merge's left id and expect a validation error.
    var json = try JSONSerialization.jsonObject(with: try tok.jsonData()) as! [String: Any]
    var mergeList = json["merges"] as! [[String: Any]]
    mergeList[0]["left"] = 65
    json["merges"] = mergeList
    var rejected = false
    let before = BPETokenizer()
    do { try before.load(jsonData: try JSONSerialization.data(withJSONObject: json)) }
    catch let e as TokenizerError { rejected = true; print("  corrupt file rejected: \(e)") }
    check(rejected && before.merges.isEmpty, "corrupted JSON is rejected and leaves the tokenizer untouched")

    print("\n=== 6. Overlapping special tokens (longest first) ===")
    let ov = BPETokenizer()
    try ov.loadFromJSON(url: url)
    let shortId = try ov.registerSpecialToken("<|end")
    let longId = try ov.registerSpecialToken("<|end|>")
    let ovIds = ov.encode(text: "a<|end|>b<|end", allowedSpecial: ["<|end", "<|end|>"])
    print("  ids: \(ovIds)  tokens: " + ovIds.map { ov.tokenDisplay($0) }.joined(separator: " | "))
    check(ovIds.filter { $0 == longId }.count == 1 && ovIds.filter { $0 == shortId }.count == 1,
          "\"<|end|>\" wins over its prefix \"<|end\"")
    check(try ov.decode(ovIds) == "a<|end|>b<|end", "overlapping specials round trip")

    print("\n=== 7. Edge cases (exact round trip) ===")
    let edgeCases: [(String, String)] = [
        ("empty string", ""),
        ("emoji 👋", "👋"),
        ("world's", "world's"),
        ("double spaces", "hello  world"),
        ("newline", "line one\nline two"),
        ("leading/trailing spaces", "   padded   "),
        ("only whitespace", " \n\t \n"),
        ("CJK", "你好，世界"),
        ("combining accent", "cafe\u{301}"),
        ("ZWJ family emoji + flag", "👨‍👩‍👧 🇺🇸"),
        ("digits and symbols", "CME 295: x^2 + y^2 = 1.0 (100%)"),
    ]
    for (name, text) in edgeCases {
        let e = tok.encode(text: text, allowedSpecial: tok.allSpecialTokens)
        let d = try tok.decode(e)
        check(d == text && Array(d.utf8) == Array(text.utf8), "round trip: \(name)", "\(text.utf8.count) bytes -> \(e.count) tokens")
    }
    check(tok.encode(text: "").isEmpty, "empty string encodes to []")
    check(tok.preTokenize("world's") == ["world", "'s"], "\"world's\" pre-tokenizes to [\"world\", \"'s\"]")
    check(tok.preTokenize("hello  world") == ["hello", " ", " world"], "double space keeps every space", "\(tok.preTokenize("hello  world"))")
    let wave = tok.encode(text: "👋")
    print("  👋 -> ids \(wave) tokens " + wave.map { tok.tokenDisplay($0) }.joined(separator: " | "))
    check(try tok.decode(Array(wave.prefix(1))).contains("\u{FFFD}") || wave.count == 1,
          "partial emoji bytes decode lossily to U+FFFD instead of crashing")

    print("\n=== 8. Compression benchmark ===")
    let sTrain = tok.benchmarkCompression(text: trainingCorpus, label: "training corpus")
    let sHeld = tok.benchmarkCompression(text: heldOutText, label: "held-out English")
    let sIntl = tok.benchmarkCompression(text: "你好，世界 👋 Grüße", label: "non-English + emoji")
    let sEmpty = tok.benchmarkCompression(text: "", label: "empty")
    check(sTrain.tokens < sTrain.bytes && sHeld.tokens < sHeld.bytes, "BPE uses fewer tokens than raw bytes")
    check(sEmpty.tokens == 0 && sEmpty.charsPerToken == 0 && !sEmpty.savingsPercent.isNaN, "empty benchmark: no divide-by-zero")
    _ = sIntl

    print("\n=== Summary ===")
    print("\(passCount) passed, \(failCount) failed")
}

do {
    try runDemo()
} catch {
    print("FAIL: unexpected error: \(error)")
    failCount += 1
}
exit(failCount == 0 ? 0 : 1)
