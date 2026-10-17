# Byte-Level BPE Tokenizer in Swift: Summary

`BPETokenizer.swift` merges your four versions (`ByteLevelTokenizer`, `AdvancedByteLevelTokenizer`, `FinalByteLevelTokenizer`, `ProductionBPETokenizer`) into one class, `BPETokenizer`. It works like GPT-2 and tiktoken: it starts from 256 byte tokens, splits text with the GPT-2 regex, learns **ordered** merges, encodes by always applying the lowest-rank merge first, handles special tokens, saves and loads stable JSON, and measures compression. Every step has a comment explaining *why* it is there.

I built it with **Swift 6.4.1** (`swiftc -O`) on Debian 13 x86_64 and ran it. **All 37 checks passed, with 0 failures.** The full output is in `run-output.txt`.

Files:
- `BPETokenizer.swift`: the tokenizer, plus a demo and self-test at the bottom
- `run-output.txt`: the real output from the run
- `bpe_tokenizer.json`: the model the demo saved

---

## 1. Bugs found and how they were fixed

| # | Bug | Version(s) that had it | Why it matters | Fix in final |
|---|-----|------------------------|----------------|--------------|
| 1 | Split on whitespace, so spaces were thrown away | ByteLevelTokenizer | `decode(encode(x)) != x`. The model sees text that differs from your input. | Split with the GPT-2 regex, which covers every character. `preTokenize` also keeps any gap the regex misses. Joining the pieces gives back the original exactly. |
| 2 | Merges stored in an unordered `[PairWrapper: [UInt8]]` dictionary | All four | BPE depends on order: "th" must form before "the". Swift dictionary order is random on each run, so encoding didn't follow the order training learned and the result could change between runs. | `merges: [Merge]` is an ordered array where index = rank. A `mergeRanks: [TokenPair: Int]` table gives O(1) lookup. Encoding keeps merging the adjacent pair with the lowest rank until no pair matches. A test confirms this equals applying every merge in order. |
| 3 | Ties during training broken by dictionary order | All four | The same corpus could produce different merges and IDs on different runs. | Ties go to the pair whose bytes come first lexicographically (left token's bytes, then right's). If that's still tied, compare IDs. Test: two separate training runs give identical merges. |
| 4 | Pairs keyed by byte arrays | All four | Hashing arrays is slow. Also the bytes "abc" can come from more than one pair of tokens. | `TokenPair` holds two token IDs (Ints), which is cheap and unambiguous. |
| 5 | Special-token IDs could collide with merge IDs, and vocab size was unclear | ProductionBPETokenizer | Two meanings for one ID breaks the embedding table. | Documented layout: 0–255 are bytes, `256 + rank` are merges, specials come after. `targetVocabSize` counts only base + merges. Calling `train()` after registering specials throws an error. Registering the same special twice returns its existing ID. |
| 6 | Allowed but unregistered special tokens were silently dropped | ProductionBPETokenizer | Text disappears. | They are encoded as ordinary text, so the round trip stays exact (tested with `[UNK]`). |
| 7 | Special-token regex not ordered longest-first | ProductionBPETokenizer | With `<|end` and `<|end|>` both registered, the short one matched first and left `|>` behind as junk. | Specials are sorted longest-first before building the alternation, and the compiled regex is cached. Tested. |
| 8 | Benchmark divided by zero on empty input | ProductionBPETokenizer | NaN or infinity in the report. | Guards report 0 for the ratios, with a note. Tested. |
| 9 | `loadFromJSON` didn't fully restore state | FinalByteLevelTokenizer, ProductionBPETokenizer | After loading, encoding was wrong or new IDs collided. | It restores vocab, ordered merges, ranks, specials and `nextTokenId`. The vocab is rebuilt from the merges and must match the stored one. It also checks ranks, IDs, duplicates, special-ID ranges and `nextTokenId`. Everything is validated before it is applied, so a corrupt file leaves the tokenizer unchanged (tested). |
| 10 | JSON wasn't stable (dictionary key order) | FinalByteLevelTokenizer, ProductionBPETokenizer | Noisy diffs, and checksums didn't match. | Vocab and specials are saved as arrays sorted by ID, merges in rank order, with `.sortedKeys`. Saving again gives byte-identical JSON (tested). |
| 11 | Regex had no optional leading space (improvement, not a crash bug) | AdvancedByteLevelTokenizer onward | Every space was its own piece, so " the" could never become one token and compression was worse. | Uses the real GPT-2 pattern (` ?\p{L}+` …) together with `\s+(?!\S)`. Double spaces still round-trip: `"hello  world"` → `["hello", " ", " world"]`. |
| 12 | Encoding repeated the same work for every occurrence of a piece | All four | Slow on real text. | A per-piece cache (bounded at 50k entries) that is cleared whenever merges change. |
| 13 | Unknown IDs in decode weren't handled clearly | All four | Text could be silently lost. | `decode` throws `TokenizerError.unknownTokenId`. |

## 2. Features per version vs final

| Feature | ByteLevel | Advanced | Final | Production | **BPETokenizer (final)** |
|---------|:-:|:-:|:-:|:-:|:-:|
| 256 base byte tokens | ✅ | ✅ | ✅ | ✅ | ✅ |
| Train by merging the most frequent pair | ✅ | ✅ | ✅ | ✅ | ✅ (weighted piece counts) |
| Lossless pre-tokenization | ❌ (whitespace split) | ✅ regex | ✅ | ✅ | ✅ GPT-2 regex with leading space, plus a gap guard |
| Merges only inside pieces | ❌ | ✅ | ✅ | ✅ | ✅ |
| Ordered merges and rank-based encoding | ❌ | ❌ | ❌ | ❌ | ✅ |
| Deterministic tie-break | ❌ | ❌ | ❌ | ❌ | ✅ |
| Lossy UTF-8 decode (U+FFFD) | ✅ | ✅ | ✅ | ✅ | ✅ (joins all bytes first, then decodes) |
| Save/load JSON | ❌ | ❌ | ✅ (unordered) | ✅ (unordered) | ✅ ordered, stable, validated, atomic write |
| Special tokens | ❌ | ❌ | ❌ | ✅ | ✅ no collisions, idempotent, longest-first |
| `allowedSpecial` opt-in | ❌ | ❌ | ❌ | ✅ | ✅ (unregistered → text) |
| Compression benchmark | ❌ | ❌ | ❌ | ✅ | ✅ (empty input guarded; returns stats) |
| Encoding cache | ❌ | ❌ | ❌ | ❌ | ✅ |
| Typed errors | ❌ | ❌ | ❌ | ❌ | ✅ `TokenizerError` |
| Built-in self-test | ❌ | ❌ | ❌ | ❌ | ✅ 37 PASS/FAIL checks |

## 3. Public API

| Method / property | Purpose | Input | Output |
|---|---|---|---|
| `init()` | New tokenizer with only the 256 byte tokens | – | `BPETokenizer` |
| `train(corpus:targetVocabSize:minPairFrequency:)` | Learn ordered merges | corpus `String`; target size (base + merges, ≥ 256); minimum pair count (default 2) | merges learned (`Int`). Throws if specials are already registered or the target is < 256. |
| `registerSpecialToken(_:)` | Add a token like `[BOS]` | non-empty `String` | its ID (`Int`). Returns the existing ID if already registered. |
| `encode(text:allowedSpecial:)` | Text → IDs; listed specials become single IDs | `String`, `Set<String>` (default empty) | `[Int]` |
| `encodeOrdinary(_:)` | Text → IDs, treating specials as plain text | `String` | `[Int]` |
| `decode(_:)` | IDs → text (lossy UTF-8) | `[Int]` | `String`. Throws on an unknown ID. |
| `decodeBytes(_:)` | IDs → raw bytes | `[Int]` | `[UInt8]` |
| `preTokenize(_:)` | Show the regex split | `String` | `[String]` |
| `tokenDisplay(_:)` | Readable view of one token | ID `Int` | `String` such as `" the"`, `<F0>`, `[EOS]` |
| `saveToJSON(url:)` / `jsonData()` | Write the stable JSON model | `URL` / – | – / `Data` |
| `loadFromJSON(url:)` / `load(jsonData:)` | Load and validate a model | `URL` / `Data` | – (throws `invalidModel`) |
| `benchmarkCompression(text:label:allowedSpecial:)` | Print and return compression stats | `String`, label, specials | `CompressionStats` |
| `vocab`, `merges`, `specialTokens`, `nextTokenId`, `vocabSize`, `allSpecialTokens` | Read-only state | – | – |
| `static gpt2SplitPattern`, `baseVocabSize` | Constants | – | – |

## 4. How the pipeline works

| Step | What happens | Example | Why it exists |
|---|---|---|---|
| 0. Split out specials | Allowed special strings are cut out first and mapped to their reserved IDs. The longest match wins. | `"[BOS]Hello"` → `[BOS]`, `"Hello"` | Control signals must be one ID that normal text can't produce by accident. |
| 1. Pre-tokenize | GPT-2 regex splits the text into words, numbers, punctuation and whitespace. | `"Hello world's"` → `"Hello"`, `" world"`, `"'s"` | Stops merges from crossing word boundaries, so the vocabulary is spent on real word pieces. |
| 2. Bytes | Each piece → UTF-8 bytes → base IDs 0–255. | `"👋"` → `[240, 159, 145, 139]` | Any text can be represented, so there are no unknown tokens. |
| 3. Merges | Keep merging the adjacent pair with the lowest rank until none applies (with caching). | `" " + "t"` → `" t"`, then `" t" + "he"` → `" the"` | Replays training's order, giving short, reproducible sequences. |
| 4. IDs | Concatenate the IDs of all pieces and specials. | `[431, 72, 101, …, 432, 433]` | This is what the model's embedding table consumes. |
| 5. Decode | ID → bytes (specials → their text), join everything, then decode UTF-8 once. | `<F0><9F><91><8B>` → `👋` | Tokens can split a character, so decoding happens on the whole byte stream. Invalid bytes become `�` instead of crashing. |

## 5. Results from the run

I trained on the embedded corpus (1,591 chars, 4 paragraphs) with `targetVocabSize = 512`. Training **learned 175 merges** and then stopped early because no remaining pair appeared twice. Specials: `[BOS]=431`, `[EOS]=432`, `[PAD]=433`, so `vocabSize=434`. The first merges were `" t"`, `"he"`, `"in"`, `"er"`, `" a"`, `" the"` …

| Text | Chars | Bytes | Tokens | Chars/token | Bytes/token | Savings vs raw bytes |
|---|---:|---:|---:|---:|---:|---:|
| Training corpus | 1591 | 1591 | 581 | 2.74 | 2.74 | 63.5% |
| Held-out English | 224 | 224 | 97 | 2.31 | 2.31 | 56.7% |
| Non-English + emoji (`你好，世界 👋 Grüße`) | 13 | 28 | 28 | 0.46 | 1.00 | 0.0% |
| Empty string | 0 | 0 | 0 | 0.00 | 0.00 | 0.0% (guarded) |

What the numbers show: the held-out text compresses less than the training text, which is normal because merges fit the data they were learned from. Text unlike the corpus (CJK, emoji) gets no merges at all and costs one token per byte. This is why tokenizers trained mostly on English are "expensive" for other languages. GPT-2 reaches about 4 chars/token because it has 50k merges learned from gigabytes of text, compared with 175 merges here.

## 6. How to use

```swift
let tok = BPETokenizer()
try tok.train(corpus: myCorpus, targetVocabSize: 1000)   // 256 bytes + up to 744 merges
let bos = try tok.registerSpecialToken("[BOS]")          // register specials AFTER training
let eos = try tok.registerSpecialToken("[EOS]")

let ids  = tok.encode(text: "[BOS]Hello world 👋[EOS]", allowedSpecial: tok.allSpecialTokens)
print(ids.map(tok.tokenDisplay))
let text = try tok.decode(ids)                           // == original text

try tok.saveToJSON(url: URL(fileURLWithPath: "tok.json"))
let again = BPETokenizer()
try again.loadFromJSON(url: URL(fileURLWithPath: "tok.json"))  // same ids as tok

tok.benchmarkCompression(text: someText, label: "my text")
```

Build: `swiftc -O BPETokenizer.swift -o tok && ./tok`. To use it as a library, delete the demo section below `// MARK: - Demo & self-test`.
