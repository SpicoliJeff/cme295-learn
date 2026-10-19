// =============================================================================
//  MyAI.swift - build your own tiny AI that learns from what you type
//
//  A small text AI (a decoder-only Transformer, the same design as GPT) in one
//  file of pure Swift + Foundation. No libraries, no GPU: every piece, even
//  the calculus that makes it learn (backpropagation), is written out by hand.
//
//  Build:   swiftc -O MyAI.swift -o myai
//  Run:     ./myai                    self checks + train on built-in text + demo
//           ./myai train notes.txt    train a fresh AI on your own text file
//           ./myai generate "my cat"  continue a prompt
//           ./myai teach "my dog is named biscuit."   teach it one sentence
//           ./myai chat               type lines; it learns each one, then replies
//           options: --steps N  --seed N  --temp T
//
//  | Step | What it does                                   | CME 295 concept               |
//  |------|------------------------------------------------|-------------------------------|
//  |  1   | seeded random numbers (same seed = same run)   | reproducibility               |
//  |  2   | byte-level BPE tokenizer: text <-> token IDs   | tokenization, BPE, vocabulary |
//  |  3   | math building blocks: Linear, LayerNorm, GELU  | linear layer, LayerNorm, FFN  |
//  |  4   | the model: embeddings, Transformer blocks      | embeddings, decoder-only      |
//  |  5   | forward pass, softmax, cross-entropy loss      | self-attention, causal mask   |
//  |  6   | backward pass (hand-written gradients)         | backpropagation               |
//  |  7   | Adam optimizer with gradient clipping          | Adam, learning rate           |
//  |  8   | training loop on random text windows           | next-token prediction         |
//  |  9   | generation with temperature and top-k          | decoding, temperature, top-k  |
//  | 10   | save / load weights as JSON                    | checkpoints                   |
//  | 11   | learning from new inputs (online fine-tuning)  | fine-tuning, forgetting       |
//  | 12   | self checks that print PASS or FAIL            | gradient check, evaluation    |
//  | 13   | built-in training text + commands              | training data                 |
//
//  Honest note: this AI is tiny (about 0.1 million numbers; ChatGPT-class
//  models have billions). It learns patterns and memorizes what you teach
//  it. It does not understand or reason.
// =============================================================================

import Foundation

// MARK: - Step 1: Random numbers (seeded)
// A computer's "random" numbers come from a formula. Starting the formula from
// the same seed gives the same numbers, so every run can be repeated exactly.

struct RNG {
    var state: UInt64
    init(seed: UInt64) { state = seed }

    // SplitMix64: add a constant, then scramble the bits.
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func uniform() -> Double { Double(next() >> 11) * 0x1.0p-53 }   // 0 <= x < 1
    mutating func below(_ n: Int) -> Int { Int(next() % UInt64(n)) }          // 0 ..< n

    // Bell-curve number (mean 0, spread 1) using the Box-Muller formula.
    mutating func gaussian() -> Double {
        let u1 = max(uniform(), 1e-12), u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
    }
}

// MARK: - Step 2: Tokenizer (byte-level BPE)
// The model only reads numbers. IDs 0-255 are the raw bytes of the text, so any
// text works. Training then repeatedly glues the most common neighbouring pair
// into a new token (256, 257, ...), so common words become single tokens.

struct Tokenizer {
    let merges: [[Int]]                               // merges[r] = [left, right] -> token 256 + r
    var pieces: [[UInt8]] = (0..<256).map { [UInt8($0)] }   // the bytes each token stands for
    var rank: [Int: Int] = [:]                        // pair -> merge number (lower = learned first)
    var size: Int { pieces.count }

    static func pairKey(_ a: Int, _ b: Int) -> Int { a * 100_000 + b }

    init(merges: [[Int]]) {
        self.merges = merges
        for (r, m) in merges.enumerated() {
            pieces.append(pieces[m[0]] + pieces[m[1]])
            rank[Tokenizer.pairKey(m[0], m[1])] = r
        }
    }

    // Split text into word-sized chunks like "my", " dog", "."; merges never cross chunks.
    static func chunks(_ text: String) -> [[Int]] {
        func isWordByte(_ b: Int) -> Bool { (b >= 97 && b <= 122) || (b >= 65 && b <= 90) || (b >= 48 && b <= 57) || b >= 128 }
        var out: [[Int]] = [], current: [Int] = []
        for byte in text.utf8 {
            let b = Int(byte)
            if let last = current.last, !isWordByte(b) || (!isWordByte(last) && last != 32) {
                out.append(current)
                current = []
            }
            current.append(b)
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    // Replace every neighbouring (a, b) in a chunk with the new token id.
    static func merge(_ ids: [Int], _ a: Int, _ b: Int, _ newID: Int) -> [Int] {
        var out: [Int] = [], i = 0
        while i < ids.count {
            if i + 1 < ids.count && ids[i] == a && ids[i + 1] == b {
                out.append(newID)
                i += 2
            } else {
                out.append(ids[i])
                i += 1
            }
        }
        return out
    }

    // Learn `count` merges from the text: count pairs, glue the most common, repeat.
    static func train(_ text: String, merges count: Int) -> Tokenizer {
        var words: [[Int]: Int] = [:]
        for c in chunks(text) { words[c, default: 0] += 1 }
        var merges: [[Int]] = []
        for _ in 0..<count {
            var pairCounts: [Int: Int] = [:]
            for (w, n) in words where w.count > 1 {
                for i in 0..<(w.count - 1) { pairCounts[pairKey(w[i], w[i + 1]), default: 0] += n }
            }
            // Most frequent pair; ties go to the smaller key so the result never changes.
            guard let best = pairCounts.max(by: { $0.value != $1.value ? $0.value < $1.value : $0.key > $1.key }),
                  best.value >= 2 else { break }
            let a = best.key / 100_000, b = best.key % 100_000
            let newID = 256 + merges.count
            merges.append([a, b])
            var updated: [[Int]: Int] = [:]
            for (w, n) in words { updated[merge(w, a, b, newID), default: 0] += n }
            words = updated
        }
        return Tokenizer(merges: merges)
    }

    // Text -> IDs: in each chunk, keep applying the earliest-learned merge that fits.
    func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        for var w in Tokenizer.chunks(text) {
            while w.count > 1 {
                var bestRank = Int.max
                for i in 0..<(w.count - 1) {
                    if let r = rank[Tokenizer.pairKey(w[i], w[i + 1])], r < bestRank { bestRank = r }
                }
                if bestRank == Int.max { break }
                w = Tokenizer.merge(w, merges[bestRank][0], merges[bestRank][1], 256 + bestRank)
            }
            ids += w
        }
        return ids
    }

    // IDs -> text: glue the bytes back together (unknown IDs become "?").
    func decode(_ ids: [Int]) -> String {
        let bytes = ids.flatMap { $0 >= 0 && $0 < size ? pieces[$0] : [63] }
        return String(decoding: bytes, as: UTF8.self)
    }
}

// MARK: - Step 3: Math building blocks
// Vectors are plain arrays of Double. A batch of T vectors of width n is one
// flat array of T*n numbers, row t starting at index t*n. Each layer has a
// forward (compute) and a backward (gradient) function side by side.

func zeros(_ n: Int) -> [Double] { [Double](repeating: 0, count: n) }

func add(_ a: [Double], _ b: [Double]) -> [Double] { zip(a, b).map { $0 + $1 } }

func addInPlace(_ a: inout [Double], _ b: [Double]) { for i in a.indices { a[i] += b[i] } }

// Softmax: turn any scores into positive probabilities that add up to 1.
func softmax(_ x: [Double]) -> [Double] {
    let m = x.max() ?? 0                              // subtract the max so exp() cannot overflow
    let e = x.map { exp($0 - m) }
    let total = e.reduce(0, +)
    return e.map { $0 / total }
}

// GELU: a smooth "keep positives, squash negatives" curve, and its slope.
func gelu(_ x: Double) -> Double { 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))) }

func geluSlope(_ x: Double) -> Double {
    let th = tanh(0.7978845608 * (x + 0.044715 * x * x * x))
    return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * 0.7978845608 * (1 + 3 * 0.044715 * x * x)
}

// A Param is a list of learnable numbers (w) plus their gradients (g).
final class Param {
    var w: [Double]
    var g: [Double]
    init(_ count: Int, std: Double, fill: Double = 0, rng: inout RNG) {
        w = (0..<count).map { _ in std > 0 ? rng.gaussian() * std : fill }
        g = zeros(count)
    }
}

// Linear layer: y = x * W + b (each output is a weighted sum of the inputs).
final class Linear {
    let W: Param, b: Param, nIn: Int, nOut: Int
    init(_ nIn: Int, _ nOut: Int, rng: inout RNG) {
        self.nIn = nIn
        self.nOut = nOut
        W = Param(nIn * nOut, std: 1 / Double(nIn).squareRoot(), rng: &rng)
        b = Param(nOut, std: 0, rng: &rng)
    }
    var params: [Param] { [W, b] }

    func forward(_ x: [Double], _ T: Int) -> [Double] {
        let w = W.w                                   // local copy name: faster to read
        var y = zeros(T * nOut)
        for t in 0..<T {
            for o in 0..<nOut { y[t * nOut + o] = b.w[o] }
            for i in 0..<nIn {
                let xi = x[t * nIn + i]
                for o in 0..<nOut { y[t * nOut + o] += xi * w[i * nOut + o] }
            }
        }
        return y
    }

    // Given dLoss/dy, add dLoss/dW and dLoss/db to the gradients, return dLoss/dx.
    func backward(_ dy: [Double], _ x: [Double], _ T: Int) -> [Double] {
        let w = W.w
        var dx = zeros(T * nIn), dw = zeros(nIn * nOut), db = zeros(nOut)
        for t in 0..<T {
            for o in 0..<nOut { db[o] += dy[t * nOut + o] }
            for i in 0..<nIn {
                let xi = x[t * nIn + i]
                var sum = 0.0
                for o in 0..<nOut {
                    sum += dy[t * nOut + o] * w[i * nOut + o]
                    dw[i * nOut + o] += xi * dy[t * nOut + o]
                }
                dx[t * nIn + i] = sum
            }
        }
        addInPlace(&W.g, dw)
        addInPlace(&b.g, db)
        return dx
    }
}

// LayerNorm: rescale each vector to average 0 and spread 1, then apply a
// learned scale (g) and shift (b). This keeps numbers in a healthy range.
struct NormCache { let xhat: [Double]; let rstd: [Double] }

final class LayerNorm {
    let g: Param, b: Param, n: Int
    init(_ n: Int, rng: inout RNG) {
        self.n = n
        g = Param(n, std: 0, fill: 1, rng: &rng)
        b = Param(n, std: 0, rng: &rng)
    }
    var params: [Param] { [g, b] }

    func forward(_ x: [Double], _ T: Int) -> ([Double], NormCache) {
        var y = zeros(T * n), xhat = zeros(T * n), rstd = zeros(T)
        for t in 0..<T {
            let row = x[(t * n)..<((t + 1) * n)]
            let mean = row.reduce(0, +) / Double(n)
            let variance = row.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(n)
            rstd[t] = 1 / (variance + 1e-5).squareRoot()
            for i in 0..<n {
                xhat[t * n + i] = (x[t * n + i] - mean) * rstd[t]
                y[t * n + i] = g.w[i] * xhat[t * n + i] + b.w[i]
            }
        }
        return (y, NormCache(xhat: xhat, rstd: rstd))
    }

    func backward(_ dy: [Double], _ c: NormCache, _ T: Int) -> [Double] {
        var dx = zeros(T * n)
        for t in 0..<T {
            var dxhat = zeros(n), meanD = 0.0, meanDX = 0.0
            for i in 0..<n {
                let k = t * n + i
                g.g[i] += dy[k] * c.xhat[k]
                b.g[i] += dy[k]
                dxhat[i] = dy[k] * g.w[i]
                meanD += dxhat[i] / Double(n)
                meanDX += dxhat[i] * c.xhat[k] / Double(n)
            }
            for i in 0..<n { dx[t * n + i] = c.rstd[t] * (dxhat[i] - meanD - c.xhat[t * n + i] * meanDX) }
        }
        return dx
    }
}

// MARK: - Step 4: The model (embeddings + Transformer blocks + output head)
// Token IDs become vectors (embeddings). Each Transformer block lets every
// position look back at earlier positions (attention), then "think" on its own
// (MLP). The head turns the final vector into one score per vocabulary token.

struct Config: Codable, Equatable {
    var vocab: Int                                    // number of different tokens
    var dim = 48                                      // width of every vector
    var layers = 2                                    // number of Transformer blocks
    var context = 32                                  // how many tokens it can see at once
}

final class Attention {
    let q: Linear, k: Linear, v: Linear, o: Linear   // query, key, value, output
    init(_ d: Int, rng: inout RNG) {
        q = Linear(d, d, rng: &rng)
        k = Linear(d, d, rng: &rng)
        v = Linear(d, d, rng: &rng)
        o = Linear(d, d, rng: &rng)
    }
    var params: [Param] { q.params + k.params + v.params + o.params }
}

final class Block {
    let ln1: LayerNorm, attn: Attention, ln2: LayerNorm, fc1: Linear, fc2: Linear
    init(_ d: Int, rng: inout RNG) {
        ln1 = LayerNorm(d, rng: &rng)
        attn = Attention(d, rng: &rng)
        ln2 = LayerNorm(d, rng: &rng)
        fc1 = Linear(d, 4 * d, rng: &rng)            // MLP widens to 4x ...
        fc2 = Linear(4 * d, d, rng: &rng)            // ... then narrows back
    }
    var params: [Param] { ln1.params + attn.params + ln2.params + fc1.params + fc2.params }
}

final class Model {
    let config: Config
    let tokenEmb: Param                               // vocab x dim: one vector per token
    let posEmb: Param                                 // context x dim: one vector per position
    let blocks: [Block]
    let lnFinal: LayerNorm
    let head: Linear                                  // dim -> vocab scores (logits)

    init(_ config: Config, seed: UInt64) {
        var rng = RNG(seed: seed)
        self.config = config
        tokenEmb = Param(config.vocab * config.dim, std: 0.1, rng: &rng)
        posEmb = Param(config.context * config.dim, std: 0.1, rng: &rng)
        blocks = (0..<config.layers).map { _ in Block(config.dim, rng: &rng) }
        lnFinal = LayerNorm(config.dim, rng: &rng)
        head = Linear(config.dim, config.vocab, rng: &rng)
    }

    var params: [Param] { [tokenEmb, posEmb] + blocks.flatMap { $0.params } + lnFinal.params + head.params }
    var parameterCount: Int { params.reduce(0) { $0 + $1.w.count } }
}

// MARK: - Step 5: Forward pass, softmax, cross-entropy loss
// The forward pass runs the tokens through every layer and saves what it saw
// (the cache), because the backward pass needs those numbers. The loss says
// how surprised the model was by each true next token: -log(probability).

struct AttnCache { let x, q, k, v, att, mixed: [Double] }

extension Attention {
    func forward(_ x: [Double], _ T: Int) -> ([Double], AttnCache) {
        let d = q.nIn, scale = 1 / Double(q.nIn).squareRoot()
        let Q = q.forward(x, T), K = k.forward(x, T), V = v.forward(x, T)
        var att = zeros(T * T), mixed = zeros(T * d)
        for t in 0..<T {
            var scores = zeros(t + 1)                 // CAUSAL MASK: only positions s <= t
            for s in 0...t {
                for j in 0..<d { scores[s] += Q[t * d + j] * K[s * d + j] }
                scores[s] *= scale
            }
            let weights = softmax(scores)             // how much t looks at each s
            for s in 0...t {
                att[t * T + s] = weights[s]
                for j in 0..<d { mixed[t * d + j] += weights[s] * V[s * d + j] }
            }
        }
        return (o.forward(mixed, T), AttnCache(x: x, q: Q, k: K, v: V, att: att, mixed: mixed))
    }
}

struct BlockCache { let n1: [Double], c1: NormCache, a: AttnCache, n2: [Double], c2: NormCache, h: [Double], hg: [Double] }

extension Block {
    func forward(_ x: [Double], _ T: Int) -> ([Double], BlockCache) {
        let (n1, c1) = ln1.forward(x, T)
        let (a, ac) = attn.forward(n1, T)
        let x2 = add(x, a)                            // residual: keep x, add what attention found
        let (n2, c2) = ln2.forward(x2, T)
        let h = fc1.forward(n2, T)
        let hg = h.map(gelu)
        let y = add(x2, fc2.forward(hg, T))           // residual again around the MLP
        return (y, BlockCache(n1: n1, c1: c1, a: ac, n2: n2, c2: c2, h: h, hg: hg))
    }
}

struct Cache { let tokens: [Int]; let blocks: [BlockCache]; let cf: NormCache; let final: [Double] }

extension Model {
    // tokens (at most `context` of them) -> logits: T rows of `vocab` scores.
    func forward(_ tokens: [Int]) -> ([Double], Cache) {
        let T = tokens.count, d = config.dim
        precondition(T >= 1 && T <= config.context, "need 1...context tokens")
        var x = zeros(T * d)
        for t in 0..<T {
            for j in 0..<d { x[t * d + j] = tokenEmb.w[tokens[t] * d + j] + posEmb.w[t * d + j] }
        }
        var caches: [BlockCache] = []
        for block in blocks {
            let (y, c) = block.forward(x, T)
            caches.append(c)
            x = y
        }
        let (f, cf) = lnFinal.forward(x, T)
        return (head.forward(f, T), Cache(tokens: tokens, blocks: caches, cf: cf, final: f))
    }
}

// Average -log P(target) over all positions, and dLoss/dlogits = (p - onehot) / T.
func crossEntropy(_ logits: [Double], _ targets: [Int], _ V: Int) -> (loss: Double, dLogits: [Double]) {
    let T = targets.count
    var loss = 0.0, d = zeros(T * V)
    for t in 0..<T {
        let p = softmax(Array(logits[(t * V)..<((t + 1) * V)]))
        loss -= log(max(p[targets[t]], 1e-12))
        for i in 0..<V { d[t * V + i] = p[i] / Double(T) }
        d[t * V + targets[t]] -= 1 / Double(T)
    }
    return (loss / Double(T), d)
}

// MARK: - Step 6: Backprop (hand-written gradients)
// The chain rule, run backwards: starting from dLoss/dlogits, every layer turns
// "how the loss changes with my output" into "how it changes with my input"
// and with my weights. The weight gradients tell Adam which way to nudge.

extension Attention {
    func backward(_ dOut: [Double], _ c: AttnCache, _ T: Int) -> [Double] {
        let d = q.nIn, scale = 1 / Double(q.nIn).squareRoot()
        let dMixed = o.backward(dOut, c.mixed, T)
        var dQ = zeros(T * d), dK = zeros(T * d), dV = zeros(T * d)
        for t in 0..<T {
            var dW = zeros(t + 1)                     // gradient for each attention weight
            for s in 0...t {
                for j in 0..<d {
                    dW[s] += dMixed[t * d + j] * c.v[s * d + j]
                    dV[s * d + j] += c.att[t * T + s] * dMixed[t * d + j]
                }
            }
            var avg = 0.0                             // softmax backward needs sum(w * dW)
            for s in 0...t { avg += c.att[t * T + s] * dW[s] }
            for s in 0...t {
                let dScore = c.att[t * T + s] * (dW[s] - avg) * scale
                for j in 0..<d {
                    dQ[t * d + j] += dScore * c.k[s * d + j]
                    dK[s * d + j] += dScore * c.q[t * d + j]
                }
            }
        }
        var dx = q.backward(dQ, c.x, T)
        addInPlace(&dx, k.backward(dK, c.x, T))
        addInPlace(&dx, v.backward(dV, c.x, T))
        return dx
    }
}

extension Block {
    func backward(_ dy: [Double], _ c: BlockCache, _ T: Int) -> [Double] {
        let dhg = fc2.backward(dy, c.hg, T)
        let dh = zip(dhg, c.h).map { $0 * geluSlope($1) }
        var dx2 = ln2.backward(fc1.backward(dh, c.n2, T), c.c2, T)
        addInPlace(&dx2, dy)                          // the residual path passes dy straight through
        var dx = ln1.backward(attn.backward(dx2, c.a, T), c.c1, T)
        addInPlace(&dx, dx2)
        return dx
    }
}

extension Model {
    func zeroGrads() { for p in params { p.g = zeros(p.g.count) } }

    // Adds this sequence's gradients into every Param's g.
    func backward(_ dLogits: [Double], _ c: Cache) {
        let T = c.tokens.count, d = config.dim
        var dx = lnFinal.backward(head.backward(dLogits, c.final, T), c.cf, T)
        for (block, bc) in zip(blocks, c.blocks).reversed() { dx = block.backward(dx, bc, T) }
        for t in 0..<T {
            for j in 0..<d {
                tokenEmb.g[c.tokens[t] * d + j] += dx[t * d + j]
                posEmb.g[t * d + j] += dx[t * d + j]
            }
        }
    }
}

// MARK: - Step 7: Adam optimizer
// Plain gradient descent takes w -= lr * g. Adam also keeps a running average
// of each gradient (m) and of its square (v), so every weight gets its own
// sensible step size. Clipping stops one bad batch from making a huge jump.

struct Adam {
    var lr: Double
    let beta1 = 0.9, beta2 = 0.999, eps = 1e-8, clip = 1.0
    var m: [[Double]], v: [[Double]], t = 0

    init(_ params: [Param], lr: Double) {
        self.lr = lr
        m = params.map { zeros($0.w.count) }
        v = m
    }

    mutating func step(_ params: [Param]) {
        t += 1
        let norm = params.reduce(0) { $0 + $1.g.reduce(0) { $0 + $1 * $1 } }.squareRoot()
        let shrink = norm > clip ? clip / norm : 1    // gradient clipping
        let c1 = 1 - pow(beta1, Double(t)), c2 = 1 - pow(beta2, Double(t))
        for (n, p) in params.enumerated() {
            for i in p.w.indices {
                let g = p.g[i] * shrink
                m[n][i] = beta1 * m[n][i] + (1 - beta1) * g
                v[n][i] = beta2 * v[n][i] + (1 - beta2) * g * g
                p.w[i] -= lr * (m[n][i] / c1) / ((v[n][i] / c2).squareRoot() + eps)
            }
        }
    }
}

// MARK: - Step 8: Training loop
// Pick random windows of text. The input is tokens 0..T-1 and the target is
// the same window shifted by one (tokens 1..T): at every position the model
// practises predicting the next token. Average gradients, take an Adam step.

typealias Example = (inputs: [Int], targets: [Int])

func randomWindow(_ data: [Int], _ T: Int, _ rng: inout RNG) -> Example {
    let start = rng.below(data.count - T)
    return (Array(data[start..<(start + T)]), Array(data[(start + 1)...(start + T)]))
}

// One Adam step on a batch; returns the average loss.
func trainStep(_ model: Model, _ opt: inout Adam, _ batch: [Example]) -> Double {
    model.zeroGrads()
    var total = 0.0
    for ex in batch {
        let (logits, cache) = model.forward(ex.inputs)
        let (loss, d) = crossEntropy(logits, ex.targets, model.config.vocab)
        model.backward(d.map { $0 / Double(batch.count) }, cache)
        total += loss
    }
    opt.step(model.params)
    return total / Double(batch.count)
}

// Average loss on 24 fixed windows (same windows every time, so numbers compare).
func evalLoss(_ model: Model, _ data: [Int]) -> Double {
    var rng = RNG(seed: 999)
    let T = model.config.context
    return (0..<24).map { _ -> Double in
        let ex = randomWindow(data, T, &rng)
        return crossEntropy(model.forward(ex.inputs).0, ex.targets, model.config.vocab).loss
    }.reduce(0, +) / 24
}

func train(_ model: Model, _ data: [Int], steps: Int, batch: Int = 8, lr: Double = 0.003, seed: UInt64) -> [Double] {
    var rng = RNG(seed: seed)
    var opt = Adam(model.params, lr: lr)
    var losses: [Double] = []
    for step in 1...steps {
        opt.lr = lr * (0.1 + 0.9 * 0.5 * (1 + cos(Double.pi * Double(step) / Double(steps))))  // cosine decay
        let examples = (0..<batch).map { _ in randomWindow(data, model.config.context, &rng) }
        losses.append(trainStep(model, &opt, examples))
        if step % 100 == 0 || step == 1 { print(String(format: "  step %4d/%d  loss %.3f", step, steps, losses.last!)) }
    }
    return losses
}

// MARK: - Step 9: Generation (temperature + guarded top-k)
// To write text: run the model, turn the last position's scores into
// probabilities, pick one token, append it, repeat. Temperature below 1 makes
// safe choices, above 1 makes wild ones. Top-k only allows the k best tokens.

func sampleTopK(_ probs: [Double], k: Int, temperature: Double, rng: inout RNG) -> Int? {
    guard !probs.isEmpty else { return nil }
    let valid = probs.enumerated().filter { $0.element > 0 && $0.element.isFinite }
    guard let maxP = valid.map({ $0.element }).max() else { return 0 }    // all zero or NaN
    if temperature <= 0 { return valid.first { $0.element == maxP }!.offset }  // greedy
    var cands = valid.map { (id: $0.offset, w: exp((log($0.element) - log(maxP)) / temperature)) }
    cands.sort { $0.w != $1.w ? $0.w > $1.w : $0.id < $1.id }
    let top = cands.prefix(k <= 0 ? cands.count : min(k, cands.count))
    let total = top.reduce(0.0) { $0 + $1.w }
    var roll = rng.uniform() * total
    for c in top {
        roll -= c.w
        if roll < 0 { return c.id }
    }
    return top.last!.id                               // rounding fallback, never crash
}

func generate(_ model: Model, _ tok: Tokenizer, _ prompt: String, temperature: Double = 0.7,
              topK: Int = 8, maxTokens: Int = 30, rng: inout RNG) -> String {
    var ids = tok.encode("\n" + prompt)              // a line start, then the prompt
    var out: [Int] = []
    let V = model.config.vocab
    for _ in 0..<maxTokens {
        let window = Array(ids.suffix(model.config.context))
        let logits = model.forward(window).0
        let last = Array(logits[((window.count - 1) * V)..<(window.count * V)])
        guard let next = sampleTopK(softmax(last), k: topK, temperature: temperature, rng: &rng),
              next != 10 else { break }               // token 10 is "\n": the line is finished
        ids.append(next)
        out.append(next)
    }
    return prompt + tok.decode(out)
}

// MARK: - Step 10: Save / load weights
// Everything needed to rebuild the AI goes into one JSON file: the settings,
// the tokenizer merges, every weight, and the memory (all text it learned).

struct SavedModel: Codable { let config: Config; let merges: [[Int]]; let weights: [[Double]]; let memory: [String] }

func save(_ model: Model, _ tok: Tokenizer, _ memory: [String], to path: String) throws {
    let saved = SavedModel(config: model.config, merges: tok.merges, weights: model.params.map { $0.w }, memory: memory)
    try JSONEncoder().encode(saved).write(to: URL(fileURLWithPath: path))
}

func load(_ path: String) throws -> (Model, Tokenizer, [String]) {
    let saved = try JSONDecoder().decode(SavedModel.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    let model = Model(saved.config, seed: 0)
    guard model.params.count == saved.weights.count else { throw CocoaError(.fileReadCorruptFile) }
    for (p, w) in zip(model.params, saved.weights) {
        guard p.w.count == w.count else { throw CocoaError(.fileReadCorruptFile) }
        p.w = w
    }
    return (model, Tokenizer(merges: saved.merges), saved.memory)
}

// MARK: - Step 11: Learning from new inputs (online fine-tuning)
// A new sentence is added to memory, then the model takes a few gradient steps
// on it. Each batch also replays a few old windows so it does not forget
// everything else (forgetting old skills is a real problem in fine-tuning).

// P(completion | prompt): multiply the probabilities of each completion token.
func completionProbability(_ model: Model, _ tok: Tokenizer, _ prompt: String, _ completion: String) -> Double {
    let start = tok.encode("\n" + prompt).count
    let all = Array(tok.encode("\n" + prompt + completion).suffix(model.config.context + 1))
    let shift = tok.encode("\n" + prompt + completion).count - all.count   // tokens cut off the front
    let V = model.config.vocab
    let logits = model.forward(Array(all.dropLast())).0
    var logP = 0.0
    for t in max(start - 1 - shift, 0)..<(all.count - 1) {
        logP += log(softmax(Array(logits[(t * V)..<((t + 1) * V)]))[all[t + 1]])
    }
    return exp(logP)
}

func learn(_ model: Model, _ tok: Tokenizer, _ sentence: String, memory: inout [String],
           steps: Int = 10, seed: UInt64 = 1) -> Double {
    if memory.last != sentence { memory.append(sentence) }
    let ids = Array(tok.encode("\n" + sentence + "\n").prefix(model.config.context + 1))
    let replay = tok.encode(memory.joined(separator: "\n"))
    var rng = RNG(seed: seed)
    var opt = Adam(model.params, lr: 0.002)
    var loss = 0.0
    for _ in 0..<steps {
        var batch: [Example] = [(Array(ids.dropLast()), Array(ids.dropFirst()))]
        for _ in 0..<3 where replay.count > model.config.context + 1 { batch.append(randomWindow(replay, model.config.context, &rng)) }
        loss = trainStep(model, &opt, batch)
    }
    return loss
}

// MARK: - Step 12: Self checks (PASS / FAIL)
// Tests that catch real bugs: wrong gradients, peeking at the future,
// broken probabilities, randomness that is not repeatable, lossy tokenizing.

var failures = 0
func check(_ ok: Bool, _ name: String, _ detail: String) {
    print("  [\(ok ? "PASS" : "FAIL")] \(name): \(detail)")
    if !ok { failures += 1 }
}

// Compare backprop gradients with finite differences: (L(w+h) - L(w-h)) / 2h.
func gradientCheck(_ tok: Tokenizer) {
    let model = Model(Config(vocab: tok.size, dim: 8, layers: 2, context: 6), seed: 3)
    let ids = tok.encode("the dog of mia likes to swim. a token is text.")
    let x = Array(ids.prefix(6)), y = Array(ids[1...6])  // targets = inputs shifted by one
    func loss() -> Double { crossEntropy(model.forward(x).0, y, tok.size).loss }
    model.zeroGrads()
    let (logits, cache) = model.forward(x)
    model.backward(crossEntropy(logits, y, tok.size).dLogits, cache)
    var rng = RNG(seed: 5), worst = 0.0, tested = 0
    for p in model.params {
        let used = p === model.tokenEmb ? x.map { $0 * 8 + rng.below(8) } : (0..<3).map { _ in rng.below(p.w.count) }
        for i in used {
            let old = p.w[i], h = 1e-5
            p.w[i] = old + h; let up = loss()
            p.w[i] = old - h; let down = loss()
            p.w[i] = old
            let numeric = (up - down) / (2 * h)
            let err = abs(numeric - p.g[i]) / max(abs(numeric) + abs(p.g[i]), 1e-8)
            if abs(numeric) + abs(p.g[i]) > 1e-9 { worst = max(worst, err) }
            tested += 1
        }
    }
    check(worst < 1e-5, "gradient check", String(format: "%d weights, worst relative error %.1e", tested, worst))
}

func causalCheck(_ model: Model, _ tok: Tokenizer) {
    var a = tok.encode("the cat likes to sleep.")
    let before = model.forward(a).0
    a[a.count - 1] = (a[a.count - 1] + 1) % tok.size  // change only the LAST token
    let after = model.forward(a).0
    let V = tok.size, cut = (a.count - 1) * V
    let earlier = zip(before.prefix(cut), after.prefix(cut)).map { abs($0 - $1) }.max() ?? 0
    let last = zip(before.suffix(V), after.suffix(V)).map { abs($0 - $1) }.max() ?? 0
    check(earlier == 0 && last > 0, "causal mask", String(format: "earlier positions changed by %.1e, last by %.2f", earlier, last))
}

func quickChecks(_ tok: Tokenizer) {
    let samples = ["my dog is named biscuit.", "Attention, please!\n  two  spaces", "café ☕ 日本", ""]
    let ok = samples.allSatisfy { tok.decode(tok.encode($0)) == $0 }
    check(ok, "tokenizer round trip", "\(samples.count) texts incl. emoji and Japanese decode back exactly")
    let p = softmax([2.0, -1.0, 0.5, 1000.0, -1000.0])
    check(abs(p.reduce(0, +) - 1) < 1e-12 && p.allSatisfy { $0 >= 0 }, "softmax sums to 1", String(format: "sum = %.15f (even with scores of +-1000)", p.reduce(0, +)))
    let m1 = Model(Config(vocab: tok.size), seed: 42), m2 = Model(Config(vocab: tok.size), seed: 42)
    let same = zip(m1.params, m2.params).allSatisfy { $0.w == $1.w }
    var r1 = RNG(seed: 7), r2 = RNG(seed: 7)
    let g1 = generate(m1, tok, "the", temperature: 1, rng: &r1), g2 = generate(m2, tok, "the", temperature: 1, rng: &r2)
    check(same && g1 == g2, "determinism with seed", "same seed -> identical weights and identical sampled text")
    gradientCheck(tok)
    causalCheck(m1, tok)
}

// MARK: - Step 13: Built-in training text and commands
// About 120 short lines: simple facts from the CME 295 cheatsheet plus little
// pattern sentences about pets. Then the commands that tie everything together.

func builtInText() -> String {
    var lines = [
        "a token is a small piece of text.", "a tokenizer turns text into numbers.",
        "bpe merges the most common pair of tokens.", "an embedding turns a token into a vector.",
        "attention lets a token look at earlier tokens.", "the causal mask hides the future.",
        "a query asks and a key answers.", "the value carries the information.",
        "softmax turns scores into probabilities.", "probabilities always add up to one.",
        "the loss is small when the guess is right.", "cross entropy measures surprise.",
        "backprop finds the gradient of the loss.", "the gradient points uphill.",
        "adam takes a careful step downhill.", "the learning rate sets the step size.",
        "layer norm keeps the numbers calm.", "a residual adds the input back.",
        "the mlp thinks about each token alone.", "a transformer stacks many blocks.",
        "temperature controls how wild the text is.", "top k keeps only the best tokens.",
        "fine tuning teaches a model new things.", "lora trains only a few small matrices.",
        "rlhf uses human feedback to improve answers.", "a larger model needs more data.",
    ]
    let names = ["tom", "mia", "max", "luna", "sam", "zoe", "leo", "ivy"]
    let pets = ["cat", "dog", "bird", "fish"], colors = ["red", "blue", "green", "brown"]
    let verbs = ["run", "sleep", "eat", "play", "swim", "sing"]
    for (i, name) in names.enumerated() {
        for (j, pet) in pets.enumerated() {
            lines.append("my \(pet) is named \(name).")
            lines.append("\(name) has a \(colors[(i + j) % 4]) \(pet).")
            lines.append("the \(pet) of \(name) likes to \(verbs[(i * 3 + j) % 6]).")
        }
    }
    var rng = RNG(seed: 11)                           // shuffle the lines the same way every run
    for i in stride(from: lines.count - 1, to: 0, by: -1) { lines.swapAt(i, rng.below(i + 1)) }
    return lines.joined(separator: "\n")
}

func option(_ name: String, _ fallback: Double) -> Double {
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: name), i + 1 < args.count, let v = Double(args[i + 1]) { return v }
    return fallback
}

let modelPath = "myai_model.json"
let steps = Int(option("--steps", 1500)), seed = UInt64(option("--seed", 1)), temp = option("--temp", 0.7)
let words = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("--") && Double($0) == nil }
let command = words.first ?? "demo"

func trainFresh(_ text: String) -> (Model, Tokenizer, [String]) {
    let tok = Tokenizer.train(text, merges: 200)
    let data = tok.encode(text)
    guard data.count > 2 * Config(vocab: tok.size).context else {
        print("That text is too short to train on (need more than a few lines).")
        exit(1)
    }
    let model = Model(Config(vocab: tok.size), seed: seed)
    print("Tokenizer: \(tok.size) tokens. Text: \(text.utf8.count) bytes -> \(data.count) tokens.")
    print("Model: \(model.parameterCount) learnable numbers. Training \(steps) steps...")
    let start = Date(), before = evalLoss(model, data)
    _ = train(model, data, steps: steps, seed: seed)
    let after = evalLoss(model, data)
    print(String(format: "Trained in %.1f s. Loss %.3f -> %.3f", Date().timeIntervalSince(start), before, after))
    check(after < before * 0.5, "loss decreases", String(format: "%.3f -> %.3f", before, after))
    return (model, tok, text.components(separatedBy: "\n"))
}

func loadOrExit() -> (Model, Tokenizer, [String]) {
    guard let loaded = try? load(modelPath) else {
        print("No \(modelPath) here yet. Run ./myai (demo) or ./myai train <file.txt> first.")
        exit(1)
    }
    return loaded
}

// Teach one sentence: report P(last word | the rest) before and after.
func teachAndReport(_ model: Model, _ tok: Tokenizer, _ sentence: String, _ memory: inout [String], rounds: Int) -> (Double, Double) {
    var parts = sentence.split(separator: " ").map(String.init)
    let lastWord = " " + (parts.popLast() ?? ""), prompt = parts.joined(separator: " ")
    let before = completionProbability(model, tok, prompt, lastWord)
    print(String(format: "  before: P(\"%@\" | \"%@\") = %.2e", lastWord, prompt, before))
    for r in 1...rounds {
        let loss = learn(model, tok, sentence, memory: &memory, steps: 5, seed: UInt64(r))
        print(String(format: "  round %d (5 steps): loss %.3f  P = %.2e", r, loss, completionProbability(model, tok, prompt, lastWord)))
    }
    let after = completionProbability(model, tok, prompt, lastWord)
    var rng = RNG(seed: seed)
    print("  generate(\"\(prompt)\", greedy) -> \(generate(model, tok, prompt, temperature: 0, rng: &rng))")
    return (before, after)
}

switch command {
case "demo", "test":
    let clock = Date()
    print("== Self checks ==")
    let text = builtInText()
    quickChecks(Tokenizer.train(text, merges: 200))
    print("== Training on the built-in text ==")
    var (model, tok, memory) = trainFresh(text)
    var rng = RNG(seed: seed)
    print("== Samples (temperature \(temp), top-k 8) ==")
    for prompt in ["my cat is named", "attention", "the dog of", "softmax"] {
        print("  " + generate(model, tok, prompt, temperature: temp, rng: &rng))
    }
    print("== Learning from input: teaching \"my dog is named biscuit.\" ==")
    let baseBefore = evalLoss(model, tok.encode(text))
    let (pBefore, pAfter) = teachAndReport(model, tok, "my dog is named biscuit.", &memory, rounds: 4)
    var greedy = RNG(seed: 1)
    let reply = generate(model, tok, "my dog is named", temperature: 0, rng: &greedy)
    check(pAfter > pBefore * 10 && pAfter > 0.5, "learns from input", String(format: "P(biscuit) %.2e -> %.3f", pBefore, pAfter))
    check(reply == "my dog is named biscuit.", "reproduces taught fact", reply)
    print(String(format: "  old text loss before/after teaching: %.3f -> %.3f (small rise = a little forgetting)", baseBefore, evalLoss(model, tok.encode(text))))
    print("  my cat is named -> \(generate(model, tok, "my cat is named", temperature: 0, rng: &greedy))")
    try! save(model, tok, memory, to: modelPath)
    let (copy, _, _) = try! load(modelPath)
    let probe = tok.encode("my dog is named")
    check(copy.forward(probe).0 == model.forward(probe).0, "save/load round trip", "\(modelPath) reloads to identical outputs")
    print(String(format: "== %@: %d check(s) failed. Total time %.1f s ==", failures == 0 ? "ALL PASS" : "PROBLEMS", failures, Date().timeIntervalSince(clock)))
    if failures > 0 { exit(1) }
case "train":
    guard words.count > 1, let text = try? String(contentsOfFile: words[1], encoding: .utf8) else {
        print("usage: ./myai train <file.txt>")
        exit(1)
    }
    let (model, tok, memory) = trainFresh(text.lowercased())
    try! save(model, tok, memory, to: modelPath)
    print("Saved \(modelPath)")
case "generate":
    let (model, tok, _) = loadOrExit()
    var rng = RNG(seed: seed)
    for _ in 0..<3 { print(generate(model, tok, words.dropFirst().joined(separator: " ").lowercased(), temperature: temp, rng: &rng)) }
case "teach":
    var (model, tok, memory) = loadOrExit()
    let sentence = words.dropFirst().joined(separator: " ").lowercased()
    guard sentence.contains(" ") else {
        print("usage: ./myai teach \"a sentence with at least two words\"")
        exit(1)
    }
    _ = teachAndReport(model, tok, sentence, &memory, rounds: 4)
    try! save(model, tok, memory, to: modelPath)
case "chat":
    var (model, tok, memory) = loadOrExit()
    var rng = RNG(seed: seed)
    print("Type a sentence to teach me, '? words' to just ask me to continue, 'quit' to stop.")
    while let line = readLine()?.trimmingCharacters(in: .whitespaces).lowercased(), line != "quit" {
        if line.isEmpty { continue }
        if line.hasPrefix("?") {
            print("ai> " + generate(model, tok, String(line.dropFirst()).trimmingCharacters(in: .whitespaces), temperature: temp, rng: &rng))
            continue
        }
        let loss = learn(model, tok, line, memory: &memory, steps: 15)
        let firstHalf = line.split(separator: " ").prefix(max(1, line.split(separator: " ").count / 2)).joined(separator: " ")
        print(String(format: "ai> learned it (loss %.3f). Recall: ", loss) + generate(model, tok, firstHalf, temperature: 0, rng: &rng))
    }
    try! save(model, tok, memory, to: modelPath)
    print("Saved what I learned to \(modelPath)")
default:
    print("commands: demo | train <file.txt> | generate \"prompt\" | teach \"sentence\" | chat")
}
