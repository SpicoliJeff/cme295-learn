// =====================================================================
//  UberSim.swift - a pretend Uber, built 100% in code
//  A teaching project for Spicoli Jeff (made with Grok Bot)
//
//  NOTHING here touches a real network. Every wire, router, server
//  and packet is just a Swift value living in the computer's memory.
//
//  We follow ONE ride from start to finish. Every STEP says:
//    OSI LAYER  - which of the 7 layers it lives in
//    REAL WORLD - what you would plug in there in a real app
//
//  The 7 OSI layers (top to bottom):
//    7 Application   - HTTP, APIs, DNS, DHCP (what apps say)
//    6 Presentation  - encryption, compression, data format
//    5 Session       - keeping a conversation open (TLS session)
//    4 Transport     - TCP / UDP, ports (door numbers)
//    3 Network       - IP addresses, routers, NAT, subnets
//    2 Data Link     - MAC addresses, ARP, Wi-Fi frames
//    1 Physical      - radio waves, cables, the actual bits
//
//  Build + run:  swiftc -O UberSim.swift -o ubersim && ./ubersim
// =====================================================================

import Foundation  // Apple's basic toolbox (text, math, formatting)

// MARK: STEP 0 - Helpers: dice, self-checks, titles
// OSI LAYER: none (these are just tools for the simulation)
// REAL WORLD: nothing to connect; a real app would use XCTest for checks.

struct Dice {  // a predictable "random" number maker, so every run is the same
    var state: UInt64  // the secret number the dice remembers
    mutating func next() -> UInt64 {  // roll: makes the next number
        state = state &* 6364136223846793005 &+ 1442695040888963407  // scramble it (&* lets it wrap around)
        return state >> 33  // keep the top bits, they are the most random
    }
    mutating func chance(_ percent: UInt64) -> Bool {  // true about `percent` times in 100
        return next() % 100 < percent  // a roll from 0-99 under the percent is a hit
    }
}
var dice = Dice(state: 295)  // seed 295 (CME 295!) so the story is identical every run

var passCount = 0  // how many self-checks passed
var failCount = 0  // how many self-checks failed
func check(_ ok: Bool, _ name: String) {  // one self-check: prints PASS or FAIL
    if ok { passCount += 1; print("  PASS  \(name)") }  // true: count it, say PASS
    else { failCount += 1; print("  FAIL  \(name)") }  // false: count it, say FAIL
}
func banner(_ title: String, _ layer: String) {  // prints a big title for each step
    print("\n==== \(title)  [\(layer)] ====")  // \n = blank line first; \(x) drops x into the text
}

// MARK: STEP 1 - Network, interface, IP vs MAC
// Notes: "network", "ip address communicate", "interface",
//        "ip local v mac network interface"
// OSI LAYER: 1 Physical (Wi-Fi radio) + 2 Data Link (MAC) + 3 Network (IP)
// A network   = devices that can send each other messages.
// An interface = one "door" on a device (Wi-Fi chip, Ethernet port).
// MAC address = the door's factory serial number, used only on the LOCAL network.
// IP address  = the device's mailing address, used ACROSS networks.
// REAL WORLD: connect the iPhone's Wi-Fi chip / a server's Ethernet port here.

struct IP: Hashable, CustomStringConvertible {  // an IPv4 address such as 192.168.1.20
    let value: UInt32  // stored as one 32-bit number (4 bytes)
    init(_ text: String) {  // build one from text like "10.0.0.5"
        let p = text.split(separator: ".").map { UInt32($0)! }  // split at the dots -> 4 numbers
        value = p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]  // pack the 4 bytes into one number
    }
    init(raw: UInt32) { value = raw }  // build one straight from the number
    var description: String {  // how it prints: back to dotted text
        return "\(value >> 24 & 255).\(value >> 16 & 255).\(value >> 8 & 255).\(value & 255)"  // unpack each byte
    }
}
struct Interface {  // one network "door" on a device
    let name: String  // e.g. "wlan0" (Wi-Fi) or "eth0" (cable)
    let mac: String  // hardware address, burned in at the factory
    var ip: IP?  // IP address; nil (empty) until DHCP gives us one
}
struct Device {  // anything on the network: phone, router, server
    let name: String  // a friendly name for printing
    var nic: Interface  // its network interface card
}
var riderPhone = Device(name: "Rider iPhone",  // the rider's phone...
    nic: Interface(name: "wlan0", mac: "AA:AA:AA:00:00:01", ip: nil))  // ...with Wi-Fi but no IP yet
let homeRouter = Device(name: "Home Wi-Fi router",  // the router at the edge of the home network...
    nic: Interface(name: "lan0", mac: "BB:BB:BB:00:00:01", ip: IP("192.168.1.1")))  // ...always at .1

// MARK: STEP 2 - Subnetting
// Note: "subnetting"
// OSI LAYER: 3 Network
// A subnet mask splits an IP into "street" (network part) + "house" (host part).
// /24 means the first 24 bits are the street, so all 192.168.1.x are neighbors.
// Neighbors talk directly; everyone else goes through the gateway (the router).
// REAL WORLD: connect your router's LAN settings / a cloud VPC subnet (10.0.1.0/24).

struct Subnet {  // a block of addresses like 192.168.1.0/24
    let base: IP  // the first address of the block
    let prefix: Int  // how many bits are the "street" part
    var mask: UInt32 { return prefix == 0 ? 0 : UInt32.max << UInt32(32 - prefix) }  // /24 -> 255.255.255.0
    func contains(_ ip: IP) -> Bool {  // is this address inside the block?
        return ip.value & mask == base.value & mask  // same street = same subnet
    }
    var size: Int { return 1 << (32 - prefix) }  // how many addresses fit (/24 -> 256)
}

// MARK: STEP 3 - DHCP hands out IP, gateway and DNS
// Note: "dhcp ip gateway dns"
// OSI LAYER: 7 Application protocol (over UDP ports 67/68) that sets up Layer 3
// The phone shouts "anyone have an address for me?" Four messages = DORA:
// Discover -> Offer -> Request -> Acknowledge.
// REAL WORLD: connect your Wi-Fi router's DHCP server (or a cloud's DHCP options).

struct Lease {  // the settings DHCP hands back
    let ip: IP  // your address on this network
    let gateway: IP  // the router to send far-away traffic to
    let dns: IP  // who to ask for name -> IP lookups
}
struct DHCPServer {  // lives inside the home router
    let subnet: Subnet  // the block it hands addresses out from
    var nextHost: UInt32 = 20  // first free house number (.20)
    var leases: [String: IP] = [:]  // MAC -> IP already handed out
    mutating func handle(mac: String) -> Lease {  // runs the 4 DORA messages
        print("  DHCP DISCOVER from \(mac) to 255.255.255.255 (broadcast: 'need an IP!')")  // 1 phone shouts
        let ip = leases[mac] ?? IP(raw: subnet.base.value + nextHost)  // reuse old lease, or next free address
        if leases[mac] == nil { nextHost += 1 }  // only use up an address if it was a new one
        leases[mac] = ip  // remember who has it
        print("  DHCP OFFER    \(ip) for \(mac)")  // 2 server offers an address
        print("  DHCP REQUEST  \(mac) asks to keep \(ip)")  // 3 phone accepts
        print("  DHCP ACK      lease confirmed for 24h")  // 4 server confirms
        return Lease(ip: ip, gateway: IP("192.168.1.1"), dns: IP("192.168.1.1"))  // the bundle of settings
    }
}

// MARK: STEP 4 - ARP turns an IP into a MAC
// Note: "arp address resolution protocol"
// OSI LAYER: 2 Data Link (helps Layer 3 IP find a Layer 2 MAC)
// Wi-Fi frames are addressed by MAC, not IP. To reach the gateway the phone
// broadcasts "Who has 192.168.1.1?" and the owner replies with its MAC.
// The answer is cached so we don't have to ask again.
// REAL WORLD: the operating system's ARP cache (type `arp -a` on a Mac to see it).

struct ARP {  // the address-resolution helper on the phone
    var cache: [IP: String] = [:]  // IP -> MAC answers we already know
    var broadcasts = 0  // how many times we had to shout
    mutating func resolve(_ ip: IP, onLAN lan: [Device]) -> String? {  // find the MAC for an IP
        if let mac = cache[ip] { print("  ARP cache hit: \(ip) is at \(mac)"); return mac }  // known: no shouting
        broadcasts += 1  // we must ask everyone
        print("  ARP WHO-HAS \(ip)? (broadcast to FF:FF:FF:FF:FF:FF)")  // the question every device hears
        for d in lan where d.nic.ip == ip {  // only the device with that IP answers
            print("  ARP REPLY   \(ip) is-at \(d.nic.mac) (\(d.name))")  // the answer
            cache[ip] = d.nic.mac  // save it for next time
            return d.nic.mac  // hand back the MAC
        }
        print("  ARP (no reply - nobody has \(ip))")  // silence
        return nil  // nobody answered
    }
}

// MARK: STEP 5 - NAT, the NAT table, and a NAT gateway
// Notes: "network address translation", "nat table", "nat gateway",
//        "private servers updates external", "internet public app data"
// OSI LAYER: 3 Network (rewrites IPs) + 4 Transport (rewrites ports)
// Private IPs (192.168.x.x, 10.x.x.x) are not allowed on the public internet.
// The router swaps the private address for its ONE public address and writes
// the swap in a NAT table, so replies can find their way back home.
// A NAT gateway does the same for private cloud servers: they can go OUT
// (e.g. download updates) but nobody outside can start a connection IN.
// REAL WORLD: connect your home router / an AWS or Google Cloud NAT Gateway here.

struct Packet {  // one chunk of data travelling across the network
    var srcIP: IP, srcPort: Int  // who sent it (address + door number)
    var dstIP: IP, dstPort: Int  // where it is going
    var payload: String  // the actual data inside
}
struct NATRouter {  // a router that hides private addresses
    let publicIP: IP  // the single public address the world sees
    var table: [Int: (ip: IP, port: Int)] = [:]  // NAT table: public port -> private IP + port
    var nextPort = 40000  // next free public port to hand out
    mutating func outbound(_ p: Packet) -> Packet {  // inside -> internet
        var out = p  // make an editable copy
        out.srcIP = publicIP  // replace the private IP with the public one
        out.srcPort = nextPort  // replace the port with a fresh public port
        table[nextPort] = (p.srcIP, p.srcPort)  // write the swap in the NAT table
        nextPort += 1  // the next connection gets the next port
        return out  // this version goes to the internet
    }
    func inbound(_ p: Packet) -> Packet? {  // internet -> inside
        guard let entry = table[p.dstPort] else { return nil }  // no table entry = nobody asked = DROP
        var back = p  // make an editable copy
        back.dstIP = entry.ip  // put the private IP back
        back.dstPort = entry.port  // put the private port back
        return back  // deliver it to the right device inside
    }
}

// MARK: STEP 6 - DNS turns a name into an IP
// Note: "dns"
// OSI LAYER: 7 Application (usually over UDP port 53)
// Humans type api.uber.sim; computers need an IP. DNS is the phone book.
// Answers carry a TTL (time to live) so they can be cached for speed.
// REAL WORLD: connect Route 53 / Cloudflare DNS / your internet provider's resolver.

struct DNSResolver {  // the phone-book service
    var records: [String: IP]  // the official phone book (authoritative records)
    var cache: [String: (ip: IP, ttl: Int)] = [:]  // name -> answer + seconds left
    var lookups = 0  // how many slow, full lookups we did
    mutating func resolve(_ name: String) -> IP? {  // look up one name
        if let hit = cache[name], hit.ttl > 0 {  // in the cache and not expired?
            print("  DNS cache hit  \(name) -> \(hit.ip) (TTL \(hit.ttl)s)")  // fast answer
            return hit.ip  // done, no trip across the network
        }
        lookups += 1  // a full lookup costs time
        print("  DNS query  \(name)?  root '.' -> 'sim.' -> 'uber.sim.'")  // walk down the name tree
        guard let ip = records[name] else { print("  DNS NXDOMAIN (no such name)"); return nil }  // unknown name
        cache[name] = (ip, 300)  // remember it for 300 seconds
        print("  DNS answer \(name) -> \(ip)")  // the IP we wanted
        return ip  // hand back the address
    }
}

// MARK: STEP 7 - TCP: handshake, loss, flow control, congestion control
// Notes: "tcp reliable communication detect loss flow control congestion control",
//        "three way handshake connection", "syn syn-ack ack"
// OSI LAYER: 4 Transport
// TCP promises: everything arrives, in order, without drowning anyone.
//  - Handshake: SYN -> SYN-ACK -> ACK agrees on starting sequence numbers.
//  - Detect loss: every segment is numbered; no ACK in time = send it again.
//  - Flow control: the RECEIVER says how much it can hold (its window).
//  - Congestion control: the SENDER slows down when the network is jammed.
//    Slow start doubles each round; AIMD adds 1 when fine, halves on loss.
// REAL WORLD: the phone's operating system does this inside URLSession / sockets.

struct TCPConnection {  // simulates client + server ends of one connection
    var log: [String] = []  // every handshake message, in order
    var established = false  // true once the handshake finishes
    mutating func handshake() {  // the famous three-way handshake
        let c = UInt32(dice.next() % 10000)  // client picks a random starting number
        log.append("SYN")  // 1 client: "let's talk, I start at c"
        print("  -> SYN      seq=\(c)")  // show message 1
        let s = UInt32(dice.next() % 10000)  // server picks its own starting number
        log.append("SYN-ACK")  // 2 server: "ok, I start at s, and I got yours"
        print("  <- SYN-ACK  seq=\(s) ack=\(c + 1)")  // show message 2
        log.append("ACK")  // 3 client: "got yours too"
        print("  -> ACK      ack=\(s + 1)")  // show message 3
        established = true  // both sides agree: the connection is open
    }
    // Sends `count` segments over a lossy link.
    // Returns: segments delivered (in order), number of resends, window size each round.
    mutating func send(count: Int, lossPercent: UInt64, receiverWindow: Int) -> ([Int], Int, [Int]) {
        var delivered: [Int] = []  // segment numbers the receiver got, in order
        var next = 0  // the next segment number to send
        var cwnd = 1  // congestion window: how many segments we dare send at once
        var ssthresh = 8  // slow-start threshold: where doubling turns into +1
        var resends = 0  // how many segments had to be sent again
        var history: [Int] = []  // window size each round, to print later
        while next < count {  // keep going until everything arrives
            let window = min(cwnd, receiverWindow)  // flow control: never more than receiver allows
            history.append(window)  // record this round
            var lostOne = false  // did anything get lost this round?
            for _ in 0..<window where next < count {  // send a burst of `window` segments
                if dice.chance(lossPercent) {  // the network dropped this segment
                    print("     segment \(next) LOST -> no ACK -> timeout -> resend")  // loss detected
                    resends += 1  // count the retransmission
                    lostOne = true  // remember there was loss
                    break  // stop the burst; resend from `next` next round
                }
                delivered.append(next)  // receiver got it and ACKs it
                next += 1  // move on to the next segment
            }
            if lostOne { ssthresh = max(cwnd / 2, 1); cwnd = ssthresh }  // loss: cut in half (Multiplicative Decrease)
            else if cwnd < ssthresh { cwnd *= 2 }  // slow start: double each round
            else { cwnd += 1 }  // congestion avoidance: add one (Additive Increase)
        }
        return (delivered, resends, history)  // results for the self-checks
    }
}

// MARK: STEP 8 - HTTPS / TLS: secrecy + integrity + proof of identity
// Notes: "sensitive data", "https TLS packet interception blockers integrity",
//        "tls at the edge"
// OSI LAYER: 6 Presentation + 5 Session (TLS sits between TCP and HTTP)
// TLS does 3 jobs:
//  1. Encryption: someone intercepting packets sees only gibberish.
//  2. Integrity: every record carries a tag; any change is caught and rejected.
//  3. Authentication: the server's certificate proves it really is api.uber.sim.
// "At the edge" = TLS ends at the first Uber server we reach (the reverse proxy),
// so the servers inside don't each need certificates.
// !!! TOY CRYPTO - for learning only. NEVER use this for real secrets. !!!
// REAL WORLD: connect Apple's CryptoKit / Network.framework and a real
//             certificate from Let's Encrypt or your cloud's certificate manager.

func toyHash(_ text: String) -> UInt64 {  // FNV-1a: turns any text into a 64-bit "fingerprint"
    var h: UInt64 = 14695981039346656037  // the standard FNV starting number
    for b in text.utf8 { h = (h ^ UInt64(b)) &* 1099511628211 }  // mix in each byte
    return h  // same text -> same fingerprint; tiny change -> totally different
}
func toyHMAC(key: UInt64, _ msg: String) -> UInt64 {  // integrity tag: only the key holder can make it
    return toyHash("\(key)|" + String(toyHash("\(key)|\(msg)")))  // hash twice with the key, like real HMAC
}
func modPow(_ base: UInt64, _ exp: UInt64, _ mod: UInt64) -> UInt64 {  // (base^exp) % mod, done fast
    var result: UInt64 = 1, b = base % mod, e = exp  // starting values
    while e > 0 {  // square-and-multiply loop, one bit at a time
        if e & 1 == 1 { result = result * b % mod }  // this bit is 1: multiply it in
        b = b * b % mod  // square the base
        e >>= 1  // move to the next bit
    }
    return result  // the answer
}
func toyCipher(_ bytes: [UInt8], key: UInt64) -> [UInt8] {  // XOR with a key stream (same call locks + unlocks)
    var stream = Dice(state: key)  // key-stream maker seeded by the secret key
    return bytes.map { $0 ^ UInt8(stream.next() & 255) }  // flip bits using the stream
}
struct TLSRecord {  // one encrypted chunk on the wire
    var cipher: [UInt8]  // the scrambled bytes
    var tag: UInt64  // the integrity tag
}
struct TLSSession {  // one secure tunnel between the phone and the edge
    var key: UInt64 = 0  // the shared secret both sides worked out
    mutating func handshake(serverName: String, certName: String) -> Bool {  // false = fake server
        print("  -> ClientHello  (I speak TLS 1.3, I want \(serverName))")  // phone says hello
        print("  <- ServerHello + Certificate for '\(certName)'")  // server answers with its ID card
        guard certName == serverName else { print("  !! certificate name mismatch -> ABORT"); return false }  // identity check
        let p: UInt64 = 2147483647, g: UInt64 = 16807  // public toy Diffie-Hellman numbers
        let a = 1 + dice.next() % 100000, b = 1 + dice.next() % 100000  // each side's private secret (never sent)
        let A = modPow(g, a, p), B = modPow(g, b, p)  // the public halves sent over the wire
        let clientKey = modPow(B, a, p), serverKey = modPow(A, b, p)  // each side mixes: same answer!
        print("  key exchange: phone sends \(A), server sends \(B) (spies see these, useless alone)")  // show it
        print("  both sides computed the same secret key: \(clientKey == serverKey)")  // the magic
        key = clientKey  // save the shared secret
        return clientKey == serverKey  // handshake worked if they match
    }
    func seal(_ text: String) -> TLSRecord {  // encrypt + tag one message
        let c = toyCipher(Array(text.utf8), key: key)  // scramble the bytes
        return TLSRecord(cipher: c, tag: toyHMAC(key: key, c.map { String($0) }.joined()))  // tag the scrambled bytes
    }
    func open(_ r: TLSRecord) -> String? {  // check the tag, then decrypt; nil = rejected
        let expected = toyHMAC(key: key, r.cipher.map { String($0) }.joined())  // recompute the tag
        guard expected == r.tag else { return nil }  // tag mismatch = someone changed it = REJECT
        return String(decoding: toyCipher(r.cipher, key: key), as: UTF8.self)  // unscramble back to text
    }
}

// MARK: STEP 9 - Authentication with a token
// Note: "authentication"
// OSI LAYER: 7 Application
// Log in once -> get a signed token -> send it with every request
// ("Authorization: Bearer <token>"). The server checks the signature,
// so a made-up or edited token is refused with 401 Unauthorized.
// REAL WORLD: connect Sign in with Apple / OAuth 2.0 / a JWT library here.

let authSecret: UInt64 = 0x5EC12E7  // server-only secret used to sign tokens
func issueToken(user: String) -> String {  // called after a correct password
    let sig = toyHMAC(key: authSecret, user)  // sign the user name with the secret
    return "\(user).\(String(sig, radix: 16))"  // token = name + "." + signature in hex
}
func verifyToken(_ token: String?) -> String? {  // returns the user if valid, nil if not
    guard let t = token else { return nil }  // no token at all
    let parts = t.split(separator: ".").map(String.init)  // split "user.signature"
    guard parts.count == 2 else { return nil }  // wrong shape
    let good = String(toyHMAC(key: authSecret, parts[0]), radix: 16)  // the signature it SHOULD have
    return parts[1] == good ? parts[0] : nil  // match = trust the name; else refuse
}

// MARK: STEP 10 - Firewalls, security groups, the access matrix
// Notes: "control traffic", "access matrix firewalls",
//        "cloud security group network access rules container environments
//         debug timeouts rule dropping traffic", "network boundary accidentals"
// OSI LAYER: 3 Network + 4 Transport (rules match addresses, ports, protocol)
// A firewall checks each packet against its rules. No matching ALLOW = DROP.
// DROP sends nothing back, so the client waits... and waits... = TIMEOUT.
// (REJECT answers "refused" right away, which is much easier to debug.)
// Access matrix = a grid of "which zone may talk to which zone, on which port".
// REAL WORLD: connect AWS Security Groups / Google Cloud firewall rules /
//             Kubernetes NetworkPolicies for your containers here.

struct Rule {  // one line in the firewall's allow list
    let from: String  // zone that may send ("internet", "edge", "app", "db")
    let to: String  // zone that may receive
    let port: Int  // which door number
}
struct Firewall {  // the traffic cop between zones
    var rules: [Rule]  // the allow list (anything else is denied)
    var dropped = 0  // how many packets were thrown away
    func permits(_ from: String, _ to: String, _ port: Int) -> Bool {  // look only, no counting
        return rules.contains { $0.from == from && $0.to == to && $0.port == port }  // any rule match?
    }
    mutating func allows(_ from: String, _ to: String, _ port: Int) -> Bool {  // check a real packet
        let ok = permits(from, to, port)  // look it up
        if !ok { dropped += 1 }  // no match: silently drop it
        return ok  // tell the caller
    }
}
func connectWithTimeout(_ fw: inout Firewall, _ from: String, _ to: String, _ port: Int) -> String {
    for attempt in 1...3 {  // clients retry a few times before giving up
        if fw.allows(from, to, port) { return "connected on try \(attempt)" }  // let through
        print("     try \(attempt): SYN sent to \(to):\(port)... no reply (silently dropped)")  // nothing back
    }
    return "TIMEOUT after 3 tries (debug tip: check the security group rule!)"  // the classic mystery
}

// MARK: STEP 11 - HTTP requests and responses
// Notes: "http application request", "api request response", "get get post get"
// OSI LAYER: 7 Application
// HTTP = a text conversation: METHOD + PATH + HEADERS + BODY.
// GET reads data. POST creates something. The server answers with a status
// code (200 OK, 201 Created, 401 Unauthorized, 404 Not Found) and a body.
// Our ride uses: GET nearby drivers, POST request ride, GET ride status.
// REAL WORLD: connect URLSession in the iPhone app + a web framework on the server.

struct HTTPRequest {  // what the phone asks
    let method: String  // "GET" or "POST"
    let path: String  // e.g. "/v1/drivers/nearby"
    let user: String  // who is asking (from the checked token)
}
struct HTTPResponse {  // what the server answers
    let status: Int  // 200, 201, 401, 404...
    let body: String  // the data (simple text)
    var headers: [String: String] = [:]  // extra info such as "X-Cache: HIT"
}

// MARK: STEP 12 - The Uber "svc" services: match, price, trip, pay
// Note: "svc"
// OSI LAYER: 7 Application (pure business logic)
// "svc" = service: a small program that does one job well.
// REAL WORLD: each would be its own container (driver-svc, ride-svc, pay-svc)
//             backed by a database such as Postgres, plus Redis for live locations.

struct Driver {  // a driver on the map
    let id: String  // driver id
    let name: String  // their first name
    let x: Double, y: Double  // position on a simple map, in km
    var available: Bool  // free to take a ride?
}
var drivers = [  // the drivers online right now (the rider stands at 0,0)
    Driver(id: "d1", name: "Maya", x: 2.0, y: 1.0, available: true),  // about 2.2 km away, free
    Driver(id: "d2", name: "Leo", x: 0.5, y: 0.4, available: true),  // about 0.6 km away, free
    Driver(id: "d3", name: "Ana", x: 0.3, y: 0.2, available: false),  // closest, but already busy
    Driver(id: "d4", name: "Sam", x: 4.0, y: 3.0, available: true),  // 5 km away, free
]  // end of the list
func distance(_ x1: Double, _ y1: Double, _ x2: Double, _ y2: Double) -> Double {  // straight-line km
    return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)).squareRoot()  // Pythagoras: a^2 + b^2 = c^2
}
func nearestDriver(x: Double, y: Double) -> Driver? {  // the MATCHING service
    return drivers.filter { $0.available }  // keep only free drivers
        .min { distance($0.x, $0.y, x, y) < distance($1.x, $1.y, x, y) }  // the closest one wins
}
func price(km: Double, minutes: Double, surge: Double) -> Double {  // the PRICING service
    let raw = 2.50 + 1.20 * km + 0.30 * minutes  // base fare + per km + per minute
    return (max(raw, 7.00) * surge * 100).rounded() / 100  // minimum fare, times surge, round to cents
}
enum TripState: String {  // every stage a ride can be in (the TRIP LIFECYCLE)
    case requested, accepted, arriving, inTrip, completed, paid  // in this order
}
struct Trip {  // one ride
    let id: String  // ride id like "r1"
    let rider: String  // who asked for it
    var driver: Driver?  // who is driving (nil until matched)
    var state: TripState = .requested  // every ride starts as "requested"
    var fare: Double = 0  // price in dollars
    var history: [TripState] = [.requested]  // every stage it has been in
    mutating func advance(to next: TripState) {  // move to the next stage
        print("  trip \(id): \(state.rawValue) -> \(next.rawValue)")  // show the change
        state = next  // remember the new stage
        history.append(next)  // add it to the history
    }
}
struct Payments {  // the PAYMENT service
    var holds: [String: Double] = [:]  // money reserved on the card (not taken yet)
    var charged: [String: Double] = [:]  // money actually taken
    mutating func authorize(trip: String, amount: Double) { holds[trip] = amount }  // reserve the money
    mutating func capture(trip: String, amount: Double) -> Bool {  // take the final amount
        guard let hold = holds[trip], amount <= hold * 1.2 else { return false }  // never take way more than quoted
        holds[trip] = nil  // release the hold
        charged[trip] = amount  // record the real charge
        return true  // success
    }
}
var trips: [String: Trip] = [:]  // the "database" of rides
var payments = Payments()  // the payment service's memory

// MARK: STEP 13 - The API routes (what each request actually does)
// Notes: "api request response", "get get post get"
// OSI LAYER: 7 Application
// REAL WORLD: connect Vapor (Swift) / Express / Django routes + a database here.

func router(_ req: HTTPRequest) -> HTTPResponse {  // matches METHOD + PATH to code
    switch (req.method, req.path) {  // look at both together
    case ("GET", "/v1/drivers/nearby"):  // GET #1: who is around me?
        let list = drivers.filter { $0.available }  // free drivers only
            .sorted { distance($0.x, $0.y, 0, 0) < distance($1.x, $1.y, 0, 0) }  // closest first
            .map { "driver \($0.name) km \(String(format: "%.1f", distance($0.x, $0.y, 0, 0)))" }  // one line each
        return HTTPResponse(status: 200, body: list.joined(separator: " | "))  // 200 OK + the list
    case ("POST", "/v1/rides"):  // POST: please send me a car
        guard let d = nearestDriver(x: 0, y: 0) else { return HTTPResponse(status: 503, body: "no drivers") }  // none free
        let id = "r\(trips.count + 1)"  // a new ride id
        var trip = Trip(id: id, rider: req.user, driver: d)  // create the ride
        trip.fare = price(km: 5.2, minutes: 14, surge: 1.3)  // up-front price for a 5.2 km, 14 min trip
        payments.authorize(trip: id, amount: trip.fare)  // hold the money on the card
        trip.advance(to: .accepted)  // the driver accepts
        trips[id] = trip  // save it in the "database"
        if let i = drivers.firstIndex(where: { $0.id == d.id }) { drivers[i].available = false }  // driver now busy
        return HTTPResponse(status: 201, body: "ride \(id) driver \(d.name) fare $\(trip.fare)")  // 201 Created
    case ("GET", let path) where path.hasPrefix("/v1/rides/"):  // GET #2 and #3: how is my ride?
        let id = String(path.dropFirst("/v1/rides/".count))  // pull the id out of the path
        guard let t = trips[id] else { return HTTPResponse(status: 404, body: "no such ride") }  // unknown id
        return HTTPResponse(status: 200, body: "ride \(id) state \(t.state.rawValue)")  // the current status
    default:  // anything else
        return HTTPResponse(status: 404, body: "not found")  // 404 Not Found
    }
}

// MARK: STEP 14 - App servers and the load balancer
// Notes: "cpu load balancer multiple app servers", "load balancers"
// OSI LAYER: 4 Transport (an L4 balancer) or 7 Application (an L7 balancer). Ours is L7.
// One server's CPU can only do so much. Run several copies and let a load
// balancer spread the requests: round-robin (take turns) or least-loaded
// (pick whoever is least busy).
// REAL WORLD: connect AWS Application Load Balancer / NGINX upstreams /
//             a Kubernetes Service here.

final class AppServer {  // one copy of our backend ("class" so everyone shares the same one)
    let name: String  // e.g. "app-1"
    let privateIP: IP  // inside-only address, hidden from the internet
    var handled = 0  // requests it has served so far
    var active = 0  // requests it is busy with right now (its load)
    init(_ name: String, _ ip: String) { self.name = name; privateIP = IP(ip) }  // set up one server
    func handle(_ req: HTTPRequest) -> HTTPResponse {  // do the work
        handled += 1  // count it
        return router(req)  // run the API code from Step 13
    }
}
struct LoadBalancer {  // spreads requests over the servers
    let servers: [AppServer]  // the pool of servers
    var turn = 0  // whose turn it is (for round-robin)
    mutating func roundRobin() -> AppServer {  // take turns: 1, 2, 3, 1, 2, 3...
        let s = servers[turn % servers.count]  // the server whose turn it is
        turn += 1  // next time, the next one
        return s  // send the request there
    }
    func leastLoaded() -> AppServer {  // pick whoever is least busy
        return servers.min { $0.active < $1.active }!  // smallest "active" count wins
    }
}

// MARK: STEP 15 - The reverse proxy at the edge
// Notes: "tls route cache compress hide reverse proxy", "compress routing cache tls"
// OSI LAYER: 7 Application (and it ends TLS, so Layers 5/6 too)
// The ONE public front door. It does 5 jobs:
//  - TLS: decrypts here ("TLS termination at the edge")
//  - route: picks the service by path (/v1/drivers -> driver-svc)
//  - cache: remembers recent GET answers so app servers can rest
//  - compress: shrinks answers before they travel to the phone
//  - hide: the phone only ever sees the proxy's public IP, never app-1's
// REAL WORLD: connect NGINX / Envoy / Cloudflare / AWS CloudFront here.

func compress(_ text: String) -> (text: String, saved: Int) {  // toy compression: repeated words -> short codes
    var codes: [String: String] = [:]  // word -> short code like "~0"
    var out: [String] = []  // the compressed words
    for w in text.split(separator: " ").map(String.init) {  // look at each word
        if let c = codes[w], c.count < w.count { out.append(c) }  // seen before and code is shorter: use code
        else { if codes[w] == nil { codes[w] = "~\(codes.count)" }; out.append(w) }  // first time: keep word
    }
    let result = out.joined(separator: " ")  // glue the words back together
    return (result, text.count - result.count)  // compressed text + characters saved
}
struct EdgeProxy {  // the reverse proxy
    let publicIP: IP  // the only Uber address the world knows
    var tls: TLSSession  // the TLS tunnel it terminates
    var routes: [String: String]  // path prefix -> service name
    var lb: LoadBalancer  // the load balancer behind it
    var cache: [String: HTTPResponse] = [:]  // GET path -> saved answer
    var cacheHits = 0  // how many answers came from the cache
    mutating func handle(_ record: TLSRecord) -> HTTPResponse {  // one encrypted request arrives
        guard let plain = tls.open(record) else { return HTTPResponse(status: 400, body: "tampered, rejected") }  // integrity
        let parts = plain.split(separator: " ").map(String.init)  // "METHOD PATH TOKEN"
        guard let user = verifyToken(parts[2]) else { return HTTPResponse(status: 401, body: "unauthorized") }  // auth
        let req = HTTPRequest(method: parts[0], path: parts[1], user: user)  // rebuild the request
        let cacheable = req.method == "GET" && req.path.hasPrefix("/v1/drivers")  // only nearby-drivers is cached
        if cacheable, var hit = cache[req.path] {  // already have a fresh answer?
            cacheHits += 1  // count it
            hit.headers["X-Cache"] = "HIT"  // tell the client it came from the cache
            return hit  // no app server needed
        }
        guard let prefix = routes.keys.first(where: { req.path.hasPrefix($0) }) else {  // ROUTING by path
            return HTTPResponse(status: 404, body: "no route")  // no service for that path
        }
        let server = lb.roundRobin()  // LOAD BALANCER picks an app server
        print("     proxy: \(req.method) \(req.path) -> \(routes[prefix]!) on \(server.name) (\(server.privateIP), hidden)")
        var resp = server.handle(req)  // forward over the private network (plain HTTP inside)
        let squeezed = compress(resp.body)  // COMPRESS the body
        resp.headers["Content-Encoding"] = "toy-dict, saved \(squeezed.saved) chars"  // say how it shrank
        resp.headers["X-Cache"] = "MISS"  // this one came fresh from a server
        if cacheable { cache[req.path] = resp }  // CACHE GET answers for next time
        return resp  // back to the phone (re-encrypted with TLS in real life)
    }
}

// MARK: STEP 16 - Live driver location: polling vs SSE vs WebSockets
// Notes: "websockets, unnecessary traffic, two way communication", "push",
//        "http polling server sent events websockets communication model"
// OSI LAYER: 7 Application (all three ride on TCP, Layer 4)
//  - HTTP polling: the phone asks "any news?" every second. Mostly "no" = waste.
//  - Server-Sent Events (SSE): one request, then the server PUSHES news. One way.
//  - WebSocket: one upgrade handshake, then BOTH sides push any time. Two way.
// REAL WORLD: connect URLSessionWebSocketTask on the iPhone + a WebSocket server,
//             and Apple Push Notifications (APNs) for when the app is closed.

struct StreamStats {  // the counts for one communication model
    let name: String  // which model
    var messages = 0  // total messages on the wire
    var wasted = 0  // messages that carried no news
}
func simulateStreams(seconds: Int, updateEvery: Int) -> [StreamStats] {  // all 3 side by side
    var poll = StreamStats(name: "HTTP polling (every 1s)")  // counters for polling
    var sse = StreamStats(name: "Server-Sent Events")  // counters for SSE
    var ws = StreamStats(name: "WebSocket")  // counters for WebSocket
    sse.messages += 1  // SSE: one GET opens the stream
    ws.messages += 2  // WS: the upgrade request + "101 Switching Protocols" reply
    for t in 1...seconds {  // tick once per second
        let moved = t % updateEvery == 0  // does the driver have a new position this second?
        poll.messages += 2  // polling: a request AND a response every second
        if !moved { poll.wasted += 2 }  // nothing new: both messages were wasted
        if moved { sse.messages += 1; ws.messages += 1 }  // push models only talk when there is news
    }
    ws.messages += 1  // two way: the rider sends "I'm by the blue door" up the same socket
    return [poll, sse, ws]  // the three results
}

// MARK: STEP 17 - UDP for low-latency GPS pings (TCP vs UDP)
// Notes: "UDP low latency", "speed", "tcp v udp"
// OSI LAYER: 4 Transport
// UDP = no handshake, no resend, no ordering. Just fire the packet.
// For GPS, a late update is useless - a fresher one is coming anyway - so a lost
// ping is fine and nobody waits. TCP would resend the old ping and make the
// newer ones wait behind it ("head-of-line blocking").
// REAL WORLD: connect Network.framework NWConnection with .udp, or QUIC (HTTP/3).

func gpsRace(pings: Int, lossPercent: UInt64) -> (udpGot: Int, udpMs: Double, tcpGot: Int, tcpMs: Double) {
    var udpGot = 0, udpTotal = 0.0  // UDP: pings received + their total delay
    var tcpTotal = 0.0, backlog = 0.0  // TCP: total delay + extra wait caused by resends
    for _ in 0..<pings {  // send each GPS ping over both
        let lost = dice.chance(lossPercent)  // the same bad network for both
        if !lost { udpGot += 1; udpTotal += 20 }  // UDP: arrives in 20 ms, or never
        if lost { backlog += 200 }  // TCP: waits ~200 ms to notice the loss and resend
        tcpTotal += 20 + backlog  // 20 ms travel + any waiting behind a resend
        backlog = max(0, backlog - 100)  // the queue slowly drains
    }
    return (udpGot, udpTotal / Double(max(udpGot, 1)), pings, tcpTotal / Double(pings))  // the averages
}

// MARK: STEP 18 - THE RIDE: run every step in order
// OSI LAYER: all of them, from the bottom up, then back down
// REAL WORLD: this is what happens in about one second when you tap "Request".

print("UberSim - one pretend ride through all 7 OSI layers (no real network used)")  // title

banner("STEP 1  Network + interface", "L1 Physical / L2 Data Link / L3 Network")  // step title
print("  \(riderPhone.name): interface \(riderPhone.nic.name), MAC \(riderPhone.nic.mac), IP none yet")  // the door
print("  Wi-Fi radio joins 'HomeWiFi' - radio waves carry the bits (Layer 1)")  // physical layer

banner("STEP 2+3  DHCP + subnetting", "L7 protocol that configures L3")  // step title
let homeNet = Subnet(base: IP("192.168.1.0"), prefix: 24)  // the home network block
var dhcp = DHCPServer(subnet: homeNet)  // the router's DHCP service
let lease = dhcp.handle(mac: riderPhone.nic.mac)  // the phone asks, the router answers
riderPhone.nic.ip = lease.ip  // the phone now has an address
print("  phone IP \(lease.ip)  mask \(IP(raw: homeNet.mask))  gateway \(lease.gateway)  DNS \(lease.dns)")  // settings
let edgeIP = IP("203.0.113.10")  // Uber's public front door (a documentation-only IP)
print("  gateway \(lease.gateway) in my subnet? \(homeNet.contains(lease.gateway)) -> talk directly")  // neighbor
print("  Uber \(edgeIP) in my subnet? \(homeNet.contains(edgeIP)) -> send via the gateway")  // far away
check(homeNet.contains(lease.ip) && !homeNet.contains(edgeIP), "Subnet: phone is local, Uber is remote")  // self-check
check(dhcp.handle(mac: riderPhone.nic.mac).ip == lease.ip, "DHCP: same phone gets the same IP again")  // self-check

banner("STEP 4  ARP", "L2 Data Link")  // step title
var arp = ARP()  // the phone's ARP helper
let lan = [riderPhone, homeRouter]  // the devices on the Wi-Fi
let gatewayMAC = arp.resolve(lease.gateway, onLAN: lan)  // first time: must broadcast
_ = arp.resolve(lease.gateway, onLAN: lan)  // second time: comes from the cache
check(gatewayMAC == homeRouter.nic.mac, "ARP: 192.168.1.1 -> router MAC BB:BB:BB:00:00:01")  // self-check
check(arp.broadcasts == 1, "ARP: second lookup used the cache (only 1 broadcast)")  // self-check
check(arp.resolve(IP("192.168.1.99"), onLAN: lan) == nil, "ARP: unknown IP gets no reply")  // self-check

banner("STEP 5  NAT + NAT table + NAT gateway", "L3 Network / L4 Transport")  // step title
var homeNAT = NATRouter(publicIP: IP("198.51.100.7"))  // the home router's NAT, with its public IP
let syn = Packet(srcIP: lease.ip, srcPort: 51515, dstIP: edgeIP, dstPort: 443, payload: "SYN")  // phone's first packet
let outPkt = homeNAT.outbound(syn)  // the router rewrites it
print("  before NAT: \(syn.srcIP):\(syn.srcPort) -> \(syn.dstIP):\(syn.dstPort)")  // private source
print("  after  NAT: \(outPkt.srcIP):\(outPkt.srcPort) -> \(outPkt.dstIP):\(outPkt.dstPort)")  // public source
print("  NAT table:  public port \(outPkt.srcPort) = \(syn.srcIP):\(syn.srcPort)")  // the saved swap
let reply = Packet(srcIP: edgeIP, srcPort: 443, dstIP: homeNAT.publicIP, dstPort: outPkt.srcPort, payload: "SYN-ACK")
let back = homeNAT.inbound(reply)  // the reply finds its way home via the table
print("  reply comes back to \(back!.dstIP):\(back!.dstPort) - internet public app data reaches the phone")  // delivered
check(back?.dstIP == lease.ip && back?.dstPort == 51515, "NAT: 192.168.1.20:51515 <-> 198.51.100.7:40000 and back")
let stranger = Packet(srcIP: IP("6.6.6.6"), srcPort: 9999, dstIP: homeNAT.publicIP, dstPort: 41234, payload: "hi?")
check(homeNAT.inbound(stranger) == nil, "NAT: unsolicited packet from the internet is dropped")  // self-check
var natGateway = NATRouter(publicIP: IP("203.0.113.50"))  // the cloud NAT gateway for private servers
let update = natGateway.outbound(Packet(srcIP: IP("10.0.2.11"), srcPort: 33000,  // app-1 asks for...
    dstIP: IP("192.0.2.80"), dstPort: 443, payload: "GET /security-updates"))  // ...software updates
print("  app-1 10.0.2.11 fetches updates as \(update.srcIP):\(update.srcPort) (private -> external is OK)")  // out
let probe = Packet(srcIP: IP("6.6.6.6"), srcPort: 1, dstIP: natGateway.publicIP, dstPort: 40001, payload: "hack")
check(natGateway.inbound(probe) == nil, "NAT gateway: outsiders cannot start a connection to app-1")  // self-check

banner("STEP 6  DNS", "L7 Application over UDP port 53")  // step title
var dns = DNSResolver(records: ["api.uber.sim": edgeIP])  // the phone book has one entry
let apiIP = dns.resolve("api.uber.sim")  // first lookup walks the tree
_ = dns.resolve("api.uber.sim")  // second lookup hits the cache
check(apiIP == edgeIP && dns.lookups == 1, "DNS: api.uber.sim -> 203.0.113.10, repeat answered from cache")

banner("STEP 7  TCP", "L4 Transport")  // step title
var tcp = TCPConnection()  // a new connection to api.uber.sim:443
tcp.handshake()  // SYN, SYN-ACK, ACK
check(tcp.log == ["SYN", "SYN-ACK", "ACK"] && tcp.established, "TCP: handshake order SYN -> SYN-ACK -> ACK")
let (got, resends, windows) = tcp.send(count: 40, lossPercent: 8, receiverWindow: 12)  // send 40 segments
print("  window each round: \(windows)")  // shows slow start, halving, +1
print("  delivered \(got.count)/40 segments, \(resends) resent after loss")  // the totals
check(got == Array(0..<40), "TCP: all 40 segments arrived in order despite loss")  // self-check
check(windows.max()! <= 12, "TCP flow control: never more than the receiver's window of 12")  // self-check
check(resends > 0 && zip(windows, windows.dropFirst()).contains { $1 < $0 }, "TCP congestion: loss made the window shrink")

banner("STEP 8  HTTPS / TLS at the edge", "L5 Session / L6 Presentation")  // step title
var phoneTLS = TLSSession()  // the phone's side of the tunnel
check(phoneTLS.handshake(serverName: "api.uber.sim", certName: "api.uber.sim"), "TLS: real server, shared key agreed")
var fakeTLS = TLSSession()  // an impostor tries to pretend to be Uber
check(!fakeTLS.handshake(serverName: "api.uber.sim", certName: "evil.sim"), "TLS: fake certificate refused")
let record = phoneTLS.seal("login jeff password=hunter2")  // sensitive data, sealed
let spyView = record.cipher.prefix(12).map { String(format: "%02x", $0) }.joined()  // what an interceptor sees
print("  packet interceptor sees: \(spyView)... (gibberish)")  // no password visible
check(phoneTLS.open(record) == "login jeff password=hunter2", "TLS: edge decrypts the real message")  // self-check
var tampered = record  // an attacker copies the record...
tampered.cipher[0] ^= 1  // ...and flips one single bit
check(phoneTLS.open(tampered) == nil, "TLS integrity: tampered packet (1 bit flipped) rejected")  // self-check

banner("STEP 9  Authentication", "L7 Application")  // step title
let token = issueToken(user: "jeff")  // password was right, so the server signs a token
print("  login ok -> token \(token)")  // the phone stores this
check(verifyToken(token) == "jeff", "Auth: real token accepted as 'jeff'")  // self-check
check(verifyToken("jeff.deadbeef") == nil && verifyToken(nil) == nil, "Auth: forged or missing token refused")

banner("STEP 10  Firewall + security groups", "L3 Network / L4 Transport")  // step title
var fw = Firewall(rules: [  // the allow list
    Rule(from: "internet", to: "edge", port: 443),  // the public may reach the proxy on HTTPS only
    Rule(from: "edge", to: "app", port: 8080),  // the proxy may reach app servers
    Rule(from: "app", to: "db", port: 5432),  // app servers may reach the database
])  // end of the rules
let zones = ["internet", "edge", "app", "db"]  // the 4 zones in our access matrix
print("  ACCESS MATRIX  (row may talk to column on port; - = DROP)")  // matrix title
print("  from\\to   " + zones.map { $0.padding(toLength: 9, withPad: " ", startingAt: 0) }.joined())  // header row
for f in zones {  // one row per sender zone
    let cells = zones.map { t in fw.rules.first { $0.from == f && $0.to == t }.map { String($0.port) } ?? "-" }  // ports
    print("  " + f.padding(toLength: 10, withPad: " ", startingAt: 0)  // row label
        + cells.map { $0.padding(toLength: 9, withPad: " ", startingAt: 0) }.joined())  // the cells
}
check(fw.allows("internet", "edge", 443), "Firewall: internet -> edge:443 allowed")  // self-check
check(!fw.allows("internet", "app", 8080), "Firewall: internet cannot skip the proxy to reach app servers")
check(!fw.allows("internet", "db", 5432), "Firewall: internet -> database dropped")  // self-check
print("  someone points the edge straight at the database by mistake:")  // a debugging story
print("  result: " + connectWithTimeout(&fw, "edge", "db", 5432))  // drops look like timeouts
check(fw.dropped == 5, "Firewall: 5 packets dropped (2 bad + 3 retries), none answered")  // self-check
var audit = fw  // a copy to test a risky change
audit.rules.append(Rule(from: "internet", to: "db", port: 5432))  // "temporary" opening: a boundary accident
let accidents = audit.rules.filter { $0.from == "internet" && $0.to != "edge" }  // anything public past the edge
print("  boundary audit found: \(accidents.map { "\($0.from) -> \($0.to):\($0.port)" })")  // show it
check(accidents.count == 1 && fw.permits("internet", "db", 5432) == false, "Boundary audit: caught internet -> db")

banner("STEP 11-15  Reverse proxy + load balancer + API + Uber services", "L7 Application")  // step title
let apps = [AppServer("app-1", "10.0.2.11"), AppServer("app-2", "10.0.2.12"), AppServer("app-3", "10.0.2.13")]
var edge = EdgeProxy(publicIP: edgeIP, tls: phoneTLS,  // the proxy shares the TLS key from Step 8
    routes: ["/v1/drivers": "driver-svc", "/v1/rides": "ride-svc"], lb: LoadBalancer(servers: apps))  // routes + LB
func call(_ method: String, _ path: String, _ tk: String) -> HTTPResponse {  // the phone sends one request
    let r = edge.handle(phoneTLS.seal("\(method) \(path) \(tk)"))  // encrypt it and send it to the edge
    print("  \(method) \(path) -> \(r.status) \(r.body)  [cache \(r.headers["X-Cache"] ?? "-")]")  // show the answer
    return r  // hand it back
}
check(nearestDriver(x: 0, y: 0)?.name == "Leo", "Matching: Leo picked (Ana is closer but busy)")  // self-check
check(price(km: 5.2, minutes: 14, surge: 1.3) == 16.82, "Pricing: 2.50 + 5.2km + 14min, x1.3 surge = $16.82")
let r1 = call("GET", "/v1/drivers/nearby", token)  // GET #1: nearby drivers
let r2 = call("GET", "/v1/drivers/nearby", token)  // same again: from the cache
check(r1.headers["X-Cache"] == "MISS" && r2.headers["X-Cache"] == "HIT", "Proxy cache: 2nd GET served from cache")
print("  compression: \(r1.headers["Content-Encoding"]!)")  // how much the body shrank
let r3 = call("POST", "/v1/rides", token)  // POST: request the ride
check(r3.status == 201 && r3.body.contains("Leo"), "API: POST /v1/rides -> 201 Created, driver Leo")  // self-check
let r4 = call("GET", "/v1/rides/r1", token)  // GET #2: ride status
check(r4.body.contains("accepted"), "API: GET /v1/rides/r1 -> accepted")  // self-check
check(call("GET", "/v1/rides/r1", "-").status == 401, "Auth at the edge: no token -> 401")  // self-check
var badRecord = phoneTLS.seal("GET /v1/rides/r1 \(token)")  // a real request...
badRecord.tag ^= 0xFF  // ...whose tag gets changed in flight
check(edge.handle(badRecord).status == 400, "Proxy: tampered TLS record rejected with 400")  // self-check
trips["r1"]!.advance(to: .arriving)  // Leo drives to the rider
trips["r1"]!.advance(to: .inTrip)  // rider gets in
trips["r1"]!.advance(to: .completed)  // arrived at the destination
let r5 = call("GET", "/v1/rides/r1", token)  // GET #3: ride status again
check(r5.body.contains("completed"), "API: GET /v1/rides/r1 -> completed")  // self-check
let finalFare = price(km: 5.6, minutes: 16, surge: 1.3)  // the real trip was a little longer
let paid = payments.capture(trip: "r1", amount: finalFare)  // charge the card
if paid { trips["r1"]!.advance(to: .paid) }  // ride fully done
print("  quoted $\(trips["r1"]!.fare), charged $\(finalFare)")  // the receipt
check(paid && payments.charged["r1"] == finalFare, "Payment: hold then capture $\(finalFare) (within 20% of quote)")
check(trips["r1"]!.history == [.requested, .accepted, .arriving, .inTrip, .completed, .paid], "Trip lifecycle in order")
check(!payments.capture(trip: "r1", amount: 1), "Payment: cannot charge the same ride twice")  // self-check
print("  served per app server: \(apps.map { "\($0.name)=\($0.handled)" })")  // the spread so far
var testLB = LoadBalancer(servers: [AppServer("a", "10.0.3.1"), AppServer("b", "10.0.3.2"), AppServer("c", "10.0.3.3")])
for _ in 0..<300 { testLB.roundRobin().handled += 1 }  // 300 pretend requests
print("  round-robin 300 requests: \(testLB.servers.map { "\($0.name)=\($0.handled)" })")  // even split
check(testLB.servers.allSatisfy { $0.handled == 100 }, "Load balancer: round-robin 100/100/100")  // self-check
testLB.servers[0].active = 5; testLB.servers[1].active = 1; testLB.servers[2].active = 3  // pretend loads
check(testLB.leastLoaded().name == "b", "Load balancer: least-loaded picks 'b' (1 active vs 5 and 3)")

banner("STEP 16  Polling vs SSE vs WebSockets", "L7 Application on L4 TCP")  // step title
let streams = simulateStreams(seconds: 60, updateEvery: 5)  // 60 s trip, driver moves every 5 s
for s in streams {  // print one line per model
    print("  \(s.name.padding(toLength: 24, withPad: " ", startingAt: 0)) messages \(s.messages)  wasted \(s.wasted)")
}
check(streams[0].messages == 120 && streams[0].wasted == 96, "Polling: 120 messages, 96 wasted (80%)")  // self-check
check(streams[1].messages == 13, "SSE: 13 messages (1 open + 12 pushes)")  // self-check
check(streams[2].messages == 15, "WebSocket: 15 messages (2 upgrade + 12 pushes + 1 rider->server)")  // self-check

banner("STEP 17  UDP GPS pings vs TCP", "L4 Transport")  // step title
let race = gpsRace(pings: 100, lossPercent: 10)  // 100 pings on a network that loses 10%
print("  UDP: \(race.udpGot)/100 arrived, average delay \(String(format: "%.0f", race.udpMs)) ms (lost ones skipped)")
print("  TCP: \(race.tcpGot)/100 arrived, average delay \(String(format: "%.0f", race.tcpMs)) ms (waits for resends)")
check(race.udpGot < 100 && race.tcpGot == 100 && race.udpMs < race.tcpMs, "UDP faster but lossy; TCP complete but slower")

// MARK: STEP 19 - OSI summary, TCP vs UDP, and review questions
// Notes: "osi model mental model", "tcp v udp", "questions"
// OSI LAYER: all 7 - this is the mental model that ties it together.
// REAL WORLD: keep this table in your head when you debug: start at the bottom.

banner("STEP 19  OSI summary", "all layers")  // step title
let osi: [(String, String)] = [  // layer -> what we built for it
    ("7 Application", "DHCP, DNS, HTTP GET/POST, API, auth token, load balancer, services, polling/SSE/WS"),
    ("6 Presentation", "TLS encryption + integrity tag, compression"),
    ("5 Session", "TLS session (shared key), WebSocket stays open"),
    ("4 Transport", "TCP handshake/loss/flow/congestion, UDP GPS, ports, NAT ports, firewall ports"),
    ("3 Network", "IP addresses, subnetting, gateway, NAT table, NAT gateway, firewall/security groups"),
    ("2 Data Link", "MAC addresses, ARP, network interface"),
    ("1 Physical", "Wi-Fi radio waves / cables carrying the bits"),
]  // end of the table
for (layer, what) in osi { print("  \(layer.padding(toLength: 15, withPad: " ", startingAt: 0)) \(what)") }  // print it
print("\n  TCP vs UDP")  // small comparison table
print("  TCP: handshake, resends lost data, in order, slower  -> API calls, payments, login")  // TCP use
print("  UDP: no handshake, no resend, any order, fastest      -> GPS pings, voice, video, games, DNS")  // UDP use

let questions: [(String, String)] = [  // review questions with answers (your "questions" note)
    ("What does DHCP give the phone?", "An IP address, subnet mask, gateway and DNS server."),
    ("IP vs MAC?", "IP = mailing address across networks (L3). MAC = local door serial number (L2)."),
    ("Why do we need ARP?", "Wi-Fi frames need a MAC; ARP finds the MAC that owns an IP."),
    ("What does NAT do?", "Swaps private IP:port for a public one and remembers it in the NAT table."),
    ("Why can app servers download updates but not be reached?", "A NAT gateway only allows outbound-started traffic."),
    ("What are the 3 TCP handshake messages?", "SYN, SYN-ACK, ACK."),
    ("Flow control vs congestion control?", "Flow = receiver's limit. Congestion = sender slows for the network."),
    ("What 3 things does TLS give you?", "Encryption, integrity (tamper detection), server authentication."),
    ("Why does a firewall drop cause a timeout?", "DROP sends no reply, so the client waits until it gives up."),
    ("What does a reverse proxy do?", "TLS termination, routing, caching, compression, hides servers."),
    ("Round-robin vs least-loaded?", "Taking turns vs picking the least busy server."),
    ("Polling vs WebSocket?", "Polling asks every second (mostly wasted); WebSocket pushes only news, both ways."),
    ("Why UDP for GPS?", "Low latency; a lost ping is replaced by the next one, so no waiting."),
]  // end of questions
print("\n  REVIEW QUESTIONS")  // section title
for (i, qa) in questions.enumerated() { print("  Q\(i + 1). \(qa.0)\n      A: \(qa.1)") }  // print each Q and A

print("\n==== SELF-CHECKS: \(passCount) passed, \(failCount) failed ====")  // the final score
if failCount > 0 { exit(1) }  // tell the computer something broke
