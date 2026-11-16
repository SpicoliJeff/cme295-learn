# UberSim: a pretend Uber, built 100% in Swift code

A teaching project that follows one Uber ride through all 7 OSI layers. **It makes no real network connections.** Every router, server and packet is simulated in memory.

| Step | What it covers | OSI layer |
|---|---|---|
| 1 | Network, interface, IP vs MAC | 1, 2, 3 |
| 2 | Subnetting | 3 |
| 3 | DHCP (IP, gateway, DNS) | 7, which sets up 3 |
| 4 | ARP | 2 |
| 5 | NAT, NAT table, NAT gateway | 3 + 4 |
| 6 | DNS | 7 |
| 7 | TCP handshake, loss, flow control and congestion control | 4 |
| 8 | TLS at the edge (toy crypto), plus tamper rejection | 5 + 6 |
| 9 | Token authentication | 7 |
| 10 | Firewall, security groups, the access matrix and debug timeouts | 3 + 4 |
| 11-15 | HTTP, the API (GET, POST, GET), services, load balancer and reverse proxy | 7 |
| 16 | Polling vs SSE vs WebSockets | 7 |
| 17 | UDP GPS pings vs TCP | 4 |
| 19 | OSI summary and review questions | all |

To run it:

```
swiftc -O UberSim.swift -o ubersim && ./ubersim
```

It has 41 built-in self-checks, and all of them PASS. The crypto is a TOY, made for learning only.
