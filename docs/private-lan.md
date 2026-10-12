# Explicit trusted private-LAN access

Direct CLI Web binding and Host restrictions remain localhost-only. TUI's
explicit Chat Start defaults to LAN binding and passes the opt-in flag;
`DS41F_CHAT_BIND=loopback` restores TUI localhost-only launch. The TUI shows
connection URLs and a trusted-network warning; see
[operator control](operator-live-observability.md#process-and-control-boundary).
For manual launch, opt in:

```sh
python3 -m ds41f_mlx.web --host 0.0.0.0 --port 8080 --allow-private-lan \
  --runtime-url http://127.0.0.1:8000
```

On another PC, open `http://<server-private-IP>:8080/` (for example
`http://192.168.68.56:8080/`). Use the server's numeric private IP, not a DNS/mDNS
alias. `--allow-private-lan` does not change the default bind address: specify
`--host 0.0.0.0`, a private interface IP, or `::` explicitly.

For the standard-OFF runtime, use the provisioned non-editable environment's
canonical isolated launcher (see [operations](operations.md)):

```sh
/path/to/off-env/bin/ds41f start --profile standard-off --host 0.0.0.0 --port 8000
```

Do not substitute direct development-checkout execution or refresh a local seal.
The [source edition](archive/editorial-baseline/private-lan.md.txt) retains the
earlier generic module-launch example; LAN opt-in does not bypass admission.

It does not require the Web flag. Keeping runtime on loopback is also possible;
remote browsers communicate only with the Web proxy. The separate MTP singleton
profile retains its existing loopback constraint. Ordinary `mtp-serving-v1`
also supports direct LAN binding without the Web flag or proxy; see
[ordinary MTP serving](mtp-production-serving.md#direct-trusted-private-lan-serving).
Its trusted private-LAN / single-operator scope provides no authentication,
adversarial multi-user isolation or Internet serving.

## Boundary and trust

With the flag, Web accepts socket peers and numeric Host addresses only in
RFC1918 IPv4 (`10/8`, `172.16/12`, `192.168/16`), IPv6 ULA (`fc00::/7`) or loopback.
IPv4-mapped IPv6 peers are normalized. Link-local, CGNAT, documentation/reserved
networks and public IPs are not treated as private LAN. Arbitrary DNS names are
rejected to retain DNS-rebinding protection. Cross-origin mutations, cross-site
Fetch Metadata and non-JSON POST remain rejected; CSP/body caps/tool certificates,
reservation/dedupe, native reconciliation and no-replay behavior remain intact.
Uvicorn proxy-header trust is disabled; spoofed X-Forwarded-For cannot bypass the
socket-peer filter.

**This is not authentication or internet serving.** All allowed LAN devices are
trusted to access the application. Protect both exposed ports with a firewall;
do not port-forward them. NAT/proxies that rewrite a public peer into a private
peer cannot be distinguished from trusted LAN by this filter. The runtime wildcard
bind itself has no new private-peer filter or authentication. HTTP is unencrypted.

Browser conversations, original images and uncertainty journals now live in each
tab's memory only. Another PC/tab or reload does not inherit them. There is no
shared persistent history/archive or runtime-restart restore. The Web process
holds only bounded correlation leases/effect receipts, with idle/orphan retirement.

## HTTP browser capabilities

Private-IP HTTP is usually an insecure context. Chat, original-byte image
picker/paste/drop, multiple conversations, streaming and Stop work there.
Reload starts a fresh UI; save/archive/fresh-restore controls have been removed.

- UUIDs use `crypto.getRandomValues` when secure-context `randomUUID` is absent;
  there is no Math.random fallback.
- Each conversation has a tab-local busy fence. No origin-wide IndexedDB mutex
  or shared persisted conversation remains. Different conversations/tabs may
  submit concurrently; expected-count/application UUID fences and the existing
  single-active Scheduler, not browser locks, own admission/queue/settlement.
- Living tabs renew internal idle leases; lost cleanup beacons are covered by
  server-side bounded nonbusy orphan retirement, never automatic resend.
- Clipboard write may be unavailable: Copy reports **Copy unavailable**. Select
  the text and use Ctrl/Cmd+C, or use HTTPS. OS-provided paste events can still
  deliver original image files; no clipboard transcoding is introduced.

For HTTPS with a certificate trusted by the browsing PCs:

```sh
python3 -m ds41f_mlx.web --host 0.0.0.0 --allow-private-lan --port 8443 \
  --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

Both TLS files must be supplied together. The certificate must cover the numeric
IP used in the URL. This does not add authentication or permit arbitrary Host
names. Forwarded headers are still not trusted.

## Qualification

`tools/qualify_private_lan.py --web http://<private-IP>:8081/` uses an isolated real
Chrome context against an existing standard-OFF runtime. It requires an actually
insecure origin with Web Locks/randomUUID/Clipboard absent, checks cryptographic
UUIDs and two-tab transaction exclusion/release/reload, real generation, reported
Copy limitations, reconnect, canonical safe Stop, and fresh native restore.
`artifacts/private-lan/browser.json` records **historical PASS at `3d29cc4`** and zero replay /
full-cache repack. ASGI tests independently reject public peers, spoofed forwarding,
DNS/public Hosts, cross-origin/non-JSON mutations and non-opted-in LAN binds.
This is same-machine access through the real LAN interface plus simulated remote
peer boundary tests, not a claim that a second physical PC or every firewall/OS
was tested. No full R1, 200K/1M, full Vision, release or runtime promotion was run.
The prior everyday GUI qualification remains historical evidence at `9312cdd`;
this change is a separate explicit network/browser-capability extension.
Later [dynamic budgeting](dynamic-capability-budget.md) changes Web/runtime policy
and has separate evidence; the LAN receipt does not automatically qualify those changes.
