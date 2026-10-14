# Separate-machine trusted-LAN acceptance — 2026-10-14

Status: PASS — separate Windows direct-runtime checks and Mac operator observation.
This is bounded transport acceptance, not release qualification.

Windows client: BEELINK-N100, Windows NT 10.0.26200.0,
Ethernet IPv4 192.168.68.54. Mac target: 192.168.68.56:8000.
`Find-NetRoute` selected Ethernet, 192.168.68.0/22, next hop 0.0.0.0.
Curl used `--noproxy '*'`; transfer receipts show local IP .54,
remote IP .56, remote port 8000, proxy_used=0. This was a separate
physical Windows client, not Mac access to its own LAN address.

Run ID: `windows-lan-20261014T021737Z`.
Windows checkout: `5e504afbd66b7fb0047fc78d0eacbf10d854b602`;
this does not identify the installed Mac package.

| Check | Observed result |
| --- | --- |
| GET /health before/after | HTTP 200, ready, mtp-serving-v1, model_ready=true |
| GET /v1/models | HTTP 200, deepseek-v4.1-flash |
| POST /v1/chat/completions JSON | HTTP 200, LAN_OK, finish_reason=stop |
| POST /v1/chat/completions stream=true | HTTP 200, LAN + _OK chunks, stop, [DONE] |
| Final runtime state | active=0, queued=0, fatal_error=null |
| Web :8080 / and /api/status after user startup | HTTP 200; runtime_base_url=http://127.0.0.1:8000 |

The JSON request ran 02:17:39.997–02:17:42.844 UTC
(11:17:39.997–11:17:42.844 JST), source port 64015.
All direct checks finished by 02:17:43.731 UTC.
Requests explicitly used supported reasoning_effort=none and max_tokens=64;
runtime default reasoning remained max. No automatic generation retries.

Raw requests, response headers/bodies, UTC timings, curl socket/transfer
metadata, client identity and route are retained under
`artifacts/remote-lan-windows-20261014/`. The bounded receipts are committed
explicitly despite the general artifacts ignore rule; the downloaded Web HTML
remains local. Separate-machine LAN acceptance is closed for this boundary.
Reproduce from this Windows checkout with:

```powershell
./tools/accept_remote_lan.ps1 -BaseUrl http://192.168.68.56:8000 -OutputDirectory artifacts/remote-lan-windows-repeat
```

Mac operator observation was supplied by the user from the Mac's existing
loopback-only control endpoint. Relevant complete fields are transcribed in
`artifacts/remote-lan-windows-20261014/mac-operator-observation.json`.
The pasted response was truncated in the cache object; it is not represented
as a complete raw response. To reproduce the observation without exposing
the control port or starting another runtime:

```sh
curl -s http://127.0.0.1:18441/ds41f/status
curl -s 'http://127.0.0.1:18441/ds41f/logs?after=0&limit=128'
```

Mac reported READY, wildcard port 8000, ordinary MTP, no current/queued/active
requests, and last request `49a8003ebba349b69d6b735054f03bf0` DONE/stop.
Its last request had prompt_tokens=28, cached_tokens=0, generated=3 (the
operator count differs from wire usage's 2 completion tokens),
prompt_replay=0 and full_cache_repack=0. This corresponds to the final Windows
SSE request (prompt_tokens=28, cached_tokens=0, LAN_OK, stop/[DONE]). Mac cache
hits=17/misses=46/evictions=28 also exactly match the Windows final health
and Web-projected health. Mac recorded receive epoch 1791944263.600705 and
finish 1791944264.1391091 (02:17:43.601–02:17:44.139 UTC).
Windows finished collecting by 02:17:43.731 UTC: clocks are not synchronized
closely enough to derive cross-host latency. Correlation uses order, bounded
time window, token counts, final cache counters and settlement state.
The subsequently supplied Mac logs provide exact wire/settlement correlation:
runtime_id=`ec95059cae3f49a49acdd334fb2909f3`, sequences 71–76,
gap=false. Sequence 73 settlement request
`fbcbc22ab4b045dba73b058229c703fc` corresponds to the JSON completion ID
`chatcmpl-fbcbc22ab4b045dba73b058229c703fc`; sequence 76 settlement request
`7500194257544357986096b5e4502e95` matches the captured SSE ID
`chatcmpl-7500194257544357986096b5e4502e95`.
Both requests were received, finished reason=stop and settled published=True,
cancelled=False, error=None, replay=0, repack=0. Relevant log records are
retained in `mac-operator-logs.json` in the artifact directory. Operator lifecycle
IDs differ from settlement/wire IDs; they are not treated as interchangeable.
Combined with direct socket metadata from the separate physical client, these
observations establish real LAN arrival, model processing and clean settlement.

The initially refused Web :8080 connection preceded the user's Web startup;
the subsequent checks passed. No production defect has been established.
Browser conversation operation was not exercised: the UI automation inventory
exposed no browsers. Web HTTP/status checks do not establish browser interaction.
No runtime, network policy, firewall, API or client implementation was changed;
only this evidence note and the bounded acceptance collector were added.
