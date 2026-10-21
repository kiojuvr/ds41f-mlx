# Explicit trusted private-LAN access

The bundled Browser Chat listener, proxy, TLS options and Browser LAN policy are
retired. No supported TUI/CLI action starts an application server or opens a Browser.

Ordinary `mtp-serving-v1` defaults to `127.0.0.1:8000`. To expose it deliberately,
use an installed runtime's canonical launcher with `--host <private-interface-IP>`
or `--host 0.0.0.0`. External clients use `http://<mac-private-IP>:8000/v1` for
Chat Completions; native Messages uses `POST /v1/messages` on the same Runtime.
See [operations](operations.md#network-and-shutdown) and
[ordinary MTP serving](mtp-production-serving.md#direct-trusted-private-lan-serving).

Runtime has **no authentication, TLS or private-peer filter**; wildcard bind
listens on every interface. Use trusted-private-LAN/firewall policy, never
Internet exposure or port forwarding. Singleton retains its loopback constraint.
Checkpoint/installed admission and graceful shutdown are unchanged. Successful
inference is not comprehensive Agent/Harness compatibility.

## Historical source excerpt

The following retained paragraph records the former Browser LAN documentation;
it does not authorize old module/profile launches. Source remains in Git history.

Do not substitute direct development-checkout execution or refresh a local seal.
The [source edition](archive/editorial-baseline/private-lan.md.txt) retains the
earlier generic module-launch example; LAN opt-in does not bypass admission.
