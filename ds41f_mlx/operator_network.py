"""Best-effort connection URLs for the local Chat launcher; no external probes."""
import ipaddress
import socket

NETWORKS = tuple(ipaddress.ip_network(n) for n in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))


def lan_addresses():
    candidates = set()
    try:
        candidates.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    # UDP connect selects a local route/source address without sending a packet.
    # This also works on Macs whose hostname resolves only to loopback.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(('192.168.0.1', 9))
            candidates.add(sock.getsockname()[0])
    except OSError:
        pass
    return sorted(value for value in candidates if any(ipaddress.ip_address(value) in net for net in NETWORKS))


def chat_endpoint(host, port, *, allow_private_lan=False, tls=False, runtime_url=None):
    scheme = 'https' if tls else 'http'
    local_host = '127.0.0.1' if host in ('0.0.0.0', 'localhost') else '[::1]' if host in ('::', '::1') else host
    if ':' in local_host and not local_host.startswith('['):
        local_host = f'[{local_host}]'
    addresses = lan_addresses() if host == '0.0.0.0' and allow_private_lan else []
    if allow_private_lan and host not in ('0.0.0.0', '::', 'localhost', '127.0.0.1', '::1'):
        addresses = [host]
    urls = [f'{scheme}://{"[" + address + "]" if ":" in address else address}:{port}' for address in addresses]
    return dict(host=host, port=port, allow_private_lan=allow_private_lan,
                local_url=f'{scheme}://{local_host}:{port}', lan_urls=urls,
                scheme=scheme, runtime_url=runtime_url)
