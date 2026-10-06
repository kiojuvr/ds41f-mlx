"""CPU-only transport/effect regressions. Not browser, Vision, or GPU evidence."""
import hashlib
import json

import pytest

from ds41f_mlx import web_tools as tools


class Socket:
    def getpeername(self): return ('93.184.216.34', 443)
    def settimeout(self, timeout): pass


class Response:
    status = 200
    def __init__(self, data, content_type, encoding='identity', location=None):
        self.data, self.content_type, self.encoding, self.location = data, content_type, encoding, location
        self.position = 0
    def getheader(self, name, default=None):
        return {'content-type': self.content_type, 'content-encoding': self.encoding,
                'Location': self.location}.get(name, default)
    def read1(self, size):
        result = self.data[self.position:self.position + size]
        self.position += len(result)
        return result


def network(monkeypatch, response, peer='93.184.216.34'):
    connections = []
    monkeypatch.setattr(tools, '_validate_public_url', lambda url: url)
    class Connection:
        def __init__(self, *args, **kwargs):
            self.sock = Socket()
            self.sock.getpeername = lambda: (peer, 443)
            self.closed, self.gets = False, []
            connections.append(self)
        def connect(self): pass
        def request(self, method, path, headers): self.gets.append((method, path, headers))
        def getresponse(self): return response
        def close(self): self.closed = True
    monkeypatch.setattr(tools.http.client, 'HTTPSConnection', Connection)
    return connections


@pytest.mark.parametrize(('kind', 'mime'), [('text', 'text/plain'), ('image', 'image/png'), ('pdf', 'application/pdf')])
@pytest.mark.parametrize('length', [4, 5, 6])
def test_shared_transfer_keeps_only_original_bytes_and_explicit_ceiling(monkeypatch, kind, mime, length):
    original = bytes(range(length))
    connections = network(monkeypatch, Response(original, mime))
    result = tools.acquire_public_resource('https://example.com/file', kind=kind, max_bytes=5)
    assert result.data == original[:5]
    assert result.truncated is (length > 5)
    assert result.sha256 == hashlib.sha256(original[:5]).hexdigest()
    assert result.source_url == result.url == 'https://example.com/file'
    assert result.redirects == 0
    assert connections[0].closed
    assert len(connections[0].gets) == 1
    headers = connections[0].gets[0][2]
    assert headers['Accept-Encoding'] == 'identity'
    assert not any('forwarded' in key.lower() for key in headers)


@pytest.mark.parametrize(('mime', 'kind', 'allowed'), [
    ('image/png;note=html', 'text', False),
    ('application/octet-stream;note=json', 'text', False),
    ('text/html; charset=UTF-8', 'text', True),
    ('application/ld+json', 'text', True),
    ('application/atom+xml', 'text', True),
    ('IMAGE/WEBP; parameter=foo', 'image', True),
    ('image/gif', 'image', False),
    ('application/pdf; version=1.7', 'pdf', True),
    ('text/plain;note=application/pdf', 'pdf', False),
])
def test_media_type_not_parameter_substring(mime, kind, allowed):
    assert tools._content_type_allowed(mime, kind) is allowed


def test_text_interpretation_does_not_trust_html_in_parameters():
    text = '<script>plain text, not HTML</script>'
    assert tools.readable_text(text, 'text/plain; note=html') == text
    assert tools.readable_text(text, 'text/html; charset=utf-8') == ''


@pytest.mark.parametrize('kind', ['text', 'image', 'pdf'])
def test_all_consumers_reject_private_socket_before_get(monkeypatch, kind):
    connections = network(monkeypatch, Response(b'bad', 'text/plain'), peer='127.0.0.1')
    with pytest.raises(tools.ToolError) as error:
        tools.acquire_public_resource('https://example.com/file', kind=kind, max_bytes=5)
    assert error.value.code == 'security_rejection'
    assert connections[0].closed and not connections[0].gets


@pytest.mark.parametrize('kind', ['text', 'image', 'pdf'])
def test_redirect_bound_is_one_initial_get_plus_bounded_redirects(monkeypatch, kind):
    response = Response(b'not read', 'text/plain', location='/next')
    response.status = 302
    connections = network(monkeypatch, response)
    with pytest.raises(tools.ToolError) as error:
        tools.acquire_public_resource('https://example.com/file', kind=kind, max_bytes=5, max_redirects=2)
    assert error.value.code == 'resource_ceiling'
    assert len(connections) == 3
    assert all(c.closed and len(c.gets) == 1 for c in connections)
    assert response.position == 0  # redirects are closed without reading a body


@pytest.mark.parametrize(('kind', 'mime'), [('text', 'text/plain'), ('image', 'image/png'), ('pdf', 'application/pdf')])
def test_compressed_body_never_read_or_decompressed(monkeypatch, kind, mime):
    response = Response(b'zip', mime, encoding='gzip')
    connections = network(monkeypatch, response)
    with pytest.raises(tools.ToolError) as error:
        tools.acquire_public_resource('https://example.com/file', kind=kind, max_bytes=5)
    assert error.value.code == 'unsupported_content'
    assert response.position == 0 and connections[0].closed


@pytest.mark.parametrize('address', ['100.64.0.1', '127.0.0.1', '169.254.169.254', '::1', '::ffff:127.0.0.1'])
def test_dns_rejects_every_nonpublic_address(monkeypatch, address):
    monkeypatch.setattr(tools.socket, 'getaddrinfo', lambda *args, **kwargs: [(None, None, None, None, (address, 443))])
    with pytest.raises(tools.ToolError) as error:
        tools._validate_public_url('https://example.com/')
    assert error.value.code == 'security_rejection'


@pytest.mark.parametrize('url', ['https://user:pass@example.com/', 'file:///etc/passwd', 'https://example.com:bad/', 'https://[broken/'])
def test_unsupported_url_has_no_dns_effect(monkeypatch, url):
    monkeypatch.setattr(tools.socket, 'getaddrinfo', lambda *a, **k: pytest.fail('invalid URL reached DNS'))
    with pytest.raises(tools.ToolError) as error: tools._validate_public_url(url)
    assert error.value.code == 'security_rejection'


def test_initial_dns_elapsed_time_is_part_of_absolute_deadline(monkeypatch):
    times = iter([0, 46])
    monkeypatch.setattr(tools, 'monotonic', lambda: next(times))
    monkeypatch.setattr(tools, '_validate_public_url', lambda url: url)
    monkeypatch.setattr(tools.http.client, 'HTTPSConnection', lambda *a, **k: pytest.fail('expired DNS budget opened a socket'))
    with pytest.raises(tools.ToolError) as error:
        tools.acquire_public_resource('https://example.com/', kind='text', max_bytes=5)
    assert error.value.code == 'resource_ceiling'


@pytest.mark.parametrize('kind', ['text', 'image', 'pdf'])
def test_redirect_private_target_revalidated_without_second_get(monkeypatch, kind):
    response = Response(b'not read', 'text/plain', location='https://private.example/')
    response.status = 302
    connections = network(monkeypatch, response)
    urls = []
    def validate(url):
        urls.append(url)
        if 'private.example' in url:
            raise tools.ToolError('private target', code='security_rejection')
        return url
    monkeypatch.setattr(tools, '_validate_public_url', validate)
    with pytest.raises(tools.ToolError) as error:
        tools.acquire_public_resource('https://example.com/file', kind=kind, max_bytes=5)
    assert error.value.code == 'security_rejection'
    assert urls == ['https://example.com/file', 'https://private.example/']
    assert len(connections) == 1 and connections[0].closed
    assert response.position == 0


class CounterTool:
    name = 'counter'
    schema = {}
    def __init__(self): self.effects = 0
    def run(self, args, context=None):
        self.effects += 1
        return tools.ToolResult('{}', {'tool': self.name})


def call(identity, name='counter', arguments='{}'):
    return {'id': identity, 'function': {'name': name, 'arguments': arguments}}


@pytest.mark.parametrize('bad', [call('first'), call('second', arguments='{'), call('second', name='unknown'), call('second', arguments={}), None])
def test_entire_batch_preflight_before_first_effect(bad):
    counter = CounterTool()
    registry = tools.ToolRegistry([counter])
    with pytest.raises(tools.ToolError): registry.execute_calls([call('first'), bad])
    assert counter.effects == 0


def test_execution_failure_is_classified_and_does_not_discard_prior_receipt():
    counter = CounterTool()
    class RejectedTool:
        name = 'rejected'
        schema = {}
        def run(self, args, context=None):
            raise tools.ToolError('private URL', code='security_rejection')
    messages, displays = tools.ToolRegistry([counter, RejectedTool()]).execute_calls([call('first'), call('second', name='rejected')])
    assert counter.effects == 1
    assert len(messages) == len(displays) == 2
    assert messages[0]['content'] == '{}'
    assert json.loads(messages[1]['content'])['error_code'] == 'security_rejection'
    assert displays[1]['error_code'] == 'security_rejection'


def test_mcp_overflow_is_explicit_not_silent_json_truncation(monkeypatch):
    class ProviderResponse:
        def __enter__(self): return self
        def __exit__(self, *args): self.closed = True
        def read(self, size):
            assert size == tools.MCP_RESPONSE_BYTES + 1
            return b'x' * size
    response = ProviderResponse()
    monkeypatch.setattr(tools, 'urlopen', lambda *args, **kwargs: response)
    with pytest.raises(tools.ToolError) as error:
        tools.HostedMCPClient('https://example.com/').call('tools/call')
    assert error.value.code == 'resource_ceiling' and response.closed
