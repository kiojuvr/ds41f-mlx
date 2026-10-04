"""Public projection and ingress around the qualified guarded singleton.

No alternate cache/parser/lifecycle owner. Trusted single-operator loopback only.
"""
import asyncio
import json
import re
from fastapi.responses import JSONResponse
from ds41f_mlx.mtp_profile import PROFILE, LIMITS, certificate_projection, sequence
from .internal_mtp import InternalMTPQualificationBackend


def public_record(rec):
    raw = rec.to_json()
    keys = ('id','state','request_count','canonical_frontier','outcome_state',
            'request_fence','next_sequence')
    return {**{k:raw[k] for k in keys}, 'profile':PROFILE,
            'certificate':certificate_projection(rec.certificate, rec.reconstruction_body,
                rec.last_turn.get('response') if rec.last_turn else None)}


class ProfileConflict(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class LocalMTPBackend(InternalMTPQualificationBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from deepseek_recipe import ConversionOptions
        self.conversion_options = ConversionOptions(default_thinking_mode=False)

    def _settle(self, rec, trace):
        super()._settle(rec, trace)
        from ds41f_mlx.mtp_profile import validate_completed_calls
        try:
            choices = trace['response'].get('choices', [])
            if len(choices) != 1:
                raise ValueError('one choice required')
            validate_completed_calls(choices[0]['message'])
            rec.certificate['profile_supported'] = True
        except ValueError:
            # A schema-unsupported completed native output is a negative profile
            # outcome, not permission to execute or repair it. Core finally owns
            # retirement exactly as for other unrepresentable settlements.
            rec.unrecoverable = rec.poisoned = True
            rec.certificate.update(representable=False, executable_tools=False,
                                   profile_supported=False)
            trace['certificate'] = rec.certificate

    async def qualification_response(self, session_id, request, *, tokenizer, body=None, sequence=None):
        if sequence is None:
            raise ValueError('mandatory request sequence missing')
        rec = self.get_stateful_session(session_id)
        def identity_error(exc):
            code = {'request sequence body mismatch':'request_body_mismatch',
                    'expired or out-of-order request sequence':
                        ('request_expired' if sequence <= rec.consumed_sequence else 'request_sequence_gap')}.get(str(exc))
            if code or 'canonical prefix' in str(exc):
                raise ProfileConflict(code or 'canonical_prefix_mismatch') from exc
            raise exc
        from .request_fence import observe_retry
        try:
            observe_retry(rec, sequence, body)
        except ValueError as exc:
            identity_error(exc)
        except RuntimeError as exc:
            raise ProfileConflict('request_active') from exc
        if rec.reconstruction_body is not None and (rec.fence is None or sequence != rec.fence['sequence']):
            current = json.loads(body)
            previous = rec.reconstruction_body
            if current.get('tools') != previous.get('tools') or current.get('reasoning_effort', 'none') != previous.get('reasoning_effort', 'none'):
                raise ProfileConflict('retained_envelope_changed')
            if rec.certificate and rec.certificate['representable']:
                prefix = previous['messages'] + [rec.last_turn['response']['choices'][0]['message']]
                if current['messages'][:len(prefix)] != prefix:
                    raise ProfileConflict('certified_history_changed')
        try:
            response = await super().qualification_response(session_id, request,
                tokenizer=tokenizer, body=body, sequence=sequence)
        except ValueError as exc:
            identity_error(exc)
        if isinstance(response, JSONResponse):
            value = json.loads(response.body)
            if 'outcome_state' in value:
                rec = self.get_stateful_session(session_id)
                value['certificate'] = certificate_projection(rec.certificate,
                    rec.reconstruction_body, value.get('response'))
                value['profile'] = PROFILE
                value['body_sha256'] = rec.fence['body_sha256']
                t = rec.last_turn or {}
                value['metrics'] = {k:t.get(k) for k in ('generated','decode_s','prefill_handoff_s',
                    'load_s','cleanup_s','elapsed_s','prompt_replay','full_cache_repack')}
                value['metrics']['aligned_idle'] = bool(t.get('target_offsets')) and set(
                    t.get('target_offsets', []) + t.get('dspark_offsets', [])) == {t.get('canonical_frontier')}
                stats = t.get('mtp_stats', {})
                value['metrics']['considered_drafts'] = sum(stats.get('depth_drafted', []))
                value['metrics']['accepted_drafts'] = sum(stats.get('depth_accepted', []))
                value['metrics']['settlement'] = t.get('quiescence', {}).get('counters')
                return JSONResponse(value)
        return response


class LocalBoundary:
    """One no-queue bounded body/preparation slot; sends have finite stalls.

    Transport timeouts never revoke native ownership. Response finally blocks own
    shielded settlement. GET can observe busy while preparation/generation runs.
    """
    def __init__(self, app, *, authority, body_timeout=30, send_timeout=30):
        self.app, self.authority = app, authority
        self.body_timeout, self.send_timeout = body_timeout, send_timeout
        self.preparing = False

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        async def deny(status, code):
            await JSONResponse({'error':{'code':code,'message':code}}, status_code=status)(scope, receive, send)
        headers = scope['headers']
        def values(key):
            return [v for k,v in headers if k.lower() == key]
        if values(b'host') != [self.authority.encode()] or values(b'origin'):
            return await deny(400, 'local_authority_required')
        if values(b'upgrade'):
            return await deny(400, 'unsupported_capability')
        method, path = scope['method'], scope['path']
        sid = r'mtp_[0-9a-f]{32}_[0-9a-f]{32}'
        create = path == '/v1/sessions' and method == 'POST'
        chat = re.fullmatch(r'/v1/sessions/[^/]+/chat/completions', path) and method == 'POST'
        lifecycle = re.fullmatch(r'/v1/sessions/[^/]+', path) and path != '/v1/sessions/restore' and method in ('GET','DELETE')
        supported = create or chat or lifecycle or (method == 'GET' and path in ('/health','/v1/models'))
        known = path in ('/v1/chat/completions','/v1/responses','/v1/messages','/v1/sessions/restore',
                         '/docs','/redoc','/openapi.json','/_ds41f/diagnostics') or re.fullmatch(r'/v1/sessions/[^/]+/(persist|responses|messages)', path)
        if not supported:
            return await deny(400 if known else 404, 'unsupported_capability' if known else 'not_found')
        if scope.get('query_string'):
            return await deny(400, 'unsupported_query')
        if values(b'content-encoding'):
            return await deny(400, 'json_uncompressed_required')
        if method == 'GET' and (values(b'transfer-encoding') or values(b'content-length') not in ([], [b'0'])):
            return await deny(400, 'get_body_unavailable')
        if method in ('POST','DELETE'):
            if values(b'content-type') != [b'application/json'] or values(b'content-encoding'):
                return await deny(400, 'json_uncompressed_required')
        if chat:
            try:
                sequence(headers)
            except ValueError:
                return await deny(400, 'invalid_request_sequence')
        owned = False
        if method in ('POST','DELETE'):
            cl = values(b'content-length')
            if len(cl) > 1 or (cl and not re.fullmatch(rb'[0-9]+', cl[0])):
                return await deny(400, 'invalid_content_length')
            if cl and int(cl[0]) > LIMITS['body_bytes']:
                return await deny(413, 'body_limit')
            if self.preparing:
                return await deny(409, 'preparation_busy')
            self.preparing = owned = True
        total = 0
        body_complete = False
        deadline = asyncio.get_running_loop().time() + self.body_timeout
        async def bounded_receive():
            nonlocal total, body_complete
            if body_complete:
                return await receive()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise IngressError(408, 'body_timeout')
            try:
                event = await asyncio.wait_for(receive(), remaining)
            except TimeoutError as exc:
                raise IngressError(408, 'body_timeout') from exc
            if event['type'] == 'http.request':
                total += len(event.get('body', b''))
                if total > LIMITS['body_bytes']:
                    raise IngressError(413, 'body_limit')
                body_complete = not event.get('more_body', False)
            return event
        started = False
        async def bounded_send(event):
            nonlocal owned, started
            if event['type'] == 'http.response.start':
                started = True
                if owned:
                    self.preparing = owned = False
            await asyncio.wait_for(send(event), self.send_timeout)
        try:
            # DELETE bodies are unsupported, but still read under the ingress bound.
            if method == 'DELETE':
                while True:
                    event = await bounded_receive()
                    if event['type'] == 'http.disconnect':
                        return
                    if event.get('body'):
                        raise IngressError(400, 'delete_body_unavailable')
                    if not event.get('more_body'):
                        break
                await self.app(scope, receive, bounded_send)
            else:
                await self.app(scope, bounded_receive if owned else receive, bounded_send)
        except IngressError as exc:
            if not started:
                await deny(exc.status, exc.code)
            else:
                raise
        finally:
            if owned:
                self.preparing = False


class IngressError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code
