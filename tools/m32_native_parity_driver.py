"""Run the M31 corpus through an installed real native recipe binding."""
import hashlib
import json
from pathlib import Path
import sys
import deepseek_recipe as d
import deepseek_recipe._native as native

fixtures = json.loads(Path(sys.argv[1]).read_text())
t = d.Tokenizer.from_file(sys.argv[2])
records = []
for case in fixtures:
    modes = ['token'] if 'tokens' in case else ['token', 'text', 'character']
    for mode in modes:
        options = d.ParsingOptions()
        options.stop_sequences = case['stops']
        options.parse_json_output = case.get('json', False)
        if case.get('reasoning'):
            options.reasoning_initial_stage = d.ReasoningStage.Reasoning
        p = d.StreamProcessor(d.ChatCompletionChunkGenerator('parity', 'v41', True, case.get('reasoning', False)), options, t)
        response = d.ChatCompletionResponse('parity', 'v41', 0, 0, 0)
        events = []
        def push(chunk):
            for out in p.push(chunk):
                value = json.loads(out.to_json())
                response.append(out)
                value['created'] = 0
                events.append(value)
        push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=4)))
        if mode == 'token':
            chunks = [d.InferenceChunk.token(i) for i in case.get('tokens', t.encode(case.get('text', '')))]
        elif mode == 'text':
            chunks = [d.InferenceChunk.text(case['text'], content_tokens=1)]
        else:
            chunks = [d.InferenceChunk.text(c, content_tokens=1) for c in case['text']]
        for chunk in chunks:
            if p.finished:
                break
            push(chunk)
        finish = case.get('finish', 'stop')
        if not p.finished and finish != 'eof':
            push(d.InferenceChunk.finish(d.InferenceFinishReason.Length if finish == 'length' else d.InferenceFinishReason.Stop))
        for out in p.finish():
            value = json.loads(out.to_json())
            response.append(out)
            value['created'] = 0
            events.append(value)
        body = json.loads(response.to_json())
        body['created'] = 0
        records.append(dict(name=case['name'], mode=mode, events=events, response=body, finished=p.finished))
        p.close()
print(json.dumps(dict(native_path=native.__file__, native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), records=records), ensure_ascii=False, indent=2))
