"""Small regressions for the existing investigation launcher, not serving policy."""
from tools.observe_prefix_frontier import (structural_snapshot, first_structure_change,
                                         mismatch_region, converted_payload,
                                         reminder_structure, reminder_change)


def test_structure_extension_and_changed_field_no_content():
    key = b'test-key'
    old = {'messages': [{'role': 'user', 'content': 'private'}], 'tools': []}
    new = {'messages': old['messages'] + [{'role': 'assistant', 'content': 'answer'}], 'tools': []}
    a = structural_snapshot(old, key)
    assert first_structure_change(a, structural_snapshot(new, key)) is None
    new['messages'][0] = {'role': 'user', 'content': 'changed'}
    change = first_structure_change(a, structural_snapshot(new, key))
    assert change == {'field': 'messages', 'message_index': 0, 'changed_fields': ['content']}
    assert 'private' not in str(a)


def test_reminder_relocation_is_not_history_equivalence():
    from deepseek_recipe import Conversation, UserMessage
    block = '<system-reminder>Plan Mode - System Reminder\nread-only</system-reminder>'
    old = Conversation([UserMessage('question\n\n' + block)], thinking_mode=False)
    new = Conversation([UserMessage('question'), UserMessage('follow-up\n\n' + block)], thinking_mode=False)
    change = reminder_change(reminder_structure(old, b'key'), reminder_structure(new, b'key'), 0)
    assert change['non_reminder_text_equal']
    assert change['identical_reminder_on_later_message']
    assert change['previous_plan_reminder'] and not change['next_plan_reminder']
    assert first_structure_change(structural_snapshot(converted_payload(old), b'key'),
                                  structural_snapshot(converted_payload(new), b'key')) is not None


def test_actual_recipe_message_region():
    from types import SimpleNamespace
    from deepseek_recipe import Conversation, SystemMessage, UserMessage, DeepseekV41Encoding
    from ds41f_mlx.serving.server import load_v41_tokenizer
    from verification_support import tokenizer as provisioned_tokenizer
    tokenizer = provisioned_tokenizer()
    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    old = Conversation([SystemMessage('policy'), UserMessage('question\n\nreminder')], thinking_mode=False)
    new = Conversation([SystemMessage('policy'), UserMessage('question')], thinking_mode=False)
    a, b = encoding.encode(old), encoding.encode(new)
    from tools.observe_prefix_frontier import lcp
    prepared = SimpleNamespace(multimodal=None, token_ids=b,
                               conversation_request=SimpleNamespace(conversation=new))
    change = first_structure_change(structural_snapshot(converted_payload(old), b'key'),
                                    structural_snapshot(converted_payload(new), b'key'))
    region = mismatch_region(prepared, tokenizer, lcp(a, b), change)
    assert region['message_index'] == 1
    assert region['role'] == 'user'
    # End-of-message divergence can be on the renderer's next structural token.
    assert region['stable_prefix_before'] <= lcp(a, b) <= region['stable_prefix_through']
