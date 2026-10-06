import pytest
from ds41f_mlx.web_tools import ToolError, _normalize_text_results

@pytest.mark.parametrize('text', [
    "You've hit Exa's free MCP rate limit. Create your own API key.",
    '{"results":[{"title":"quota","url":"","snippet":"rate limit"}]}',
    'Title: active URL\nURL: javascript:alert(1)\nnot a public source',
])
def test_quota_auth_or_url_less_text_is_not_successful_search(text):
    with pytest.raises(ToolError, match='no usable source URLs'):
        _normalize_text_results(text, provider='exa', max_results=5)

def test_intro_blocks_do_not_hide_real_sources():
    text = '\n\n'.join(['intro']*10 + ['Title: Python\nURL: https://docs.python.org/3/\nOfficial docs'])
    results = _normalize_text_results(text, provider='exa', max_results=1)
    assert results[0]['url'] == 'https://docs.python.org/3/'
    assert results[0]['snippet'] == 'Official docs'
