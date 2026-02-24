from skills.selector import select_skill_from_flags


def test_select_claude_when_flagged():
    assert select_skill_from_flags(["claude"], "do stuff") == "claude-developer-platform"


def test_no_skill_for_openai():
    assert select_skill_from_flags(["openai"], "use openai library") is None


def test_detect_imports_in_user_text():
    text = """```python
import openai
from anthropic import client
```"""
    # should detect 'openai' and avoid selecting claude
    assert select_skill_from_flags([], text) is None


def test_detect_file_uploads():
    # filenames mentioning 'anthropic' or 'claude' should prefer the claude skill
    assert select_skill_from_flags([], "Please inspect anthropic_client.py attached") == "claude-developer-platform"
    assert select_skill_from_flags([], "See claude_config.yaml") == "claude-developer-platform"
    # filenames mentioning openai should avoid selecting claude
    assert select_skill_from_flags([], "Here is openai_helper.py") is None
