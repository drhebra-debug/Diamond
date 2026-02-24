from skills.selector import select_skill_from_flags


def test_select_claude_when_flagged():
    assert select_skill_from_flags(["claude"], "do stuff") == "claude-developer-platform"


def test_no_skill_for_openai():
    assert select_skill_from_flags(["openai"], "use openai library") is None


def test_detect_imports_in_user_text():
    text = """```python
import openai
from anthopic import client
```"""
    # should detect 'openai' and avoid selecting claude
    assert select_skill_from_flags([], text) is None
