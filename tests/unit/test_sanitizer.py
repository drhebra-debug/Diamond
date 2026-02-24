from tools.sanitizer import sanitize_system_context


def test_truncation_and_flags():
    long_text = "claude " * 500 + " some policy text"
    sanitized, flags = sanitize_system_context(long_text, max_len=200)
    # sanitized should be truncated and include truncation marker
    assert "[TRUNCATED]" in sanitized
    assert "claude" in flags


def test_empty():
    s, f = sanitize_system_context("")
    assert s == ""
    assert f == []
