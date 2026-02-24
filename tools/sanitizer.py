from typing import Tuple, List


def sanitize_system_context(text: str, max_len: int = 1000) -> Tuple[str, List[str]]:
    """Trim oversized system reminders and extract mention flags.

    Returns (sanitized_text, flags)
    """
    if not text:
        return "", []

    flags = []
    lower = text.lower()
    # crude flag extraction
    for kw in ("claude", "anthropic", "openai", "google", "gemini"):
        if kw in lower:
            flags.append(kw)

    # Remove long repeated policy blocks by truncation
    sanitized = text.strip()
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "\n... [TRUNCATED]"

    return sanitized, flags
