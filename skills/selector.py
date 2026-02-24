from typing import List, Optional


def select_skill_from_flags(flags: List[str], user_text: str) -> Optional[str]:
    """Select a concise skill name based on flags or content.

    Returns skill id (e.g., 'claude-developer-platform') or None.
    """
    fl = set(flags or [])
    lower = (user_text or "").lower()

    # Prefer explicit Anthropic/Claude mentions
    if "anthropic" in fl or "claude" in fl or "claude" in lower or "anthropic" in lower:
        return "claude-developer-platform"

    # If code mentions openai or google, do not select claude skill
    if "openai" in fl or "google" in fl or "gpt" in lower:
        return None

    # Default: no special skill
    return None
