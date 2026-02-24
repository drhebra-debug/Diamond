from typing import List, Optional
import re


IMPORT_RE = re.compile(r"\b(import|from)\s+(?P<pkg>[a-zA-Z0-9_\.\-]+)")


def _detect_imported_packages(text: str) -> List[str]:
    if not text:
        return []
    pkgs = []
    for m in IMPORT_RE.finditer(text):
        pkg = m.group('pkg')
        # split dotted imports like 'openai.chat' -> take first segment
        base = pkg.split('.')[0]
        pkgs.append(base.lower())
    return pkgs


def select_skill_from_flags(flags: List[str], user_text: str) -> Optional[str]:
    """Select a concise skill name based on flags or content.

    Uses explicit flags and import-detection within `user_text` to avoid
    sending full policy documents into the model.
    Returns skill id (e.g., 'claude-developer-platform') or None.
    """
    fl = set(flags or [])
    lower = (user_text or "").lower()

    # detect explicit imports in the user text (code blocks)
    pkgs = _detect_imported_packages(user_text)

    # If imports explicitly mention non-Anthropic providers, avoid selecting Claude
    non_anthropic = {'openai', 'google', 'gpt', 'gemini'}
    if any(p in non_anthropic for p in pkgs) or any(p in non_anthropic for p in fl):
        return None

    # If imports mention Anthropic/Claude, prefer the Claude skill
    if 'anthropic' in pkgs or 'claude' in pkgs or 'anthropic' in fl or 'claude' in fl or 'claude' in lower:
        return 'claude-developer-platform'

    # Default: no special skill
    return None
