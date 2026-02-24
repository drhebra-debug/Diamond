from typing import List, Optional
import re


IMPORT_RE = re.compile(r"\b(import|from)\s+(?P<pkg>[a-zA-Z0-9_\.\-]+)")
FILENAME_RE = re.compile(r"\b(?P<name>[A-Za-z0-9_\-]+)\.(?P<ext>py|js|ts|ipynb|yaml|yml|json)\b", re.IGNORECASE)


def _detect_imported_packages(text: str) -> List[str]:
    if not text:
        return []
    pkgs = []
    for m in IMPORT_RE.finditer(text):
        pkg = m.group('pkg')
        base = pkg.split('.')[0]
        pkgs.append(base.lower())
    return pkgs


def _detect_filenames(text: str) -> List[str]:
    if not text:
        return []
    names = []
    for m in FILENAME_RE.finditer(text):
        names.append(m.group('name').lower())
    return names


def select_skill_from_flags(flags: List[str], user_text: str) -> Optional[str]:
    """Select a concise skill name based on flags or content.

    Uses explicit flags, import-detection within `user_text`, and filename
    detection (e.g., uploaded filenames like `anthropic_client.py`) to avoid
    sending full policy documents into the model. Returns skill id
    (e.g., 'claude-developer-platform') or None.
    """
    fl = set((flags or []))
    lower = (user_text or "").lower()

    # detect explicit imports in the user text (code blocks)
    pkgs = _detect_imported_packages(user_text)
    # detect filenames referenced in text or attachments
    fnames = _detect_filenames(user_text)

    # Consolidate detections
    detected = set(pkgs) | set(fnames) | set(p.lower() for p in fl)

    # If imports/filenames explicitly mention non-Anthropic providers, avoid selecting Claude
    non_anthropic = {'openai', 'google', 'gpt', 'gemini'}
    if any(any(na in d for d in detected) for na in non_anthropic):
        return None

    # If imports or filenames mention Anthropic/Claude (substring match), prefer the Claude skill
    if any(('anthropic' in d or 'claude' in d) for d in detected) or 'claude' in lower:
        return 'claude-developer-platform'

    # Default: no special skill
    return None
