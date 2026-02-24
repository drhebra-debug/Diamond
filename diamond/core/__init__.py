import importlib, sys, os

# Ensure project root is on sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Import the real top-level `core` package and re-export its submodules
_core = importlib.import_module('core')

# Import commonly used submodules so `diamond.core.xxx` resolutions work
try:
    import core.usage as _usage
    import core.headers as _headers
    import core.errors as _errors
    import core.fingerprint as _fingerprint
    import core.capabilities as _capabilities
except Exception:
    # Best-effort: ignore if specific submodules are missing in some environments
    pass

# Export what we imported
for name, val in list(locals().items()):
    if name.startswith('_') and name not in ('_core', '_usage', '_headers', '_errors', '_fingerprint', '_capabilities'):
        continue

# Attach submodules to this package namespace
for sub in ['usage', 'headers', 'errors', 'fingerprint', 'capabilities']:
    try:
        mod = importlib.import_module(f'core.{sub}')
        globals()[sub] = mod
        # Also register the module under the diamond.core.* name so imports succeed
        sys.modules[f'diamond.core.{sub}'] = mod
    except Exception:
        pass
