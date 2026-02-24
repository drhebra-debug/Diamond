import importlib, sys, os

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

_tools = importlib.import_module('tools')
# Register the top-level tools package as diamond.tools
sys.modules['diamond.tools'] = _tools

# Register common submodules under diamond.tools.*
for sub in ['executor', 'registry', 'router', 'sandbox', 'schemas']:
    try:
        mod = importlib.import_module(f'tools.{sub}')
        sys.modules[f'diamond.tools.{sub}'] = mod
    except Exception:
        pass
