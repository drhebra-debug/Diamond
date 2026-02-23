import hashlib
import platform

def generate_fingerprint(model_name):
    base = f"{model_name}-{platform.node()}-{platform.processor()}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]
