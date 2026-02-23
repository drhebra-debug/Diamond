import subprocess

def run_sandboxed(command):
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        timeout=5,
        cwd="/tmp",
        text=True
    ).stdout
