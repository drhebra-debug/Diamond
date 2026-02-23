import subprocess

class TestRunner:
    def __init__(self, repo_path):
        self.repo = repo_path

    def run_pytest(self):
        return subprocess.run(
            "pytest -q",
            cwd=self.repo,
            shell=True,
            capture_output=True,
            text=True
        )

    def run_npm_test(self):
        return subprocess.run(
            "npm test",
            cwd=self.repo,
            shell=True,
            capture_output=True,
            text=True
        )
