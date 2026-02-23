import subprocess
import os

class GitManager:
    def __init__(self, repo_path):
        self.repo = repo_path

    def run(self, cmd):
        return subprocess.run(
            cmd,
            cwd=self.repo,
            shell=True,
            capture_output=True,
            text=True
        )

    def create_branch(self, name):
        return self.run(f"git checkout -b {name}")

    def commit_all(self, message):
        self.run("git add .")
        return self.run(f'git commit -m "{message}"')

    def diff(self):
        return self.run("git diff").stdout

    def reset_hard(self):
        return self.run("git reset --hard")

    def current_branch(self):
        return self.run("git branch --show-current").stdout.strip()
