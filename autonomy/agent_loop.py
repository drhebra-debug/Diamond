import uuid
from .git_manager import GitManager
from .file_manager import FileManager
from .test_runner import TestRunner

class AutonomousCoder:

    def __init__(self, llm, repo_path):
        self.llm = llm
        self.repo = repo_path
        self.git = GitManager(repo_path)
        self.files = FileManager(repo_path)
        self.tests = TestRunner(repo_path)

    def execute_task(self, task_prompt):

        branch = f"diamond-{uuid.uuid4().hex[:6]}"
        self.git.create_branch(branch)

        messages = [
            {"role": "system", "content": "You are an autonomous coding agent."},
            {"role": "user", "content": task_prompt}
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.3
        )

        plan = response["choices"][0]["message"]["content"]

        # Run tests
        test_result = self.tests.run_pytest()

        if test_result.returncode != 0:
            fix_prompt = f"Tests failed:\n{test_result.stdout}\nFix them."
            messages.append({"role": "assistant", "content": plan})
            messages.append({"role": "user", "content": fix_prompt})

            fix_response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.2
            )

            plan = fix_response["choices"][0]["message"]["content"]

        self.git.commit_all("Autonomous fix by Diamond")

        return {
            "branch": branch,
            "plan": plan,
            "tests": test_result.stdout
        }
