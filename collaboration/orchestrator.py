from .roles import ROLES
from .shared_state import SharedState

class MultiAgentOrchestrator:

    def __init__(self, llm):
        self.llm = llm
        self.state = SharedState()

    def run(self, task):

        for role, instruction in ROLES.items():

            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": task}
            ]

            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.4
            )

            output = response["choices"][0]["message"]["content"]
            self.state.update(role, output)

        return self.state.memory
