import time

class UsageTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.start_time = time.time()

    def add_prompt(self, count):
        self.prompt_tokens += count

    def add_completion(self, count):
        self.completion_tokens += count

    def to_dict(self):
        return {
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens
        }
