class ContextManager:

    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens

    async def compact(self, messages, model_call):

        token_count = sum(len(m["content"]) for m in messages)

        if token_count < self.max_tokens:
            return messages

        summary_prompt = [
            {"role": "system", "content": "Summarize the conversation preserving facts."},
            {"role": "user", "content": str(messages)}
        ]

        summary = await model_call(summary_prompt)

        return [
            {"role": "system", "content": "Conversation summary:"},
            {"role": "assistant", "content": summary}
        ]
