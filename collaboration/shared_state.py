class SharedState:
    def __init__(self):
        self.memory = {}

    def update(self, key, value):
        self.memory[key] = value

    def get(self, key):
        return self.memory.get(key)
