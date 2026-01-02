from llm import base

class DummyLLM(base.LLM):
    def __init__(self, model_name: str = "dummy-model"):
        super().__init__(model_name)

    def generate_text(self, prompt: str = "") -> str:
        return "This is a dummy response."