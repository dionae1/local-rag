from llm import base
from langchain_google_genai import GoogleGenerativeAI


class GeminiLLM(base.LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = GoogleGenerativeAI(model=self.model_name)

    def generate_text(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response