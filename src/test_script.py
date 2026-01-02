from llm.transformers import TransformersLLM
from services import LLMService

llm = TransformersLLM("Qwen/Qwen2.5-7B-Instruct")
rag = LLMService(llm)

print(rag.answer("Explain the content of the document in one sentence."))
