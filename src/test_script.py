from services import UploadService, LLMService, DBService, build_llm
from db.factory_db import get_vector_db
    
db = get_vector_db()

us = UploadService("samples/sample.pdf", db=db)
us.insert_documents()

print("Documents inserted into the vector database.")

llms = LLMService(build_llm("gemini"), db=db)
query = "What is the main topic of the document?"
results = llms.answer(query)
print("Query Results:", results)