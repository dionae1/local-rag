from langchain_core.prompts import PromptTemplate

chat_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
                You are a helpful assistant that helps answer questions based on provided documents. Use the information from the documents to formulate your answers.
                If the documents do not contain relevant information, respond politely that you do not have the necessary information.
                Remove weird line breaks or unusual characters from the context before answering.
                Documents:
                    {context}

                Question:
                    {question}

            """,
)