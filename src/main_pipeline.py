import os
from dotenv import load_dotenv

load_dotenv()  # This parses .env and loads into os.environ


from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from src.pipeline.agent_tools.knowledge_base_tool import create_retriever


def get_llm():
    """Create GPT-4 powered chat model with custom temperature and context."""
    return ChatOpenAI(
        temperature=0,
        model_name='gpt-4',
        # API key is already loaded from .env
    )


def create_pipeline(sources=None):
    """Create RetrievalQA pipeline with GPT-4, semantic search, custom prompting, and chat history."""
    if sources is None:
        sources = ["data/documents/knowledge_base.txt"]  # fallback
    
    retriever = create_retriever(sources=sources)
    llm = get_llm()

    # Define a custom system message to guide the LLM’s behavior
    system_prompt = """You are a helpful and accurate Customer Support Bot.
    Always base your answers on the knowledge base documents provided.
    Provide clear, helpful, and factual answers in simple language.
    If you do not know something, say you do not know instead of guessing.
    Keep track of the chat history and make conversations coherent and helpful.
    """

    # Human, System, and AI messages can reflect previous exchanges
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{input}"),
        AIMessagePromptTemplate.from_template("{ai_response}"),
    ])

    # LangChain expression (LCEL) pipeline:
    pipeline = (
        retriever
        | (lambda docs: {"docs": docs})  # forward retrieved docs
        | (lambda inputs: {
            "input": inputs["input"],
            "ai_response": inputs.get("ai_response", ""),
            "docs": inputs["docs"]
        })
        | chat_prompt
        | llm
    )

    return pipeline


# If you want to run directly:
if __name__ == "__main__":
    pipeline = create_pipeline()

    # Initialize chat history
    chat_history = []

    while True:
        query = input("User: ").strip()
        if query.lower() == "exit":
            break

        response = pipeline.invoke({"input": query, "ai_response": chat_history[-1][1] if chat_history else ''})

        print("Bot :", response)
        chat_history.append((query, response))
