# AI Customer Support Bot (RAG-Powered)

## Project Overview

This pipeline demonstrates a complete pipeline featuring:

- Large Language Model fine-tuning
- Retrieval Augmented Generation (RAG) with semantic search
- LangServe (FastAPI) for service delivery
- A well-structured, deployable pipeline with clear directory structure

## 🟣Features:

- Fine-tune a large language model with custom training data.
- Develop a semantic search pipeline employing LangChain components.
- Implement service endpoints with LangServe/FastAPI.
- Provide helpful, factual answers in simple language.
- Easily deploy with Docker Compose.

## 🟣Technology Stack:

- LangChain
- LangServe
- Large Language Model (OpenAI GPT-4)
- FAISS (for semantic search)
- Docker
- FastAPI
- Python
- Pydantic

## 🟣Workflow:

1. Fine-tune the base large language model with custom data.
2. Develop LangChain pipeline employing semantic search (Retrieval Augmented Generation).
3. Implement service endpoints with LangServe/FastAPI.
4. Provide a complete, documented, deployable pipeline.

> Currently, **this pipeline does not implement a tool-using or ReAct agent** — 
> instead, we perform semantic search against a knowledge base and augment the 
> large language model’s answers with retrieved documents.

## 🟣File structure:

your-project/
└ src/
└ pipeline/
└ agent_tools/
└ knowledge_base_tool.py
└ fine_tune/
└ fine_tune.py
└ server/
└ serve.py
└ main_pipeline.py
└ data/
└ .gitignore
└ README.md
└ requirements.txt
└ docker-compose.yaml


## 🟣 Installation:

```bash
git clone <repo_url>
cd your-project
pip install -r requirements.txt

🟣 Fine-tuning:
python src/pipeline/fine_tune/fine_tune.py

🟣 Run with Docker Compose:
docker-compose up -d

🟣 Run without Docker:
uvicorn src.pipeline.server.serve:app --reload

🟣 API endpoint:
http://localhost:8000/query

with a body like:

json
{
  "input": "What is your return policy?"
}

and you'll get:
json
{
  "response": "We have a 30-day return policy with a full refund."
}
