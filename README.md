# agentic-ai-pipeline

## Brief description:

- main.py: Orchestrates pipeline
- fine_tune.py: Fine-tunes the base LLM
- chain_manager.py: Manages LangChain expression chains
- server.py: Serves the pipeline through LangServe
- test_pipeline.py: Tests the pipeline components

## Installation:

```bash
pip install -r requirements.txt
```

## Run:

```bash
uvicorn langserve.server:app
```

