# src/pipeline/server/serve.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline.main_pipeline import create_pipeline

# ------------------------------------------------------------------------------
# LangServe/FastAPI application
# ------------------------------------------------------------------------------

app = FastAPI(
    title='AI Customer Support Bot API',
    description='Provides answers to customer inquiries based on knowledge base documents.',
    version='1.0.0'
)


# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    input: str


class QueryResponse(BaseModel):
    response: str


# ------------------------------------------------------------------------------
# Initialize pipeline
# ------------------------------------------------------------------------------

print("Initializing pipeline...")
pipeline = create_pipeline()
print("Pipeline initialized successfully.")


# ------------------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------------------

@app.post('/query', response_model=QueryResponse)
def query_endpoint(item: QueryRequest):
    """
    Query the pipeline with a user's message and get a helpful, factual response.
    """
    try:
        response = pipeline.invoke({"input": item.input, "ai_response": ''})
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# Run with:
# uvicorn src.pipeline.server.serve:app --reload
# ------------------------------------------------------------------------------
