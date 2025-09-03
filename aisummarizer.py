from openai import OpenAI
from fastapi import FastAPI
from typing import Optional, List
from models import AnalysisResponse, RootResponse, ChatResponse, ChatRequest, PromptTemplate, TextRequest
from services import getChatResponse
from graphai import workflow
import chromadb
import tiktoken


client = OpenAI()
app = FastAPI()

chroma_client = chromadb.Client()

# Create / get a collection
collection = chroma_client.get_or_create_collection(name="docs")

# ---- Tokenizer setup ----
# Use GPT-3.5 tokenizer (works for embeddings + GPT-4 as well)
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str, max_tokens: int = 100, overlap: int = 20) -> List[str]:
    """Split text into chunks based on token count with overlap."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap  # move forward with overlap
    return chunks


# Function to load docs (for demo, just once at startup)
def load_docs():
    # Example docs in memory (replace with reading .txt files from folder)
    documents = [
        "FastAPI is a modern Python web framework for building APIs quickly.",
        "Chroma is a vector database useful for semantic search and RAG pipelines.",
        "OpenAI provides models for text generation, summarization, and embeddings."
    ]
    # Generate embeddings + add to collection
    for i, doc in enumerate(documents):
        chunks = chunk_text(doc, max_tokens=50, overlap=10)  # smaller chunks
        for j, chunk in enumerate(chunks):
            emb = client.embeddings.create(
                model="text-embedding-3-small", input=chunk
            )
            collection.add(
                ids=[f"doc_{i}_chunk_{j}"],
                documents=[chunk],
                embeddings=[emb.data[0].embedding]
            )

# Load docs once
load_docs()

@app.get("/", response_model=RootResponse)
def read_root():
    return { "message": "world"}

@app.post("/summarize", response_model=ChatResponse)
def chatOpenAi(request: ChatRequest):
    template = PromptTemplate("Can you summarize this text into 3 bullet points: {input_text}")
    prompt = template.format(input_text=request.user_message)
    resp: Optional[str] = getChatResponse(prompt)
    if resp is None:
        # Return a friendly message if OpenAI call fails
        return {"response": "Sorry, could not get a response from the AI."}
    return {"response": resp}


@app.post("/rag_summarize", response_model=ChatResponse)
def rag_summarize(request: ChatRequest):
    # Step 1: Embed user query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small", input=request.user_message
    ).data[0].embedding

    # Step 2: Retrieve relevant docs
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    context_chunks: List[str] = results.get("documents", [[]])[0]
    context_text = "\n".join(context_chunks)

    # Step 3: Build RAG prompt
    template = PromptTemplate(
    "Using the following context:\n\n{context}\n\nSummarize the answer to: {input_text}\n\nSummary:"
    )
    prompt = template.format(
        input_text=request.user_message, context=context_text
    )

    # Step 4: Get summary
    resp = getChatResponse(prompt)
    if resp is None:
        return {"response": "Sorry, could not get a response from the AI."}
    return {"response": resp}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    # Run workflow
    result = workflow.invoke({"text": request.text})
    print(result)
    return AnalysisResponse(
        summary=result["summary"],
        sentiment=result["sentiment"]
    )


