from openai import OpenAI
from fastapi import FastAPI
from typing import Optional, Generator
from models import RootResponse, ChatResponse, ChatRequest, PromptTemplate
import chromadb

client = OpenAI()
app = FastAPI()

chroma_client = chromadb.Client()

# Create / get a collection
collection = chroma_client.get_or_create_collection(name="docs")


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
        emb = client.embeddings.create(
            model="text-embedding-3-small", input=doc
        )
        collection.add(
            ids=[f"doc_{i}"],
            documents=[doc],
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


def getChatResponse(prompt: str) -> Optional[str]:
    try:

        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )

        if not getattr(response, "output", None):
            print("No output returned by the model.")
            return None
        message = next(
            extract_message_texts(response),
            None
        )
        return message
    except Exception as e:
        print("An error occurred: {e}")
        return None

def extract_message_texts(response) -> Generator[str, None, None]:
    output = response.dict().get("output", [])
    for msg in output:
        if msg.get("type") == "message" and msg.get("content"):
            yield msg["content"][0].get("text") 




