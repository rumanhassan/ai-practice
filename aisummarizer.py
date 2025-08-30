from openai import OpenAI
from fastapi import FastAPI
from typing import Optional, Generator
from models import RootResponse, ChatResponse, ChatRequest, PromptTemplate

client = OpenAI()
app = FastAPI()

template = PromptTemplate("Can you summarize this text into 3 bullet points: {input_text}")

@app.get("/", response_model=RootResponse)
def read_root():
    return { "message": "world"}

@app.post("/chat", response_model=ChatResponse)
def chatOpenAi(request: ChatRequest):
    prompt = template.format(input_text=request.user_message)
    resp: Optional[str] = getChatResponse(prompt)
    if resp is None:
        # Return a friendly message if OpenAI call fails
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




