from openai import OpenAI
from typing import Optional, Generator

client = OpenAI()

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

