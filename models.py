from pydantic import BaseModel
from typing import Dict

class RootResponse(BaseModel):
    message: str

class MessageItem(BaseModel):
    role: str
    content: str

# class ChatResponse(BaseModel):
#     assistant_message: str
#     history: List[MessageItem]

class ChatResponse(BaseModel):
    response: str

class ChatRequest(BaseModel):
    user_message: str

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kargs):
        return self.template.format(**kargs)

# Define state type
class State(Dict):
    text: str
    summary: str
    sentiment: str

# Request model
class TextRequest(BaseModel):
    text: str

# Response model
class AnalysisResponse(BaseModel):
    summary: str
    sentiment: str

