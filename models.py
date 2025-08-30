from pydantic import BaseModel
from typing import List, Optional

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
