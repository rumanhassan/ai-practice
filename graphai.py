from langgraph.graph import StateGraph, END
from openai import OpenAI
from models import State
from services import getChatResponse

client = OpenAI()

def summarize(state: State) -> State:
    text = state["text"]

    state["summary"] = getChatResponse(f"Summarize this: {text}")
    return state


def sentiment(state: State) -> State:
    text = state["text"]
    state["sentiment"] = getChatResponse(f"Classify sentiment (positive, neutral, negative): {text}")
    return state

graph = StateGraph(State)
graph.add_node("summarize", summarize)
graph.add_node("sentiment", sentiment)

graph.set_entry_point("summarize")
graph.add_edge("summarize", "sentiment")
graph.add_edge("sentiment", END)

workflow = graph.compile()

