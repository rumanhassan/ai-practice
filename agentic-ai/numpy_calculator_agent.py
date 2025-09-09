from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import (
    create_openai_functions_agent,
    AgentExecutor,
)
from langchain.tools import tool
import numpy as np

# 1. Define the Calculator Tool
@tool
def numpy_calculator(expression: str) -> str:
    """Evaluate a math expression safely using NumPy functions."""
    try:
        # Limit scope to numpy functions only
        result = eval(expression, {"__builtins__": {}}, {"np": np, **np.__dict__})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# 2. LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Prompt Template
SYSTEM_PROMPT = (
    "You are a helpful AI that solves math problems using a calculator tool. "
    "When needed, call the `numpy_calculator` tool with valid NumPy expressions."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # Needed for intermediate steps
])

# 4. Create Agent
agent = create_openai_functions_agent(llm, [numpy_calculator], prompt)

# 5. Agent Executor
executor = AgentExecutor(
    agent=agent,
    tools=[numpy_calculator],
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

# 6. Run Example
if __name__ == "__main__":
    query = "What is sin(pi/4)**2 + cos(pi/4)**2?"
    result = executor.invoke({"input": query, "chat_history": []})
    print("Final Answer:", result["output"])
