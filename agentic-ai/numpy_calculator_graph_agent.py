from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import numpy as np


# 1. Define the Calculator Tool
@tool
def numpy_calculator(expression: str) -> str:
    """Evaluate a math expression safely using NumPy functions."""
    try:
        # Restrict evaluation scope to numpy
        result = eval(expression, {"__builtins__": {}}, {"np": np, **np.__dict__})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# 2. Define Agent State (memory passed between nodes)
class AgentState(TypedDict):
    input: str
    steps: List[str]   # keeps tool calls + outputs
    output: str


# 3. LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 4. Define Graph Nodes
def agent_node(state: AgentState):
    """LLM decides what to do next (use tool or answer)."""
    question = state["input"]
    steps = "\n".join(state["steps"])

    system_prompt = (
      "You are a strict math assistant.\n"
        "RULES:\n"
        "1. You MUST NOT compute expressions yourself.\n"
        "2. Check if expression needs to evaluated further if yes then proceed as stated in step 3 else proceed as stated in step 4"
        "3. To do any math, you MUST call the tool in this exact format:\n"
        "   numpy_calculator(<expression>)\n"
        "   Example: numpy_calculator(np.sin(np.pi/4)**2 + np.cos(np.pi/4)**2)\n"
        "4. After the tool gives a result, you may output the final answer in this format:\n"
        "   Final Answer: <value>\n"
        "5. Do NOT mix reasoning text. Only output tool calls or 'Final Answer: ...'.\n\n"
        f"Previous steps:\n{steps}"
    )

    print(f"question---{question}")

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ])

    return {"steps": state["steps"] + [f"LLM: {response.content}"]}


def tool_node(state: AgentState):
    """Executes calculator tool if the LLM requested it."""
    last_step = state["steps"][-1]
    print("inside--tool")

    if "numpy_calculator" in last_step:
        # Extract expression (naive parse for demo)
        expr = last_step.split("numpy_calculator(")[-1].strip(" )")
        result = numpy_calculator(expr)
        return {"steps": state["steps"] + [f"Tool result: {result}"], "input": f"Tool result: {result}"}
    else:
        return {"output": last_step, "steps": state["steps"]}


# 5. Conditional Edge (decide when to stop)
def should_continue(state: AgentState):
    last_step = state["steps"][-1]
    if "numpy_calculator" in last_step:
        print(f"last step-- {last_step}")
        return "tool"
    else:
        return END


# 6. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tool": "tool", END: END})
workflow.add_edge("tool", "agent")  # loop back

graph = workflow.compile()


# 7. Run Example
if __name__ == "__main__":
    result = graph.invoke({"input": "What is sin(pi/4)**2 + cos(pi/4)**2?", "steps": [], "output": ""})
    print("Trace of steps:")
    for step in result["steps"]:
        print("  ", step)
    print("Final Answer:", result["output"])
