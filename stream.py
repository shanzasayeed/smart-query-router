import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def agent_node(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

memory = MemorySaver()
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile(checkpointer=memory)

# ── PATTERN 1: stream() — Node-level updates ──
print("=== PATTERN 1: stream() ===")
config1 = {"configurable": {"thread_id": "s1"}}
for chunk in graph.stream({"messages": [HumanMessage(content="Explain LangGraph in 2 sentences")]}, config=config1):
    for node_name, state_update in chunk.items():
        print(f"[Node '{node_name}' done]")
        if "messages" in state_update:
            print(f"Response: {state_update['messages'][-1].content}")

# ── PATTERN 2: stream_mode="values" — Full state at each step ──
print("\n=== PATTERN 2: stream_mode='values' ===")
config2 = {"configurable": {"thread_id": "s2"}}
for state in graph.stream({"messages": [HumanMessage(content="What is a state machine?")]}, config=config2, stream_mode="values"):
    last = state["messages"][-1]
    print(f"[{last.__class__.__name__}] {last.content[:60]}...")

# ── PATTERN 3: astream_events() — Token-by-token ──
async def token_streaming():
    print("\n=== PATTERN 3: Token-by-Token ===")
    print("Agent: ", end="", flush=True)
    config3 = {"configurable": {"thread_id": "s3"}}
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="Write a short poem about AI agents")]},
        config=config3, version="v2"
    ):
        if event.get("event") == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                print(chunk, end="", flush=True)
    print()

asyncio.run(token_streaming())