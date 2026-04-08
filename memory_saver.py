from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
SYSTEM_PROMPT = "You are a helpful customer support agent for TechFlow SaaS."

def agent_node(state: MessagesState) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    response = llm.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

memory = MemorySaver()  # In-memory checkpointer
graph = builder.compile(checkpointer=memory)  # ← KEY: checkpointer pass karo

def chat(user_input: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}  # ← thread_id mandatory hai
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    return result["messages"][-1].content

# Thread 1: User A
print("\n=== USER A (thread: user_001) ===")
ques1='Hi, my name is Rahul. I have login issues.'
print(f"\nUser: {ques1}\n")
print(f"Agent: {chat(ques1, 'user_001')}\n")

ques2='The reset email is not coming.'
print(f"User: {ques2}\n")
print(f"Agent: {chat(ques2, 'user_001')}\n")

ques3='What is my name?'
print(f"User: {ques3}\n")
print(f"Agent: {chat(ques3, 'user_001')}\n")  # "Your name is Rahul" — memory works!


snap = graph.get_state({"configurable": {"thread_id": "user_001"}})
print(f"\nThread user_001 message count: {len(snap.values['messages'])}\n")

# Thread 2: User B — completely isolated
print("\n=== USER B (thread: user_002) ===")

qus1='Hello, I need help with billing.'
print(f"\nUser: {qus1}\n")
print(f"Agent: {chat(qus1, 'user_002')}\n")

qus2='What was the previous user asking?'
print(f"User: {qus2}\n")
print(f"Agent: {chat(qus2, 'user_002')}\n")  # No info about thread 1

# State inspect karo
snap = graph.get_state({"configurable": {"thread_id": "user_002"}})
print(f"\nThread user_002 message count: {len(snap.values['messages'])}\n")