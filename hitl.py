from dotenv import load_dotenv
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command  # ← same imports, unchanged

load_dotenv()

class ReviewState(TypedDict):
    content: str
    human_decision: Optional[str]
    final_action: str

def generate_response_node(state: ReviewState) -> dict:
    generated = f"Generated response for: '{state['content']}'"
    print(f"\n[Node] Response generated: {generated}")
    return {"content": generated}

def human_review_node(state: ReviewState) -> dict:
    """interrupt() yahan call hota hai — graph yahan PAUSE hoga"""
    print(f"\n[Node] Human review required!")
    print(f"  Content: {state['content']}")
    # interrupt() graph rok deta hai — Jo value pass karo woh human ko dikhti hai
    # Jab Command(resume=value) aata hai toh interrupt() woh value return karta hai
    human_input = interrupt({
        "message": "Please review and decide",
        "content": state["content"],
        "options": ["approve", "reject", "escalate"]
    })
    print(f"  Human decided: {human_input}")
    return {"human_decision": human_input}

def process_decision_node(state: ReviewState) -> dict:
    decision = state.get("human_decision", "approve")
    if decision == "approve":   return {"final_action": "Response sent ✅"}
    elif decision == "reject":  return {"final_action": "Discarded, regenerating..."}
    else:                       return {"final_action": "Escalated to senior agent 🚨"}

builder = StateGraph(ReviewState)
builder.add_node("generate", generate_response_node)
builder.add_node("human_review", human_review_node)
builder.add_node("process", process_decision_node)
builder.add_edge(START, "generate")
builder.add_edge("generate", "human_review")
builder.add_edge("human_review", "process")
builder.add_edge("process", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)  # ← Checkpointer MANDATORY for HITL
config = {"configurable": {"thread_id": "hitl_demo"}}

# STEP 1: Graph chalao — human_review pe pause hoga
print("=== STEP 1: Starting ===")
# ← v1.1.6: invoke(version="v2") se GraphOutput milta hai
result = graph.invoke(
    {"content": "User wants refund #1234"},
    config=config,
    version="v2"    # ← NEW: GraphOutput return hoga
)

# ← v1.1.6: result.interrupts se check karo — graph.get_state().next ki zaroorat nahi!
if result.interrupts:
    print(f"Graph paused! Interrupt value: {result.interrupts[0].value}")
    # result.interrupts[0].value mein woh dict hai jo interrupt() ko pass kiya tha
else:
    print(f"Graph completed: {result.value['final_action']}")

# STEP 2: Human decide karta hai aur resume karta hai
print("\n=== STEP 2: Human reviewing ===")
final = graph.invoke(
    Command(resume="escalate"),  # ← resume karo — same as before
    config=config,
    version="v2"
)
# final.interrupts empty hoga — graph complete hua
print(f"Final Action: {final.value['final_action']}")
# Backward compatible: final["final_action"] bhi kaam karta hai