"""
╔══════════════════════════════════════════════════════════════╗
║      FULL CUSTOMER SUPPORT AGENT — Eligent AI                ║
║      Day 4 Capstone · WITH Escalation Loop                   ║
║  Flow: classify → rag_resolve → check_node →                 ║
║        END (resolved) | retry loop | escalate_node → END     ║
╚══════════════════════════════════════════════════════════════╝
"""
import os, json, uuid
from typing import TypedDict, Optional, Annotated, Literal
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

# ════ STATE ════
class SupportAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    category: Optional[str]
    sentiment: Optional[str]
    attempt_count: int
    resolved: bool           # ← NEW: rag_resolve_node set karega
    escalated: bool
    ticket_id: Optional[str] # ← NEW: escalate_node set karega
    escalation_reason: Optional[str]
    human_decision: Optional[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

KNOWLEDGE_BASE = {
    "technical": "API 401: Regenerate key in Settings→API Keys. Rate limit: exponential backoff. Timeout: increase to 30s.",
    "billing":   "Refund: full within 30 days, store credit after. Duplicate charge: resolved in 24hrs.",
    "general":   "Export: Dashboard→Settings→Data→Export. Docs: docs.techflow.com. Status: status.techflow.com"
}

# ════ NODE 1: CLASSIFY ════
def classify_node(state: SupportAgentState) -> dict:
    last_human = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    print(f"\n{'='*55}\n🔍 CLASSIFY: '{last_human[:50]}...'\n{'='*55}")
    system = 'Classify support message. Return ONLY JSON: {"category":"technical"|"billing"|"general","sentiment":"positive"|"neutral"|"negative"|"frustrated"}'
    try:
        r = json.loads(llm.invoke([SystemMessage(content=system), HumanMessage(content=last_human)]).content)
        cat, sent = r.get("category","general"), r.get("sentiment","neutral")
    except:
        cat, sent = "general", "neutral"
    print(f"  Category: {cat} | Sentiment: {sent}")
    return {"category": cat, "sentiment": sent}

# ════ NODE 2: RAG RESOLVE (FIXED) ════
def rag_resolve_node(state: SupportAgentState) -> dict:
    attempt = state.get("attempt_count", 0) + 1
    category = state.get("category", "general")
    print(f"\n{'='*55}\n🔧 RAG RESOLVE (Attempt #{attempt})\n{'='*55}")
    knowledge = KNOWLEDGE_BASE.get(category, KNOWLEDGE_BASE["general"])
    system = f"""Customer support agent for TechFlow.
KNOWLEDGE: {knowledge}
Attempt #{attempt} of 3. Be specific, professional (3-5 sentences)."""
    response = llm.invoke([SystemMessage(content=system)] + list(state["messages"]))
    resolution_phrases = ["this should resolve","hope this helps","please try","let me know if"]
    resolved_flag = any(p in response.content.lower() for p in resolution_phrases)
    print(f"  Resolved flag: {resolved_flag}")
    return {"messages": [response], "attempt_count": attempt, "resolved": resolved_flag}

# ════ NODE 3: CHECK NODE (NEW) ════
def check_node(state: SupportAgentState) -> dict:
    print(f"\n[check_node] resolved={state.get('resolved')} | attempts={state.get('attempt_count')}")
    return {}

def route_after_check(state: SupportAgentState) -> Literal["end", "rag_resolve", "escalate"]:
    if state.get("resolved", False):
        print("  [router] ✅ Resolved → END"); return "end"
    elif state.get("attempt_count", 0) < 3:
        print(f"  [router] 🔁 Retry → rag_resolve"); return "rag_resolve"
    else:
        print("  [router] 🚨 Exhausted → escalate"); return "escalate"

# ════ NODE 4: ESCALATE (NEW) ════
def escalate_node(state: SupportAgentState) -> dict:
    ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"
    last_human = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    reason = f"Sentiment:{state.get('sentiment')} | Attempts:{state.get('attempt_count',0)}"
    print(f"\n  🚨 ESCALATION | {ticket_id} | {reason}")
    # Future: await save_to_neon(ticket_id, state)
    # Future: await send_email_alert(ticket_id)
    msg = AIMessage(content=f"Support ticket {ticket_id} created. Team contacts you in 2 hrs. Track: support.techflow.com/tickets/{ticket_id}")
    return {"messages": [msg], "escalated": True, "ticket_id": ticket_id, "escalation_reason": reason}

# ════ GRAPH ════
def build_agent():
    builder = StateGraph(SupportAgentState)
    builder.add_node("classify", classify_node)
    builder.add_node("rag_resolve", rag_resolve_node)
    builder.add_node("check_node", check_node)
    builder.add_node("escalate", escalate_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "rag_resolve")
    builder.add_edge("rag_resolve", "check_node")         # ← NEW connection
    builder.add_conditional_edges("check_node", route_after_check,
        {"end": END, "rag_resolve": "rag_resolve", "escalate": "escalate"})
    builder.add_edge("escalate", END)

    return builder.compile(checkpointer=MemorySaver())

# ════ RUN ════
if __name__ == "__main__":
    graph = build_agent()
    config = {"configurable": {"thread_id": "support_001"}}

    initial: SupportAgentState = {
        "messages": [], "category": None, "sentiment": None,
        "attempt_count": 0, "resolved": False, "escalated": False,
        "ticket_id": None, "escalation_reason": None, "human_decision": None,
    }

    print("\n" + "█"*55)
    print("█     CUSTOMER SUPPORT AGENT — Eligent AI           █")
    print("█"*55 + "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit","exit"]: break
        # v1.1.6: version="v2" → GraphOutput(.value, .interrupts)
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config, version="v2")
        last_ai = next((m.content for m in reversed(result.value["messages"]) if isinstance(m, AIMessage)), "")
        print(f"\nAgent: {last_ai}")
        print(f"[Cat:{result.value.get('category','?')} | Attempts:{result.value.get('attempt_count',0)} | Escalated:{result.value.get('escalated',False)} | Ticket:{result.value.get('ticket_id','None')}]\n")