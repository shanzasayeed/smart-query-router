
"""
╔══════════════════════════════════════════════════════════════════╗
║      CUSTOMER SUPPORT AGENT v2.0 — Eligent AI                   ║
║      Day 4 Capstone · FINAL PRODUCTION-READY VERSION            ║
║                                                                  ║
║  Fixes Applied:                                                  ║
║  ✅ Structured LLM grading (no brittle phrase matching)          ║
║  ✅ Retry with varied prompt per attempt                         ║
║  ✅ Safe attempt_count reset per conversation turn               ║
║  ✅ try/except in rag_resolve so loop never silently breaks      ║
║  ✅ human_decision / interrupt wired (HITL before escalation)   ║
║  ✅ Unused imports removed / properly used                       ║
║  ✅ LangGraph 1.1.6 compatible (version="v2", result.value)     ║
║                                                                  ║
║  Flow:                                                           ║
║    classify → rag_resolve → grade → check →                     ║
║    END (resolved) | retry loop | hitl_node → escalate → END     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import uuid
from typing import TypedDict, Optional, Annotated, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages

load_dotenv()

# ════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════
class SupportAgentState(TypedDict):
    messages:           Annotated[list[BaseMessage], add_messages]
    category:           Optional[str]
    sentiment:          Optional[str]
    attempt_count:      int
    resolved:           bool
    escalated:          bool
    ticket_id:          Optional[str]
    escalation_reason:  Optional[str]
    human_decision:     Optional[str]   # "escalate" | "retry" — set by HITL node


# ════════════════════════════════════════════════════
# LLM
# ════════════════════════════════════════════════════
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ════════════════════════════════════════════════════
KNOWLEDGE_BASE = {
    "technical": (
        "API 401 error: regenerate your API key under Settings → API Keys. "
        "Rate limit (429): use exponential backoff starting at 1 s. "
        "Timeout errors: increase client timeout to 30 s. "
        "Webhook failures: verify endpoint URL and check HMAC signature validation."
    ),
    "billing": (
        "Refund policy: full refund within 30 days; store credit after 30 days. "
        "Duplicate charges: auto-resolved within 24 hrs, email confirmation sent. "
        "Invoice copies: Dashboard → Billing → Invoices → Download PDF. "
        "Plan upgrades/downgrades: prorated immediately on change."
    ),
    "general": (
        "Data export: Dashboard → Settings → Data → Export (CSV / JSON). "
        "Documentation: docs.techflow.com. "
        "System status: status.techflow.com. "
        "Account deletion: Settings → Account → Delete (30-day grace period)."
    ),
}

RETRY_HINTS = {
    2: "Your first response didn't resolve the issue. Try a completely different explanation or step-by-step approach.",
    3: "Two attempts have failed. This is the final automated attempt. Be extremely specific, include exact steps and error codes.",
}


# ════════════════════════════════════════════════════
# NODE 1 — CLASSIFY
# ════════════════════════════════════════════════════
def classify_node(state: SupportAgentState) -> dict:
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )
    print(f"\n{'='*60}")
    print(f"🔍 CLASSIFY: '{last_human[:60]}...'")
    print(f"{'='*60}")

    system = (
        'Classify this customer support message.\n'
        'Return ONLY valid JSON — no markdown, no explanation:\n'
        '{"category":"technical"|"billing"|"general",'
        '"sentiment":"positive"|"neutral"|"negative"|"frustrated"}'
    )
    try:
        raw = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=last_human)]
        ).content.strip()
        # Strip accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        category  = result.get("category", "general")
        sentiment = result.get("sentiment", "neutral")
    except Exception as e:
        print(f"  ⚠️  Classify parse error ({e}) — defaulting to general/neutral")
        category, sentiment = "general", "neutral"

    print(f"  Category : {category}")
    print(f"  Sentiment: {sentiment}")
    # Reset per-turn counters so each new user message starts fresh
    return {
        "category":      category,
        "sentiment":     sentiment,
        "attempt_count": 0,
        "resolved":      False,
        "escalated":     False,
        "ticket_id":     None,
        "escalation_reason": None,
        "human_decision": None,
    }


# ════════════════════════════════════════════════════
# NODE 2 — RAG RESOLVE
# ════════════════════════════════════════════════════
def rag_resolve_node(state: SupportAgentState) -> dict:
    attempt  = state.get("attempt_count", 0) + 1
    category = state.get("category", "general")
    knowledge = KNOWLEDGE_BASE.get(category, KNOWLEDGE_BASE["general"])
    retry_hint = RETRY_HINTS.get(attempt, "")

    print(f"\n{'='*60}")
    print(f"🔧 RAG RESOLVE — Attempt #{attempt}/3")
    print(f"{'='*60}")

    system = (
        f"You are a professional customer support agent for TechFlow.\n\n"
        f"KNOWLEDGE BASE:\n{knowledge}\n\n"
        f"ATTEMPT: #{attempt} of 3.\n"
        f"{retry_hint}\n\n"
        f"Rules:\n"
        f"- Be specific and actionable (3–5 sentences).\n"
        f"- If you can resolve the issue with the knowledge base, do so.\n"
        f"- Do NOT promise things outside the knowledge base.\n"
        f"- End your reply with exactly one of these tags on a new line:\n"
        f"  [RESOLVED] — if your response fully addresses the issue\n"
        f"  [UNRESOLVED] — if the issue needs further help"
    )

    try:
        response = llm.invoke([SystemMessage(content=system)] + list(state["messages"]))
        content  = response.content

        # ── Structured resolve grading via explicit tag (no brittle phrases) ──
        if "[RESOLVED]" in content:
            resolved_flag = True
            # Clean tag from visible message
            clean_content = content.replace("[RESOLVED]", "").strip()
        elif "[UNRESOLVED]" in content:
            resolved_flag = False
            clean_content = content.replace("[UNRESOLVED]", "").strip()
        else:
            # Fallback: ask LLM to grade its own response
            resolved_flag = _grade_resolution(content)
            clean_content = content

        final_msg = AIMessage(content=clean_content)
        print(f"  Resolved: {resolved_flag}")
        return {
            "messages":     [final_msg],
            "attempt_count": attempt,
            "resolved":      resolved_flag,
        }

    except Exception as e:
        print(f"  ❌ LLM error: {e}")
        error_msg = AIMessage(content="I'm having trouble accessing our systems. Please try again in a moment.")
        return {
            "messages":      [error_msg],
            "attempt_count": attempt,
            "resolved":      False,
        }


def _grade_resolution(response_text: str) -> bool:
    """
    Fallback grader: separate LLM call asks whether the response
    actually resolves a customer issue. Returns True/False.
    """
    system = (
        "You are a QA reviewer for customer support.\n"
        "Does the following response fully resolve a customer's issue?\n"
        "Reply ONLY with JSON: {\"resolved\": true} or {\"resolved\": false}"
    )
    try:
        raw = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=response_text)]
        ).content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw).get("resolved", False)
    except:
        return False


# ════════════════════════════════════════════════════
# NODE 3 — CHECK (router pass-through)
# ════════════════════════════════════════════════════
def check_node(state: SupportAgentState) -> dict:
    print(
        f"\n[check_node] resolved={state.get('resolved')} | "
        f"attempts={state.get('attempt_count')} | "
        f"sentiment={state.get('sentiment')}"
    )
    return {}


def route_after_check(
    state: SupportAgentState,
) -> Literal["end", "rag_resolve", "hitl"]:
    resolved  = state.get("resolved", False)
    attempts  = state.get("attempt_count", 0)
    sentiment = state.get("sentiment", "neutral")

    if resolved:
        print("  [router] ✅ Resolved → END")
        return "end"
    elif attempts < 3:
        print(f"  [router] 🔁 Retry attempt {attempts + 1} → rag_resolve")
        return "rag_resolve"
    else:
        # Exhausted retries → human-in-the-loop before escalation
        print("  [router] 🚨 Exhausted retries → HITL")
        return "hitl"


# ════════════════════════════════════════════════════
# NODE 4 — HUMAN IN THE LOOP (interrupt before escalation)
# ════════════════════════════════════════════════════
def hitl_node(state: SupportAgentState) -> Command:
    """
    Pauses graph and asks a human agent to decide:
      'escalate' → raise ticket  |  'retry' → one more rag attempt
    In terminal demo: input() simulates the human agent console.
    In production: replace with webhook / Slack bot / dashboard.
    """
    print(f"\n{'='*60}")
    print(f"🧑 HUMAN-IN-THE-LOOP REQUIRED")
    print(f"  Ticket candidate | Sentiment: {state.get('sentiment')} | Attempts: {state.get('attempt_count')}")
    print(f"  Last user message: {next((m.content for m in reversed(state['messages']) if isinstance(m, HumanMessage)), '')[:80]}")
    print(f"{'='*60}")

    # interrupt() pauses the graph and surfaces data to the caller.
    # In LangGraph 1.1.6, resume via graph.invoke(Command(resume=value), config)
    decision = interrupt({
        "message":  "Automated resolution failed. Escalate or allow one more retry?",
        "options":  ["escalate", "retry"],
        "attempts": state.get("attempt_count"),
        "sentiment": state.get("sentiment"),
    })

    # decision is whatever was passed to Command(resume=...)
    decision = str(decision).strip().lower()
    if decision not in ("escalate", "retry"):
        decision = "escalate"  # safe default

    print(f"  Human decision: {decision}")

    if decision == "retry":
        return Command(
            goto="rag_resolve",
            update={"human_decision": "retry", "attempt_count": state.get("attempt_count", 0)},
        )
    else:
        return Command(
            goto="escalate",
            update={"human_decision": "escalate"},
        )


# ════════════════════════════════════════════════════
# NODE 5 — ESCALATE
# ════════════════════════════════════════════════════
def escalate_node(state: SupportAgentState) -> dict:
    ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"
    reason    = (
        f"Sentiment:{state.get('sentiment','?')} | "
        f"Attempts:{state.get('attempt_count',0)} | "
        f"HumanDecision:{state.get('human_decision','auto')}"
    )
    print(f"\n{'='*60}")
    print(f"🚨 ESCALATION TRIGGERED")
    print(f"  Ticket ID : {ticket_id}")
    print(f"  Reason    : {reason}")
    print(f"{'='*60}")

    # ── Future production hooks (uncomment when ready) ──
    # await save_ticket_to_neon(ticket_id, state)
    # await send_slack_alert(ticket_id, reason)
    # await send_customer_email(ticket_id, state["messages"])

    msg = AIMessage(content=(
        f"I've created support ticket **{ticket_id}** for you. "
        f"Our team will contact you within 2 hours. "
        f"Track your ticket at: support.techflow.com/tickets/{ticket_id}"
    ))
    return {
        "messages":          [msg],
        "escalated":         True,
        "ticket_id":         ticket_id,
        "escalation_reason": reason,
    }


# ════════════════════════════════════════════════════
# BUILD GRAPH
# ════════════════════════════════════════════════════
def build_agent():
    builder = StateGraph(SupportAgentState)

    # Nodes
    builder.add_node("classify",    classify_node)
    builder.add_node("rag_resolve", rag_resolve_node)
    builder.add_node("check_node",  check_node)
    builder.add_node("hitl",        hitl_node)        # ← Human-in-the-loop
    builder.add_node("escalate",    escalate_node)

    # Edges
    builder.add_edge(START,          "classify")
    builder.add_edge("classify",     "rag_resolve")
    builder.add_edge("rag_resolve",  "check_node")
    builder.add_conditional_edges(
        "check_node",
        route_after_check,
        {"end": END, "rag_resolve": "rag_resolve", "hitl": "hitl"},
    )
    # hitl_node uses Command(goto=...) so no explicit edges needed from it
    builder.add_edge("escalate", END)

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=[],   # interrupts handled inside hitl_node via interrupt()
    )


# ════════════════════════════════════════════════════
# MAIN — Terminal Chat Loop
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    graph  = build_agent()
    config = {"configurable": {"thread_id": "support_001"}}

    print("\n" + "█" * 60)
    print("█     CUSTOMER SUPPORT AGENT v2.0 — Eligent AI        █")
    print("█     Type 'quit' to exit                              █")
    print("█" * 60 + "\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            version="v2",          # LangGraph 1.1.6 — returns GraphOutput
        )

        # ── Handle interrupt (HITL pause) ──
        # If the graph hit an interrupt(), result.interrupts will be non-empty
        while result.interrupts:
            interrupt_data = result.interrupts[0]
            print(f"\n⚠️  AGENT PAUSED — Human decision required")
            print(f"   {interrupt_data.value.get('message')}")
            print(f"   Options: {interrupt_data.value.get('options')}")
            decision = input("   Human agent decision [escalate/retry]: ").strip().lower()
            if decision not in ("escalate", "retry"):
                decision = "escalate"
            # Resume graph with human decision
            result = graph.invoke(
                Command(resume=decision),
                config=config,
                version="v2",
            )

        # ── Print final agent reply ──
        val = result.value
        last_ai = next(
            (m.content for m in reversed(val["messages"]) if isinstance(m, AIMessage)), ""
        )
        print(f"\nAgent: {last_ai}")
        print(
            f"\n[Cat:{val.get('category','?')} | "
            f"Attempts:{val.get('attempt_count',0)} | "
            f"Resolved:{val.get('resolved',False)} | "
            f"Escalated:{val.get('escalated',False)} | "
            f"Ticket:{val.get('ticket_id','None')}]\n"
        )
