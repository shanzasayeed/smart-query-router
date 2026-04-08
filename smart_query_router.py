import os, json
from typing import TypedDict, Literal, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

class SupportState(TypedDict):
    user_message: str
    category: Optional[str]
    confidence: Optional[float]
    response: Optional[str]
    metadata: Optional[dict]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def classify_node(state: SupportState) -> dict:
    print(f"\n{'='*55}\n🔍 CLASSIFYING: '{state['user_message']}'\n{'='*55}")
    system_prompt = """Classify the customer support message into ONE category.
Return ONLY JSON (no markdown):
{"category": "technical"|"billing"|"general", "confidence": 0.0-1.0, "reasoning": "one line"}"""
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=state["user_message"])])
    try:
        parsed = json.loads(response.content)
        category, confidence = parsed.get("category","general"), parsed.get("confidence",0.8)
        reasoning = parsed.get("reasoning","")
    except:
        category, confidence, reasoning = "general", 0.5, "parse failed"
    print(f"  📂 Category: {category.upper()} | 📊 {confidence:.0%} | 💭 {reasoning}")
    return {"category": category, "confidence": confidence, "metadata": {"reasoning": reasoning}}

def technical_support_node(state: SupportState) -> dict:
    print(f"\n{'='*55}\n🔧 TECHNICAL SUPPORT NODE\n{'='*55}")
    system = "You are a senior technical support engineer. Be specific, professional (3-5 sentences)."
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=state["user_message"])])
    return {"response": response.content, "metadata": {**state.get("metadata",{}), "handled_by":"Technical", "priority":"high"}}

def billing_support_node(state: SupportState) -> dict:
    print(f"\n{'='*55}\n💳 BILLING SUPPORT NODE\n{'='*55}")
    system = "You are a billing specialist. Be empathetic, clear (3-4 sentences)."
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=state["user_message"])])
    return {"response": response.content, "metadata": {**state.get("metadata",{}), "handled_by":"Billing", "priority":"high"}}

def general_support_node(state: SupportState) -> dict:
    print(f"\n{'='*55}\n💬 GENERAL SUPPORT NODE\n{'='*55}")
    system = "You are a friendly customer support agent. Be warm, helpful (2-3 sentences)."
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=state["user_message"])])
    return {"response": response.content, "metadata": {**state.get("metadata",{}), "handled_by":"General", "priority":"low"}}

def route_query(state: SupportState) -> Literal["technical", "billing", "general"]:
    category = state.get("category", "general")
    print(f"\n  🔀 Routing to: {category.upper()} node")
    return category

def build_graph():
    builder = StateGraph(SupportState)
    builder.add_node("classify", classify_node)
    builder.add_node("technical", technical_support_node)
    builder.add_node("billing", billing_support_node)
    builder.add_node("general", general_support_node)
    
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_query, {"technical":"technical","billing":"billing","general":"general"})
    builder.add_edge("technical", END)
    builder.add_edge("billing", END)
    builder.add_edge("general", END)
    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    test_queries = [
        "My API integration keeps returning 401 unauthorized error",
        "I was charged twice for my subscription this month",
        "How do I export my data to CSV format?",
    ]
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'▓'*55}\n▓  TEST {i}/{len(test_queries)}\n{'▓'*55}")
        result = graph.invoke({"user_message": query, "category": None, "confidence": None, "response": None, "metadata": {}})
        print(f"\n{'═'*55}")
        print(f"  Category : {result['category'].upper()} ({result['confidence']:.0%})")
        print(f"  Handled  : {result['metadata']['handled_by']}")
        print(f"\n  🤖 RESPONSE:\n  {result['response'][:200]}...")
        print(f"{'═'*55}")
        if i < len(test_queries): input("\n  Press Enter for next...")                                                                                                                                                                                                                                                                  