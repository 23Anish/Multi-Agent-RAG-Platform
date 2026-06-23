"""
LangGraph Agent Orchestrator.

Graph topology:
  START
    │
    ▼
  planner        ← decides which tools to call and in what order
    │
    ▼
  tool_executor  ← calls MCP tools (retrieve, list_docs, sql_query)
    │  ▲
    │  │ (loop if more tool calls needed)
    ▼  │
  should_continue? ──yes──► tool_executor
    │
    no
    ▼
  synthesiser    ← builds final answer with citations from tool outputs
    │
    ▼
  END

State is a TypedDict passed immutably between nodes.
Each node returns a dict of keys to update — LangGraph merges them.
"""
import json
import logging
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import get_settings
from app.mcp.client import build_mcp_tools

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # `add_messages` reducer appends rather than overwrites on each node update
    messages: Annotated[list[BaseMessage], add_messages]
    tenant_id: str
    query: str
    tool_calls_log: list[dict[str, Any]]   # audit trail
    final_answer: str
    sources: list[dict[str, Any]]
    iteration: int


# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm(tools: list | None = None) -> ChatBedrock:
    llm = ChatBedrock(
        model_id=settings.bedrock_chat_model,
        region_name=settings.aws_region,
        model_kwargs={
            "max_tokens": 4096,
            "temperature": 0.0,   # deterministic for RAG
            "top_p": 0.95,
        },
        streaming=True,
    )
    if tools:
        llm = llm.bind_tools(tools)
    return llm


# ── Node: planner ─────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a precise research agent with access to a knowledge base.

Your job:
1. Analyse the user query carefully.
2. Decide which tools to call and in what order.
3. Always start with `retrieve_chunks` to search the knowledge base.
4. If you need structured metadata, use `sql_query`.
5. If you are unsure what documents exist, call `list_documents` first.
6. After gathering enough context, stop calling tools and synthesise an answer.

Rules:
- Never make up information. Only use retrieved content.
- Always cite the source chunk_id and filename in your final answer.
- If the knowledge base has no relevant information, say so clearly.
- Maximum {max_iter} tool call rounds.
"""


async def planner_node(state: AgentState) -> dict:
    tools = build_mcp_tools(state["tenant_id"])
    llm = _build_llm(tools)

    system_msg = PLANNER_SYSTEM.format(max_iter=settings.max_agent_iterations)
    messages = [HumanMessage(content=system_msg), *state["messages"]]

    response: AIMessage = await llm.ainvoke(messages)

    logger.debug("Planner response: tool_calls=%d", len(response.tool_calls or []))
    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }


# ── Node: tool_executor ───────────────────────────────────────────────────────

async def tool_executor_node(state: AgentState) -> dict:
    tools = build_mcp_tools(state["tenant_id"])
    tool_map = {t.name: t for t in tools}

    last_ai_msg: AIMessage = state["messages"][-1]
    tool_results: list[ToolMessage] = []
    audit_log = list(state.get("tool_calls_log", []))
    sources = list(state.get("sources", []))

    for tc in last_ai_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]

        logger.info("Executing tool: %s args=%s", tool_name, tool_args)
        t0 = time.perf_counter()

        try:
            tool = tool_map[tool_name]
            result = await tool.ainvoke(tool_args)
        except KeyError:
            result = f"ERROR: unknown tool '{tool_name}'"
        except Exception as exc:
            result = f"ERROR: {exc}"

        latency_ms = int((time.perf_counter() - t0) * 1000)

        audit_log.append({
            "tool": tool_name,
            "args": tool_args,
            "latency_ms": latency_ms,
        })

        # Extract source chunks from retrieve results for citation tracking
        if tool_name == "retrieve_chunks" and "ERROR" not in str(result):
            _extract_sources(result, sources)

        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
        )

    return {
        "messages": tool_results,
        "tool_calls_log": audit_log,
        "sources": sources,
    }


def _extract_sources(retrieve_output: str, sources: list) -> None:
    """Parse chunk metadata from the formatted retrieve_chunks output."""
    for line in retrieve_output.splitlines():
        if "chunk_id=" in line:
            try:
                chunk_id = line.split("chunk_id=")[1].strip()
                sources.append({"chunk_id": chunk_id})
            except IndexError:
                pass
        if "Source:" in line:
            try:
                meta = line.split("Source:")[1].strip()
                if sources:
                    sources[-1]["source_meta"] = meta
            except IndexError:
                pass


# ── Node: synthesiser ─────────────────────────────────────────────────────────

SYNTH_SYSTEM = """You are a helpful assistant synthesising a final answer from retrieved context.

Instructions:
- Write a clear, complete answer using ONLY the context provided by the tools.
- Format the answer in Markdown with headings where appropriate.
- At the end, include a **Sources** section listing each cited filename + page.
- If the context is insufficient, say: "I could not find enough information to answer this question."
- Be concise but thorough. No padding.
"""


async def synthesiser_node(state: AgentState) -> dict:
    llm = _build_llm()  # no tools — synthesis is pure generation

    messages = [HumanMessage(content=SYNTH_SYSTEM), *state["messages"]]
    response: AIMessage = await llm.ainvoke(messages)

    return {
        "messages": [response],
        "final_answer": response.content,
    }


# ── Conditional edge: should we keep calling tools? ───────────────────────────

def should_continue(state: AgentState) -> Literal["tool_executor", "synthesiser"]:
    last_msg = state["messages"][-1]
    iteration = state.get("iteration", 0)

    # Force synthesis after max iterations to prevent infinite loops
    if iteration >= settings.max_agent_iterations:
        logger.warning("Max agent iterations reached — forcing synthesis")
        return "synthesiser"

    # If the last AI message has tool calls → execute them
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_executor"

    # Otherwise the planner decided it has enough context
    return "synthesiser"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("synthesiser", synthesiser_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", should_continue)
    graph.add_edge("tool_executor", "planner")   # loop back for multi-hop
    graph.add_edge("synthesiser", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_compiled_graph = None


def get_agent_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


async def run_agent(
    query: str,
    tenant_id: str,
    session_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """
    Run the full agent pipeline for a query.
    Returns: { final_answer, sources, tool_calls_log, latency_ms }
    """
    graph = get_agent_graph()
    t0 = time.perf_counter()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "tenant_id": tenant_id,
        "query": query,
        "tool_calls_log": [],
        "final_answer": "",
        "sources": [],
        "iteration": 0,
    }

    final_state = await graph.ainvoke(initial_state)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "final_answer": final_state["final_answer"],
        "sources": final_state.get("sources", []),
        "tool_calls_log": final_state.get("tool_calls_log", []),
        "latency_ms": latency_ms,
    }


async def stream_agent(query: str, tenant_id: str):
    """
    Async generator that yields AgentEvent dicts for SSE streaming.
    Yields token-by-token text + tool call events as they happen.
    """
    graph = get_agent_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "tenant_id": tenant_id,
        "query": query,
        "tool_calls_log": [],
        "final_answer": "",
        "sources": [],
        "iteration": 0,
    }

    async for event in graph.astream_events(initial_state, version="v2"):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chat_model_stream" and name == "synthesiser":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                yield {"event": "token", "data": chunk.content}

        elif kind == "on_tool_start":
            yield {
                "event": "tool_call",
                "data": {"tool": event["name"], "args": event["data"].get("input", {})},
            }

        elif kind == "on_chain_end" and name == "synthesiser":
            output = event["data"].get("output", {})
            yield {"event": "done", "data": {"sources": output.get("sources", [])}}
