# import sqlite3
import asyncio
from typing import Annotated, TypedDict

import aiosqlite
import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
token = "fmcp_BU6a__OTSQI-JokKRwndmZOyQep_UO9DbZSyccU06Zo"

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

client = MultiServerMCPClient(
    {
        "Remote-testing": {
            "transport": "streamable_http",
            "url": "https://popular-apricot-tick.fastmcp.app/mcp",
            "headers": {"Authorization": f"Bearer {token}"},
        }
    }
)


class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """
    Search the internet for any factual information, including people,
    companies, news, current events, products, and general knowledge.
    Use this whenever information may not be in the model's memory.
    """
    return search.invoke(query)


@tool
def get_stock_price(symbol: str):
    """
    fetches the stock price of given company
    """
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token=d98c4v1r01qr7o1ifcfgd98c4v1r01qr7o1ifcg0"
    response = requests.get(url)
    return response.json()


# tools = [get_stock_price, web_search]


graph = StateGraph(chat_state)


async def build_graph():
    tools = await client.get_tools()
    llm_with_tools = model.bind_tools(tools, tool_choice="auto")

    async def chat_node(state: chat_state) -> chat_state:
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    conn = await aiosqlite.connect("chatbot.db")
    checkpointer = AsyncSqliteSaver(conn)

    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    chatbot = graph.compile(checkpointer=checkpointer)
    return chatbot


async def main():
    chatbot = await build_graph()
    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content="give me the server info")]},
        config={"configurable": {"thread_id": "1"}},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
