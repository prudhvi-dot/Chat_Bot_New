from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import requests
from typing import TypedDict, Annotated
import sqlite3

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph = StateGraph(chat_state)

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


tools = [get_stock_price, web_search]

llm_with_tools = model.bind_tools(tools, tool_choice="auto")


def chat_node(state: chat_state) -> chat_state:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


con = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=con)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
