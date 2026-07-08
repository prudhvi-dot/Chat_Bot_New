from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import sqlite3

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph = StateGraph(chat_state)


def chat_node(state: chat_state) -> chat_state:
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


con = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=con)

graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
