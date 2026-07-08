from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph = StateGraph(chat_state)


def chat_node(state: chat_state) -> chat_state:
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


checkpointer = InMemorySaver()

graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)
