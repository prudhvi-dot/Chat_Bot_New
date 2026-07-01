from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class chat_state(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


graph = StateGraph(chat_state)


def chat_node(state: chat_state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chat_bot = graph.compile()

initial_state = {"messages": [HumanMessage(content="what is the capital of india")]}

final_state = chat_bot.invoke(initial_state)

print(final_state["messages"][-1])
