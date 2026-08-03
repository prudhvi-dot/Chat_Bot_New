import sqlite3
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

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


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision("yes" / anything else).
    """

    decision = interrupt("Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }

    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [get_stock_price, web_search, purchase_stock]

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


if __name__ == "__main__":
    thread_id = "demo_thread"

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state = {"messages": [HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state, config={"configurable": {"thread_id": thread_id}}
        )

        interrupts = result.get("__interrupt__", [])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )

        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot:{last_msg.content}\n")
