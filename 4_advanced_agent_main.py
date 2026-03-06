from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool(
    "get_weather",
    description="Get the current weather for a given location",
    return_direct=False,
)
def get_weather(city: str):
    response = requests.get(f"http://wttr.in/{city}?format=j1")
    return response.json()


@tool(
    "locate_user",
    description="Look up a user's city based on the context",
    return_direct=False,
)
def locate_user(runtime: ToolRuntime[Context]) -> str:
    match runtime.context.user_id:
        case "123":
            return "Zagreb"
        case "456":
            return "Dubrovnik"
        case _:
            return "Unknown"


model = init_chat_model(
    model="gpt-4.1-mini",
    temperature=0.3,
)

checkpointer = InMemorySaver()

# Agents are LLM‑driven orchestrators. They can decide what to do next, call tools/functions, interpret tool results, and continue reasoning.
agent = create_agent(
    model=model,  # OPENAI_API_KEY, langchain[openai]
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful assistant that can provide weather information for any city.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": 1}}

# 1st question - agent should call locate_user tool, then get_weather tool, and return the final response
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the current weather in my city?"},
        ]
    },
    context=Context(user_id="123"),
    config=config,
)

print()
print("--- 1st response ---")
# print(response["structured_response"])
print(response["structured_response"].summary)
print(f"Temperature celsius: {response['structured_response'].temperature_celsius}")
# print(f'Temperature fahrenheit: {response["structured_response"].temperature_fahrenheit}')
# print(f"Humidity: {response['structured_response'].humidity}")


config = {
    "configurable": {"thread_id": 2}
}  # Different thread_id to simulate a different conversation, but the same user

# 2nd question - agent should use the cached results from the previous question and return the final response without calling any tools
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "And is this usual?"},
        ]
    },
    context=Context(user_id="123"),
    config=config,
)

print()
print("--- 2nd response ---")
print(response["structured_response"].summary)
print()
