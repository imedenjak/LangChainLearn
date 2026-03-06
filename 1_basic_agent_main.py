import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


@tool('get_weather', description='Get the current weather for a given location', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'http://wttr.in/{city}?format=j1')
    return response.json()


agent = create_agent(
    model='gpt-4.1-mini',  # OPENAI_API_KEY, langchain[openai]
    tools=[get_weather],
    system_prompt='You are a helpful assistant that can provide weather information for any city.',
)

response = agent.invoke(
    {"messages": [
        {"role": "user", "content": "What is the current weather in Zagreb?"}
    ]}
)

# print(response)
print(response['messages'][-1].content)
