import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model = 'mistral-medium-2508',  # MISTRAL_API_KEY, langchain[mistral]
    temperature = 0.1,
)

response = model.stream('Hello, what is Python?')

for chunk in response:
    print(chunk.text, end='', flush=True)
    
