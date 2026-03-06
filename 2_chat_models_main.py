import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model = 'gpt-4.1-mini',
    temperature = 0.1,
)

conversation = [
    SystemMessage(content='You are a helpful assistant that can provide information about programming languages.'),
    HumanMessage(content='Hello, what is Python?'),
    AIMessage(content='Python is a high-level, interpreted programming language known for its readability and versatility. '),
    HumanMessage(content='When it was released?')
]

# response = model.invoke('Hello, what is Python?')
response = model.invoke(conversation)

print(response.content)
