import time

from dotenv import load_dotenv
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()


class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState) -> None:
        print("Before agent start...")
        self.start_time = time.time()

    def before_model(self, state, runtime):
        print("Before model call...")
    
    def after_model(self, state, runtime):
        print("After model call...")

    def after_agent(self, state, runtime):
        print("After agent", time.time() - self.start_time)


agent = create_agent(
    model=init_chat_model(model="gpt-4.1-mini"),
    middleware=[HooksDemo()],  # Instance of a class
)

response = agent.invoke(
    {
        "messages": [
            SystemMessage(content="What is PCA and how does it work?"),
            HumanMessage(content="What is PCA?"),
        ]
    }
)

print(response["messages"][-1].content)
