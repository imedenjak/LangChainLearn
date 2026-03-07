from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

load_dotenv()


@dataclass
class Context:
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful and very consise assistant."

    match user_role:
        case "expert":
            return f"{base_prompt} The user is an expert in the field and is looking for detailed and technical information."
        case "beginner":
            return f"{base_prompt} The user is a beginner and is looking for a simple and easy-to-understand explanation."
        case "child":
            return f"{base_prompt} The user is a child and is looking for a very simple and engaging explanation with examples."
        case _:
            return base_prompt


agent = create_agent(
    model="gpt-4.1-mini", middleware=[user_role_prompt], context_schema=Context
)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is PCA and how does it work?"},
        ]
    },
    # context=Context(user_role='beginner')
    # context=Context(user_role="expert"),
    context=Context(user_role="child"),
)

print(response)
