from base64 import b64encode
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

model = init_chat_model(
    model="gpt-4.1-mini",
)


messages = HumanMessage(
    content =[
        {"type": "text", "text": "Describe the contents of this image."},
        {
            "type": "image",
            # "url": "https://1000logos.net/wp-content/uploads/2018/10/Minecraft-Logo.png", # minecraft logo url
            "base64": b64encode(open("Minecraft-Logo.png", "rb").read()).decode(),
            "mime_type": "image/png",
        }
    ]
)

response = model.invoke([messages])

print(response.content)
