from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash")

agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant",
)

response = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

for i in response["messages"]:
    print(i.pretty_print())

