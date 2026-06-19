import requests
from decouple import config
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from dotenv import load_dotenv
import os 

os.environ["LANGSMITH_PROJECT"] = "Agent"

load_dotenv()
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_API_KEY"))

agent = create_agent(
    model=chat_model,
    tools=[search_tool, get_weather_data],
)

response = agent.invoke({"messages": [HumanMessage(content="What is the current temp of gurgaon?")]})
print("AI Response: ",response['messages'][-1].content)
print()

# response = agent.invoke({"messages": [HumanMessage(content="What is the release date of Dhadak 2?")]})
# print("AI Response: ",response['messages'][-1].content)
# print()

# response = agent.invoke({"messages": [HumanMessage(content="Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.")]})
# print("AI Response: ",response['messages'][-1].content)