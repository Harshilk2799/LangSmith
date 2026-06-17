from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from decouple import config
from dotenv import load_dotenv

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_API_KEY"))

parser = StrOutputParser()

chain = prompt | chat_model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)