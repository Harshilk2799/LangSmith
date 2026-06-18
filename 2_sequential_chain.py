from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from decouple import config
from dotenv import load_dotenv
import os 

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "Sequential Chain App"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_API_KEY"))

parser = StrOutputParser()

chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)