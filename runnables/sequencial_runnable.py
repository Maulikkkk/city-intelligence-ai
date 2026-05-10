from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt= ChatPromptTemplate.from_template(
    "explain {topic} in simple words"
)

model = ChatMistralAI(
    model="open-mistral-7b"
)
parser = StrOutputParser()

chain= prompt | model | parser
response = chain.invoke({"topic":"machine learning"})
print(response)