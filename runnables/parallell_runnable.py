from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model = ChatMistralAI(model="open-mistral-7b")
parser = StrOutputParser()


short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 2-3 lines")
detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail paragraph")

#parallel raunnable packed in dictionary
parallell_chain = RunnableParallel({  
   "short": short_prompt | model | parser,
    "long":detailed_prompt | model | parser
})

response = parallell_chain.invoke({"topic":"Machine learning"})
print(response.get("short"))
print(response.get("long"))


