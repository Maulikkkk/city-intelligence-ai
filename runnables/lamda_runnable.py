from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
load_dotenv()

model = ChatMistralAI(model="open-mistral-7b")
parser = StrOutputParser()


short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 2-3 lines")
detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail paragraph")

#rannable-lambda function
parallell_chain = RunnableParallel({  
   "short": RunnableLambda(lambda x:x["short"]) | short_prompt | model | parser,
    "long":RunnableLambda(lambda x:x["long"]) | detailed_prompt | model | parser
})

response = parallell_chain.invoke({
    "short": "Machine Learning",
    "long": "deep learning"
})

print(response.get("short"))
print(response.get("long"))


