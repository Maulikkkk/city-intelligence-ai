from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
load_dotenv()

model=ChatMistralAI(model="open-mistral-7b")
parser = StrOutputParser()

code_prompt= ChatPromptTemplate.from_messages([
    ("system", "you are a python code genrator"),
    ("human","{topic}")
])
explain_promt=ChatPromptTemplate.from_messages([
    ("system","you are helpfull assistant who expalin codes in simple words"),
    ("human","Explain the following code in words:\n{code}")
])

# way 1 - without RunnablePassthrough()
# seq1 = code_prompt | model | parser 
# seq2= RunnableParallel({
#     "code":seq1,
#     "explanation": seq1 | explain_promt | model | parser
# })
# response = seq2.invoke({"topic":"reverse array"})
# print("\n💻 Code:\n", response["code"])
# print("\n📖 Explanation:\n", response["explanation"])


# way 2 - with RunnablePassthrough()
# chain = (
#     code_prompt
#     | model
#     | parser
#     | RunnableParallel({
#         "code": RunnablePassthrough(),   # keep code as-is
#         "explanation": explain_promt | model | parser
#     })
# )
# response= chain.invoke({"topic":"sort a array"})
# print("\n💻 Code:\n", response["code"])
# print("\n📖 Explanation:\n", response["explanation"])



#way 3 with RunnablePassthrough() more clear way
seq1 = code_prompt | model | parser

seq2= RunnableParallel({
    "code": RunnablePassthrough(),
    "explaination": explain_promt | model |parser
})

chain = seq1 | seq2
response = chain.invoke({"topic":"largest element in array"})

print("\n💻 Code:\n", response["code"])
print("\n📖 Explanation:\n", response["explaination"])

