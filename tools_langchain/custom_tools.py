from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain.messages import HumanMessage , SystemMessage, AIMessage
from dotenv import load_dotenv
from rich import print
load_dotenv()

@tool
def getLength(text:str)->str:
    """ returns number of chracter in given text"""
    return len(text)

tools={
    "getLength" : getLength
}

llm = ChatMistralAI( model="open-mistral-7b")    
llm_with_tool = llm.bind_tools([getLength]) #llm with tool

message=[] #for maintaining human message, system message and ai message

query=HumanMessage("return the number of characters in the given text : 'hello how are you' ")
message.append(query)

result= llm_with_tool.invoke(message) #result has tool_call
message.append(result)


if result.tool_calls:
    tool_name= result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
    #print(message)
    
answer = llm_with_tool.invoke(message)  
print(answer.content)  
    





