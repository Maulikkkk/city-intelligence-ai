from dotenv import load_dotenv
load_dotenv()
import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage,SystemMessage,ToolMessage
from tavily import TavilyClient
from rich import print


# --------------------------------------weather tool----------------------------------------------------------------------------
@tool
def get_weather(city:str)-> str:
    """Get current weather of city"""
    
    api_key=os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    print("DEBUG:", data)
    if str(data.get("cod"))!="200":
        return f"Error: {data.get('message','could not fetch weather')}"
    temp = data["main"]["temp"]
    desc=data["weather"][0]["description"]
    
    return f"weather in {city} : is {desc} and tempreture is  {temp} Celcius"
# print(get_weather.invoke("Ghaziabad"))


# --------------------------------------news tool ----------------------------------------------------------------------------
@tool    
def get_news(city:str)->str:
    """Get latest news of city"""
    tavily_client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(
        query=f"Latest news from {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        content = r.get("content", "No content")
        url = r.get("url", "")
        snippet=r.get("content", "")

        news_list.append(
            f"-{title}\n and {url}\n {snippet[:100]}..."
        )

    return f"lastest news in {city}:\n\n" + "\n".join(news_list)
# print(get_news.invoke("ghaziabad"))


llm=ChatMistralAI(model="open-mistral-7b")
tools = {
    "get_news": get_news,
    "get_weather": get_weather
}
lmm_with_tool = llm.bind_tools([get_weather,get_news])

#--------------------------------------Agent loop ----------------------------------------------------------------------------
messages=[]
print("city intelligence system")
print("type Exit to quite")

while True:
    user_input=input("you : ")
    if user_input.lower()=="exit":
        break
    messages.append(HumanMessage(user_input))
    
    while True:
        result= lmm_with_tool.invoke(messages)
        messages.append(result)
        
        #if tool is required
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name=tool_call['name']
                
                #human_in_the_loop
                confirm=input(f"agent want to call {tool_name}. Approve (yes/no)")
                if confirm.lower()=="no":
                    print("tool call denied")
                    break
                tool_result= tools[tool_name].invoke(tool_call)
                messages.append(ToolMessage(content = tool_result,tool_call_id=tool_call['id']))
            continue
        else :
            print("\n Final Answer")
            print(result.content)
            print("\n" + "="*50 + "\n")
            break
            
                
                
                    
                    