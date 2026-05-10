from dotenv import load_dotenv
load_dotenv()
import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage,SystemMessage,ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


# --------------------------------------weather tool----------------------------------------------------------------------------
@tool
def get_weather(city:str)-> str:
    """Get current weather of city"""
    
    api_key=os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    #print("DEBUG:", data)
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

@wrap_tool_call
def handle_tool_request(request, handler):
    """ask human for permission for using tool."""
    tool_name=request.tool_call["name"]
    confirm = input(f"agent want to call '{tool_name}'. Approve? (yes/no): ")
    if confirm!="yes":
        return ToolMessage(
            content=f"Tool call denied.",
            tool_call_id=request.tool_call["id"]
        )
    return handler(request)    


llm=ChatMistralAI(model="open-mistral-7b")
agent = create_agent(llm, tools=[get_news,get_weather],system_prompt="you are helpful city assistant",middleware=[handle_tool_request])

print("city agent | type exit to quit")
while True:
    user_input= input("you : ")
    if user_input.lower()=="exit":
        break
    result = agent.invoke({
            "messages":[{"role": "user",
                         "content":user_input}]
            })
    print(result["messages"][-1].content)
