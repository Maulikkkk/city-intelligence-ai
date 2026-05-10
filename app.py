# app.py

from dotenv import load_dotenv
load_dotenv()

import os
import requests
import streamlit as st

from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.agents.middleware import wrap_tool_call


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="City Intelligence AI",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stChatMessage {
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 10px;
}

.user-msg {
    background: linear-gradient(135deg, #3B82F6, #2563EB);
    padding: 14px;
    border-radius: 18px;
    color: white;
    margin-bottom: 10px;
}

.bot-msg {
    background: #1E293B;
    padding: 14px;
    border-radius: 18px;
    color: white;
    margin-bottom: 10px;
    border: 1px solid #334155;
}

.tool-box {
    background: #111827;
    border-left: 5px solid #10B981;
    padding: 12px;
    border-radius: 12px;
    margin-top: 8px;
    margin-bottom: 8px;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
}

.subtitle {
    color: #94A3B8;
    font-size: 18px;
}

.stTextInput input {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="big-title">🌍 City Intelligence AI</div>
    <div class="subtitle">
        AI agent with Weather + News tools using LangChain & Mistral
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.title("⚡ Features")

    st.markdown("""
    - 🌤️ Weather Tool
    - 📰 News Tool
    - 🤖 AI Agent
    - 🔧 Tool Calling
    - 🧠 LangChain
    - 🔥 Streamlit UI
    """)

    st.divider()

    st.subheader("💡 Example Prompts")

    st.markdown("""
    - Weather in Delhi
    - Latest news from Mumbai
    - Is it raining in Noida?
    - What's happening in Bangalore?
    """)

    st.divider()

    st.caption("Built with LangChain + Mistral AI")


# ---------------------------------------------------------
# TOOLS
# ---------------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Get current weather of city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={city}&appid={api_key}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return (
        f"🌤️ Weather in {city}\n\n"
        f"Condition: {desc}\n"
        f"Temperature: {temp}°C"
    )


@tool
def get_news(city: str) -> str:
    """Get latest news of city"""

    tavily_client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

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
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"""
📰 {title}

🔗 {url}

{snippet[:150]}...
"""
        )

    return f"Latest News from {city}\n\n" + "\n\n".join(news_list)


# ---------------------------------------------------------
# TOOL MIDDLEWARE
# ---------------------------------------------------------

@wrap_tool_call
def handle_tool_request(request, handler):

    tool_name = request.tool_call["name"]

    st.toast(f"⚡ Agent is using tool: {tool_name}")

    return handler(request)


# ---------------------------------------------------------
# AGENT
# ---------------------------------------------------------

llm = ChatMistralAI(
    model="open-mistral-7b"
)

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="""
    You are a helpful city intelligence assistant.
    
    Help users with:
    - weather
    - city news
    - local updates
    
    Keep answers clean and readable.
    """,
    middleware=[handle_tool_request]
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="bot-msg">{msg["content"]}</div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

prompt = st.chat_input("Ask about any city...")

if prompt:

    # Store user msg
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    st.markdown(
        f'<div class="user-msg">{prompt}</div>',
        unsafe_allow_html=True
    )

    # Loading spinner
    with st.spinner("🤖 Thinking..."):

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })

        response = result["messages"][-1].content

    # Store AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.markdown(
        f'<div class="bot-msg">{response}</div>',
        unsafe_allow_html=True
    )