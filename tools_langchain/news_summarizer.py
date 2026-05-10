#prebuilt tool in langchain

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatMistralAI(
    model="open-mistral-7b"
)
# Initialize parser
parser = StrOutputParser()

# Initialize search tool
search_tool = TavilySearch(max_results=5)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
You are an AI news summarizer.

Summarize the following AI news into:
- Clear bullet points
- Simple language
- Important highlights only

News:
{news}
"""
)
# Fetch news
news_result = search_tool.run("Trending AI news")

# Create chain
chain = prompt | llm | parser

# Generate response
result = chain.invoke({
    "news": news_result
})

# Print final output
print(result)