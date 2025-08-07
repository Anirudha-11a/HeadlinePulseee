




from typing import List, Dict
import os
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from google import genai  # Gemini SDK

load_dotenv()

two_weeks_ago_str = (datetime.today() - timedelta(days=14)).strftime('%Y-%m-%d')

class MCPOverloadedError(Exception):
    pass

mcp_limiter = AsyncLimiter(1, 15)
gemini_client = genai.Client()

server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("API_TOKEN"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    args=["@brightdata/mcp"],
)

async def summarize_with_gemini(text: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text
    )
    return response.text

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=15, max=60),
    retry=retry_if_exception_type(MCPOverloadedError),
    reraise=True
)
async def process_topic(agent, topic: str) -> str:
    async with mcp_limiter:
        messages = [
            SystemMessage(content=(
                f"You are a Reddit analysis expert. Use available tools to:\n"
                f"1. Find top 2 posts about '{topic}', only after {two_weeks_ago_str}.\n"
                "2. Analyze sentiment.\n"
                "3. Summarize discussion and overall sentiment."
            )),
            HumanMessage(content="Analyze Reddit posts. Summarize key points, quotes (no usernames), and overall sentiment.")
        ]

        try:
            result = await agent.invoke({"messages": messages})
            summary_text = result["messages"][-1].content
            return await summarize_with_gemini(summary_text)
        except Exception as e:
            if "Overloaded" in str(e):
                raise MCPOverloadedError("Service overloaded")
            else:
                raise

async def scrape_reddit_topics(topics: List[str]) -> Dict[str, Dict[str, str]]:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            model = ChatAnthropic(
                model="claude-3-5-sonnet-20240620",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3,
                max_tokens=4000
            )

            agent = create_react_agent(
                model=model,
                tools=tools,
                prompt=SystemMessage(content="You are a tool-using agent that analyzes Reddit discussion.")
            )

            results = {}
            for topic in topics:
                summary = await process_topic(agent, topic)
                results[topic] = summary
                await asyncio.sleep(5)

            return {"reddit_analysis": results}
