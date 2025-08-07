

import asyncio
import os
from typing import Dict, List

from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions

from utils import (
    generate_news_urls_to_scrape,
    scrape_with_brightdata,
    clean_html_to_text,
    extract_headlines
)

load_dotenv()

# Initialize Gemini client
def get_gemini_client():
    return genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=HttpOptions(api_version="v1alpha")  # or 'v1' if using Vertex AI
    )


class NewsScraper:
    _rate_limiter = AsyncLimiter(5, 1)  # 5 req/sec

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def scrape_news(self, topics: List[str]) -> Dict[str, str]:
        client = get_gemini_client()
        results = {}

        for topic in topics:
            async with self._rate_limiter:
                try:
                    urls = generate_news_urls_to_scrape([topic])
                    html = scrape_with_brightdata(urls[topic])
                    clean_text = clean_html_to_text(html)
                    headlines = extract_headlines(clean_text)

                    prompt = (
                        f"Summarize the news headlines below for topic: {topic}\n\n"
                        f"Headlines:\n" + "\n".join(headlines)
                    )

                    response = client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt
                    )
                    results[topic] = response.text
                except Exception as e:
                    results[topic] = f"Error: {str(e)}"
                await asyncio.sleep(1)

        return {"news_analysis": results}
