
from fastapi import FastAPI, HTTPException, Response
import os
from pathlib import Path
from dotenv import load_dotenv
import traceback
from models import NewsRequest
from utils import text_to_audio_elevenlabs_sdk
from news_scraper import NewsScraper
from reddit_scraper import scrape_reddit_topics

# Use the supported Google Gen AI SDK for Gemini
from google import genai  # Updated import — use the new SDK :contentReference[oaicite:0]{index=0}

app = FastAPI()
load_dotenv()

# Initialize the Gemini client once—uses GEMINI_API_KEY or GOOGLE_API_KEY from environment :contentReference[oaicite:1]{index=1}
client = genai.Client()

@app.get("/")
async def root():
    return {"message": "Your API is running!"}

@app.post("/generate-news-audio")
async def generate_news_audio(request: NewsRequest):
    try:
        results = {}
        if request.source_type in ["news", "both"]:
            news_scraper = NewsScraper()
            results["news"] = await news_scraper.scrape_news(request.topics)

        if request.source_type in ["reddit", "both"]:
            results["reddit"] = await scrape_reddit_topics(request.topics)

        news_summary = call_gemini_summary(
            news_data=results.get("news", {}),
            reddit_data=results.get("reddit", {}),
            topics=request.topics
        )

        audio_path = text_to_audio_elevenlabs_sdk(
            text=news_summary,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            output_dir="audio"
        )

        if audio_path and Path(audio_path).exists():
            return Response(
                content=Path(audio_path).read_bytes(),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=news-summary.mp3"}
            )

        raise HTTPException(status_code=500, detail="Audio generation failed.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def call_gemini_summary(news_data, reddit_data, topics):
    # Build a prompt with combined data
    prompt = (
        f"Create a broadcast news script summarizing the following:\n"
        f"Topics: {topics}\n"
        f"News Data: {news_data}\n"
        f"Reddit Data: {reddit_data}\n"
        f"Write in full paragraphs, optimized for speech.\n"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )  # Standard usage of the new SDK :contentReference[oaicite:2]{index=2}
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8002, reload=True)
