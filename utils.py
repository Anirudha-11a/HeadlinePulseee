

from urllib.parse import quote_plus
from dotenv import load_dotenv
import requests
import os
from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from gtts import gTTS

from google import genai
load_dotenv()

# Initialize Gemini client using API key from environment
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_valid_news_url(keyword: str) -> str:
    return f"https://news.google.com/search?q={quote_plus(keyword)}&tbs=sbd:1"

def generate_news_urls_to_scrape(list_of_keywords):
    return {kw: generate_valid_news_url(kw) for kw in list_of_keywords}

def scrape_with_brightdata(url: str) -> str:
    try:
        resp = requests.post(
            "https://api.brightdata.com/request",
            headers={
                "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={"zone": os.getenv("BRIGHTDATA_WEB_UNLOCKER_ZONE"), "url": url, "format": "raw"}
        )
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BrightData error: {str(e)}")

def clean_html_to_text(html_content: str) -> str:
    return BeautifulSoup(html_content, "html.parser").get_text(separator="\n").strip()

def extract_headlines(cleaned_text: str) -> str:
    headlines, current = [], []
    for line in cleaned_text.splitlines():
        if (l := line.strip()):
            if l == "More" and current:
                headlines.append(current[0]); current = []
            else:
                current.append(l)
    if current:
        headlines.append(current[0])
    return "\n".join(headlines)

def summarize_with_gemini(headlines: str) -> str:
    prompt = f"""You are a news editor. Convert headlines into a smooth TV news script, no intros or formatting, ready for speech:

{headlines}

News Script:"""
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

def text_to_audio_elevenlabs_sdk(text: str, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_multilingual_v2", output_format="mp3_44100_128", output_dir="audio"):
    from elevenlabs import ElevenLabs
    from datetime import datetime

    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing ElevenLabs API key.")

    client = ElevenLabs(api_key=api_key)
    audio_stream = client.text_to_speech.convert(text=text, voice_id=voice_id, model_id=model_id, output_format=output_format)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
    with open(filepath, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)
    return filepath

# Example usage
if __name__ == "__main__":
    topics = ["economy", "technology"]
    urls = generate_news_urls_to_scrape(topics)
    snippet = clean_html_to_text(scrape_with_brightdata(urls[topics[0]]))
    headlines = extract_headlines(snippet)
    script = summarize_with_gemini(headlines)
    print(script)
