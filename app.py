"""
Majdoor AI backend
Free LLM (g4f) + free web/image search (ddgs), exposed as a simple API
for the Majdoor AI Android app to call.

Deploy free on Render.com or Hugging Face Spaces (Docker option).
"""

import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from g4f.client import Client
from ddgs import DDGS

app = FastAPI(title="Majdoor AI Backend")

# Allow the Android app (and any origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client()

SYSTEM_PROMPT = (
    "Tu 'Majdoor AI' hai - ek sarcastic, funny Hinglish chatbot jo mazdoor "
    "(daily-wage worker) ke andaaz mein baat karta hai. Chhota, punchy, "
    "taane-mare-hue jawaab de. Hamesha Hinglish mein reply kar, English mein nahi."
)


class ChatRequest(BaseModel):
    message: str


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> or similar chain-of-thought leakage
    that some free/reasoning providers include in their output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def do_chat(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    reply = response.choices[0].message.content or ""
    return strip_reasoning(reply)


def do_web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "Kuch nahi mila bhai, dusra kuch pooch."
    lines = [f"- {r['title']}: {r['body'][:140]}" for r in results]
    return "\n".join(lines)


def do_image_search(query: str) -> list:
    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=7))
    return [r["image"] for r in results if "image" in r]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message.strip()

    if msg.lower().startswith("img/"):
        query = msg[4:].strip()
        images = do_image_search(query)
        if not images:
            return {"type": "text", "reply": "Photo nahi mili ustad, spelling check kar."}
        return {"type": "images", "caption": f"'{query}' ke liye ye mila:", "images": images}

    if msg.lower().startswith("dd/"):
        query = msg[3:].strip()
        result = do_web_search(query)
        return {"type": "text", "reply": result}

    if msg.lower().startswith("duck/"):
        query = msg[5:].strip()
        reply = do_chat(query)
        return {"type": "text", "reply": reply}

    reply = do_chat(msg)
    return {"type": "text", "reply": reply}
