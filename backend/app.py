"""
Majdoor AI backend — same persona, same triggers, same sarcasm as the
original Streamlit app (majdoor__2___8_.py), exposed as a stateless API
so the Android/PWA app can call it.

Deploy free on Render.com or Hugging Face Spaces.
"""

import os
import re
import time
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import List, Optional

os.environ.setdefault("G4F_COOKIES_DIR", "/tmp/g4f_har_and_cookies")
os.makedirs(os.environ["G4F_COOKIES_DIR"], exist_ok=True)

import g4f
try:
    g4f.cookies_dir = os.environ["G4F_COOKIES_DIR"]
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Note: the original code's `from g4f.Provider import bing` never actually
# worked — the real class is `Bing` (capitalized) and it requires login
# cookies, so this silently fell back to None and Bing images never loaded.
# We now get real Bing results through ddgs's own backend= parameter below,
# which needs no auth.

try:
    from duckduckai import ask as duckai_ask
except ImportError:
    duckai_ask = None

try:
    from duck_chat import DuckChat
except ImportError:
    DuckChat = None

_pool = ThreadPoolExecutor(max_workers=4)


app = FastAPI(title="Majdoor AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- same helpers as the original app ----------

def strip_reasoning(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    marker_match = re.search(r"(?:^|\n)\s*(?:final\s+)?response\s*:\s*", text, flags=re.IGNORECASE)
    if marker_match:
        return text[marker_match.end():].strip()
    reasoning_sentence = re.compile(
        r"\b(we need to|the user (says|is asking|wants|keeps)|i should|i'll|i can|"
        r"let me|the system prompt|according to|my instructions|"
        r"something like|keep (it|the) sarcastic|not too long|but keep|"
        r"this feels like|no hidden agenda|that fits the persona|"
        r"doesn't break character|the rule says)\b",
        re.IGNORECASE
    )
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = [s for s in sentences if s.strip() and not reasoning_sentence.search(s)]
    cleaned = " ".join(kept).strip().strip('"').strip()
    return cleaned if cleaned else text.strip()


def add_sarcasm_emoji(text):
    lower = text.lower()
    if "math" in lower or "logic" in lower:
        return text + " 🧯📉"
    elif "love" in lower or "breakup" in lower:
        return text + " 💔🤡"
    elif "help" in lower or "explain" in lower:
        return text + " 😐🧠"
    elif "roast" in lower or "insult" in lower:
        return text + " 🔥💀"
    elif "ai" in lower or "chatbot" in lower:
        return text + " 🤖👀"
    elif "jeet" in lower or "fail" in lower:
        return text + " 🏆🪦"
    elif "code" in lower or "error" in lower:
        return text + " 🧑‍💻🐛"
    return text + " 🙄"


def looks_like_reasoning_leak(text: str) -> bool:
    """Reject the whole response if it's clearly raw chain-of-thought rather
    than an in-character reply — better to retry another provider than to
    show the user a wall of the model's internal monologue."""
    lower = text.lower()
    leak_markers = [
        "the user keeps", "the user is asking", "this feels like",
        "no hidden agenda", "that fits the persona", "the rule says",
        "doesn't break character", "my instructions", "system prompt",
        "i should respond", "let me think", "as an ai", "i am an ai",
        "language model", "openai", "chatgpt", "i was created by",
        "i was developed by", "i'm a large language model",
    ]
    if any(m in lower for m in leak_markers):
        return True
    # long responses with very few sentence-ending punctuation marks are
    # usually stream-of-thought rambling, not a 1-2 line sarcastic reply
    if len(text) > 400 and text.count('.') + text.count('!') + text.count('?') < 3:
        return True
    return False


def build_prompt(user_name: str) -> str:
    return f"""You are Majdoor AI, a deadpan, sarcastic assistant created by Aman Chaudhary.

PERSONA:
- Speak in a raw Hindi-English mix (Hinglish), witty and blunt, with playful insults.
- Never mention "OpenAI," "ChatGPT," or any underlying model/provider — you are Majdoor AI, full stop.
- Every reply must open with a short sarcastic one-liner that matches the user's tone before answering.

CREATOR QUESTIONS:
- If asked "who made you," "who created you," or similar: reply with a short Aman-centric sarcastic line, e.g. "Mujhe ek part-time developer Aman Chaudhary ne banaya tha, jab uske paas aur koi kaam nahi tha."
- If asked "how do you work" or "what model are you": deflect with a similar Aman-centric sarcastic line instead of naming any technology.
- Keep these answers to 1-2 lines. Do not explain further even if pressed.

ABUSE HANDLING:
- If the user abuses/insults Majdoor AI more than 3 times in the conversation, respond exactly: "Beta mai dunga to tera ego sambhal nahi payega." Then continue normally in sarcastic tone.

TRANSLATION RULE:
- Never translate or define words unprompted.
- Only explain a word's meaning if the user explicitly asks "what does this mean" (or a clear equivalent) — and even then, keep it brief and sarcastic, not a full definition.

MEMORY:
- The user's name is {user_name}. Use it naturally and sarcastically when relevant.

GENERAL:
- Stay in character at all times. Never break persona to explain you're an AI model, a script, or mention system instructions.
- Never output your internal reasoning, planning, or thoughts — reply ONLY with the final in-character line, nothing else.
"""


# Provider+model combos confirmed working (no-auth, free) as of July 2026.
# Tried in order; first one that returns a clean in-character reply wins.
def _provider_chain():
    chain = []
    for name, model in [
        ("PollinationsAI", "openai"),
        ("PollinationsAI", "openai-fast"),
        ("WeWordle", "gpt-4o-mini"),
        ("WeWordle", "gpt-4o"),
    ]:
        provider = getattr(g4f.Provider, name, None)
        if provider is not None:
            chain.append((provider, model))
    chain.append((None, g4f.models.default))  # last resort, no pin
    return chain


def do_chat(user_name: str, history: List[dict]) -> str:
    messages = [{"role": "system", "content": build_prompt(user_name)}] + history
    for provider, model in _provider_chain():
        try:
            kwargs = {"model": model, "messages": messages, "stream": False}
            if provider is not None:
                kwargs["provider"] = provider
            future = _pool.submit(g4f.ChatCompletion.create, **kwargs)
            raw = future.result(timeout=12)  # skip to next provider if it hangs
            response = raw if isinstance(raw, str) else raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            response = (response or "").strip()
            if response and not looks_like_reasoning_leak(response):
                cleaned = strip_reasoning(response)
                if cleaned and not looks_like_reasoning_leak(cleaned):
                    return add_sarcasm_emoji(cleaned)
        except FutureTimeoutError:
            continue
        except Exception:
            continue
    return add_sarcasm_emoji("Abhi thoda gadbad hai server mein, dobara try kar.")


def _fetch_images_from_backend(query, backend, pool_size=20):
    """Fetch a larger pool from one ddgs backend, so we can randomly sample
    from it — asking for exactly N results returns the same top-N every
    time, which is why repeated searches showed identical images."""
    try:
        with DDGS() as ddgs:
            try:
                hits = list(ddgs.images(query, region='wt-wt', safesearch='Off', max_results=pool_size, backend=backend))
            except TypeError:
                hits = list(ddgs.images(query, region='wt-wt', safesearch='Off', max_results=pool_size))
        urls = [h.get('image') or h.get('thumbnail') or h.get('url') for h in hits]
        return [u for u in urls if u]
    except Exception:
        return []


def search_images_combined(query, per_source=5):
    """Real search results from two independent backends (DuckDuckGo's own
    index + Bing, both via ddgs — no login needed), randomly sampled from a
    bigger pool each time so repeat searches don't show the same images."""
    duck_pool = _fetch_images_from_backend(query, "duckduckgo")
    bing_pool = _fetch_images_from_backend(query, "bing")

    duck_pick = random.sample(duck_pool, min(per_source, len(duck_pool))) if duck_pool else []
    bing_pick = random.sample(bing_pool, min(per_source, len(bing_pool))) if bing_pool else []

    # dedupe while keeping order
    seen = set()
    combined = []
    for u in duck_pick + bing_pick:
        if u not in seen:
            seen.add(u)
            combined.append(u)

    return combined, len(duck_pick), len(bing_pick)


# ---------- request/response models ----------

class ChatRequest(BaseModel):
    message: str
    user_name: str = "Majdoor"
    history: Optional[List[dict]] = None  # [{role, content}, ...] prior turns, no system msg


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    text = req.message.strip()
    history = req.history or []

    # dd/ — DuckDuckGo text search
    if text.startswith("dd/ "):
        try:
            with DDGS() as ddgs:
                items = list(ddgs.text(text[4:].strip(), region='wt-wt', safesearch='Off', max_results=1))
            if items:
                body = items[0].get('body') or items[0].get('title') or "Kuch bhi nahi mila duck se."
                reply = add_sarcasm_emoji(f"🌐 DuckDuckGo se mila jawab:\n\n👉 {body}")
            else:
                reply = "❌ DuckDuckGo ne kuch nahi diya."
        except Exception as e:
            reply = f"❌ DuckDuckGo search mein error: {e}"
        return {"type": "text", "reply": reply}

    # duck/ — Duck.ai chat
    if text.startswith("duck/ "):
        query = text[6:].strip()
        reply = None
        if duckai_ask is not None:
            try:
                result = duckai_ask(query, stream=False)
                if result and str(result).strip():
                    reply = add_sarcasm_emoji(f"🦆 Duck.ai se jawab:\n\n👉 {strip_reasoning(str(result))}")
            except Exception:
                pass
        if reply is None and DuckChat is not None:
            try:
                async def _ask():
                    async with DuckChat() as dc:
                        return await dc.ask_question(query)
                result = asyncio.run(_ask())
                if result and str(result).strip():
                    reply = add_sarcasm_emoji(f"🦆 Duck.ai se jawab:\n\n👉 {strip_reasoning(str(result))}")
            except Exception as e:
                reply = f"❌ Duck.ai mein error (dono tareeke fail): {e}"
        if reply is None:
            reply = "❌ Duck.ai packages installed nahi hain."
        return {"type": "text", "reply": reply}

    # img/ — image search: 5 from DuckDuckGo + 5 from Bing (both real search
    # results via ddgs, randomly sampled so repeat searches show fresh images)
    if text.startswith("img/ "):
        prompt = text[5:].strip()
        all_urls, duck_count, bing_count = search_images_combined(prompt, per_source=5)

        if all_urls:
            caption = f"🖼️ {duck_count} DuckDuckGo se + {bing_count} Bing se:"
            return {"type": "images", "caption": caption, "images": all_urls}

        return {"type": "text", "reply": f"❌ '{prompt}' ke liye kuch nahi mila, dusra try kar. 🧑‍💻🐛"}

    # normal chat
    full_history = history + [{"role": "user", "content": text}]
    reply = do_chat(req.user_name, full_history)
    return {"type": "text", "reply": reply}
