"""
Majdoor AI backend — same persona, same triggers, same sarcasm as the
original Streamlit app (majdoor__2___8_.py), exposed as a stateless API
so the Android/PWA app can call it.

Deploy free on Render.com or Hugging Face Spaces.
"""

import os
import re
import time
import asyncio
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

try:
    from g4f.Provider import bing
except ImportError:
    bing = None

try:
    from duckduckai import ask as duckai_ask
except ImportError:
    duckai_ask = None

try:
    from duck_chat import DuckChat
except ImportError:
    DuckChat = None


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
        r"\b(we need to|the user (says|is asking|wants)|i should|i'll|i can|"
        r"let me|the system prompt|according to|my instructions|"
        r"something like|keep (it|the) sarcastic|not too long|but keep)\b",
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
"""


MODEL_FALLBACK_CHAIN = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", g4f.models.default]

def do_chat(user_name: str, history: List[dict]) -> str:
    messages = [{"role": "system", "content": build_prompt(user_name)}] + history
    last_err = None
    for model in MODEL_FALLBACK_CHAIN:
        try:
            raw = g4f.ChatCompletion.create(model=model, messages=messages, stream=False)
            response = raw if isinstance(raw, str) else raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            response = (response or "").strip()
            # if the model ignored the persona and mentions itself/OpenAI, skip to next model
            if response and not re.search(r"\b(openai|chatgpt|gpt-4|gpt-3|language model|i am an ai)\b", response, re.IGNORECASE):
                return add_sarcasm_emoji(strip_reasoning(response))
        except Exception as e:
            last_err = e
            continue
    return add_sarcasm_emoji("Abhi thoda gadbad hai server mein, dobara try kar.")


def search_image_ddg(query, retries=2, delay=2, count=7):
    backends_to_try = ["auto", "bing"]
    last_error = None
    for backend in backends_to_try:
        for attempt in range(retries):
            try:
                with DDGS() as ddgs:
                    try:
                        hits = list(ddgs.images(query, region='wt-wt', safesearch='Off', max_results=count, backend=backend))
                    except TypeError:
                        hits = list(ddgs.images(query, region='wt-wt', safesearch='Off', max_results=count))
                if hits:
                    urls = [h.get('image') or h.get('thumbnail') or h.get('url') for h in hits]
                    urls = [u for u in urls if u]
                    if urls:
                        return urls, None
                break
            except Exception as e:
                last_error = e
                if "403" in str(e) or "ratelimit" in str(e).lower():
                    time.sleep(delay * (attempt + 1))
                    continue
                break
    return [], f"Duck image search error: {last_error}"


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

    # img/ — image search: 5 from DuckDuckGo + 5 from Bing, combined
    if text.startswith("img/ "):
        prompt = text[5:].strip()
        all_urls = []"""
Majdoor AI backend — same persona, same triggers, same sarcasm as the
original Streamlit app (majdoor__2___8_.py), exposed as a stateless API
so the Android/PWA app can call it.

Deploy free on Render.com or Hugging Face Spaces.
"""

import os
import re
import time
import asyncio
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

try:
    from g4f.Provider import bing
except ImportError:
    bing = None

try:
    from duckduckai import ask as duckai_ask
except ImportError:
    duckai_ask = None

try:
    from duck_chat import DuckChat
except ImportError:
    DuckChat = None


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
        r"\b(we need to|the user (says|is asking|wants)|i should|i'll|i can|"
        r"let me|the system prompt|according to|my instructions|"
        r"something like|keep (it|the) sarcastic|not too long|but keep)\b",
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
"""


MODEL_FALLBACK_CHAIN = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", g4f.models.default]

def do_chat(user_name: str, history: List[dict]) -> str:
    messages = [{"role": "system", "content": build_prompt(user_name)}] + history
    last_err = None
    for model in MODEL_FALLBACK_CHAIN:
        try:
            raw = g4f.ChatCompletion.create(model=model, messages=messages, stream=False)
            response = raw if isinstance(raw, str) else raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            response = (response or "").strip()
            # if the model ignored the persona and mentions itself/OpenAI, skip to next model
            if response and not re.search(r"\b(openai|chatgpt|gpt-4|gpt-3|language model|i am an ai)\b", response, re.IGNORECASE):
                return add_sarcasm_emoji(strip_reasoning(response))
        except Exception as e:
            last_err = e
            continue
    return add_sarcasm_emoji("Abhi thoda gadbad hai server mein, dobara try kar.")


def search_image_ddg(query, retries=2, delay=2, count=7):
    backends_to_try = ["auto", "bing"]
    last_error = None
    for backend in backends_to_try:
        for attempt in range(retries):
            try:
                with DDGS() as ddgs:
                    try:
                        hits = list(ddgs.images(query, region='wt-wt', safesearch='Off', max_results=count, backend=backend))
                    except TypeError:
                        hits = list(ddgs.images(query, region='wt-wt', safesearch='Off', max_results=count))
                if hits:
                    urls = [h.get('image') or h.get('thumbnail') or h.get('url') for h in hits]
                    urls = [u for u in urls if u]
                    if urls:
                        return urls, None
                break
            except Exception as e:
                last_error = e
                if "403" in str(e) or "ratelimit" in str(e).lower():
                    time.sleep(delay * (attempt + 1))
                    continue
                break
    return [], f"Duck image search error: {last_error}"


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

    # img/ — image search: 5 from DuckDuckGo + 5 from Bing, combined
    if text.startswith("img/ "):
        prompt = text[5:].strip()
        all_urls = []

        duck_urls, duck_error = search_image_ddg(prompt, count=5)
        all_urls.extend(duck_urls)

        bing_urls = []
        if bing:
            for _ in range(5):
                try:
                    imgs = bing.create_images(prompt)
                    if imgs:
                        pick = imgs[0] if isinstance(imgs, list) else imgs
                        if pick and pick not in bing_urls and pick not in duck_urls:
                            bing_urls.append(pick)
                except Exception:
                    break
        all_urls.extend(bing_urls)

        if all_urls:
            caption = f"🖼️ {len(duck_urls)} DuckDuckGo se + {len(bing_urls)} Bing se:"
            return {"type": "images", "caption": caption, "images": all_urls}

        return {"type": "text", "reply": f"❌ {duck_error} 🧑‍💻🐛"}

    # normal chat
    full_history = history + [{"role": "user", "content": text}]
    reply = do_chat(req.user_name, full_history)
    return {"type": "text", "reply": reply}


        duck_urls, duck_error = search_image_ddg(prompt, count=5)
        all_urls.extend(duck_urls)

        bing_urls = []
        if bing:
            for _ in range(5):
                try:
                    imgs = bing.create_images(prompt)
                    if imgs:
                        pick = imgs[0] if isinstance(imgs, list) else imgs
                        if pick and pick not in bing_urls and pick not in duck_urls:
                            bing_urls.append(pick)
                except Exception:
                    break
        all_urls.extend(bing_urls)

        if all_urls:
            caption = f"🖼️ {len(duck_urls)} DuckDuckGo se + {len(bing_urls)} Bing se:"
            return {"type": "images", "caption": caption, "images": all_urls}

        return {"type": "text", "reply": f"❌ {duck_error} 🧑‍💻🐛"}

    # normal chat
    full_history = history + [{"role": "user", "content": text}]
    reply = do_chat(req.user_name, full_history)
    return {"type": "text", "reply": reply}
    
