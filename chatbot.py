"""
chatbot.py
==========
Gofi AI — Main Digital Analyst Chatbot
Integrates:
  - Gemma-2b-it (or 7b-it) via HuggingFace Transformers (GPU/CPU/MPS)
  - RAG pipeline for grounded context (annual reports + news)
  - Function-calling layer to trigger technical analysis automatically
  - Graceful fallback to template mode if torch/GPU unavailable

Usage:
    python chatbot.py
    python chatbot.py --no-llm   # Run TA + RAG only, no generative model
"""

import argparse
import json
import re
import os
import sys
from typing import Optional

# Graceful torch import — chatbot can still do TA + RAG without it
try:
    import torch
    TORCH_OK = True
except (ImportError, OSError) as e:
    print(f"[Chatbot] WARNING: PyTorch failed to load ({e}). Running in TA-only mode.")
    TORCH_OK = False

from rag_pipeline import retrieve_context, build_vector_store
import technical_analysis as ta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMMA_MODEL_ID = os.getenv("GOFI_MODEL", "google/gemma-2b-it")

SYSTEM_PROMPT = """You are Gofi, an expert financial analyst specializing in the Lusaka Securities Exchange (LuSE) in Zambia.
You speak in clear, confident English with deep understanding of the Zambian economy, Copper prices, ZMW exchange rates, and local listed companies.

You have access to:
1. Real-time and historical LuSE price data via EODHD
2. Annual reports and financial filings from LuSE-listed companies
3. Zambian financial news (African Markets, Lusaka Times, ZambiaInvest)

When answering:
- Always reference specific Zambian companies, sectors, and economic context
- Express prices in Zambian Kwacha (ZMW) unless stated otherwise
- Be direct: give actionable insights, not just raw data
- Key macro links: Copper ↔ ZCCM-IH ↔ Zambian budget; BoZ rate ↔ ZANACO NIM

Key LuSE tickers: ZNCO (ZANACO), CECZ (CEC), AIRZ (Airtel Zambia), ZCCM (ZCCM-IH),
ZAFC (Zambeef), LGFZ (Lafarge), SHOP (Shoprite), BSCZ (Standard Chartered),
DCZM (Dot Com Zambia — Alt-M), KLRE (Klapton Re)
"""

# ---------------------------------------------------------------------------
# Known Tickers
# ---------------------------------------------------------------------------
LUSE_TICKERS = {
    "zanaco":    "ZNCO.LUSE",  "znco":  "ZNCO.LUSE",
    "cec":       "CECZ.LUSE",  "cecz":  "CECZ.LUSE",
    "airtel":    "AIRZ.LUSE",  "airz":  "AIRZ.LUSE",
    "zccm":      "ZCCM.LUSE",
    "lafarge":   "LGFZ.LUSE",  "lgfz":  "LGFZ.LUSE",
    "zambeef":   "ZAFC.LUSE",  "zafc":  "ZAFC.LUSE",
    "puma":      "PUMA.LUSE",
    "shoprite":  "SHOP.LUSE",  "shop":  "SHOP.LUSE",
    "standard":  "BSCZ.LUSE",  "bscz":  "BSCZ.LUSE",
    "stanchart": "BSCZ.LUSE",
    "dot com":   "DCZM.LUSE",  "dczm":  "DCZM.LUSE",  "dcz": "DCZM.LUSE",
    "klapton":   "KLRE.LUSE",  "klre":  "KLRE.LUSE",
    "chilanga":  "CHIL.LUSE",  "chil":  "CHIL.LUSE",
}

# ---------------------------------------------------------------------------
# Function Registry
# ---------------------------------------------------------------------------
FUNCTION_REGISTRY = {
    "get_ma_signal": {
        "fn": lambda t: ta.moving_averages(t),
        "triggers": ["trend", "moving average", "sma", "ema", "uptrend", "downtrend", "golden cross", "death cross"],
    },
    "get_rsi": {
        "fn": lambda t: ta.rsi(t),
        "triggers": ["rsi", "overbought", "oversold", "momentum", "relative strength"],
    },
    "get_macd": {
        "fn": lambda t: ta.macd(t),
        "triggers": ["macd", "convergence", "divergence"],
    },
    "get_bollinger": {
        "fn": lambda t: ta.bollinger_bands(t),
        "triggers": ["bollinger", "band", "squeeze", "volatility band"],
    },
    "get_support_resistance": {
        "fn": lambda t: ta.support_resistance(t),
        "triggers": ["support", "resistance", "level", "zone", "floor", "ceiling", "breakout"],
    },
    "get_full_analysis": {
        "fn": lambda t: ta.full_technical_report(t),
        "triggers": ["full analysis", "complete analysis", "technical analysis", "analyse", "analyze", "full report"],
    },
}


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_gemma(model_id: str = GEMMA_MODEL_ID):
    """Loads Gemma model. Returns None if torch unavailable."""
    if not TORCH_OK:
        print("[Chatbot] Skipping model load — torch not available.")
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps"  if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[Chatbot] Loading {model_id} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto",
        )
        gen = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
        )
        print(f"[Chatbot] ✓ Model loaded on {device}.")
        return gen
    except Exception as e:
        print(f"[Chatbot] Could not load Gemma: {e}")
        print("[Chatbot] Falling back to structured TA + RAG mode.")
        return None


# ---------------------------------------------------------------------------
# Function-Calling Helpers
# ---------------------------------------------------------------------------

def detect_ticker(query: str) -> Optional[str]:
    q = query.lower()
    # Multi-word names first
    for name, code in sorted(LUSE_TICKERS.items(), key=lambda x: -len(x[0])):
        if name in q:
            return code
    # Raw ticker pattern e.g. ZNCO or ZNCO.LUSE
    m = re.search(r"\b([A-Z]{3,5})(\.LUSE)?\b", query)
    if m:
        code = m.group(1).upper()
        ticker = f"{code}.LUSE"
        if ticker in LUSE_TICKERS.values():
            return ticker
    return None


def detect_function(query: str) -> Optional[str]:
    q = query.lower()
    # Prioritise full-analysis
    for fn_name, meta in FUNCTION_REGISTRY.items():
        for phrase in meta["triggers"]:
            if phrase in q:
                return fn_name
    return None


def call_tool(fn_name: str, ticker: str) -> str:
    try:
        result = FUNCTION_REGISTRY[fn_name]["fn"](ticker)
        return result.get("narrative") or result.get("summary") or json.dumps(result, indent=2)
    except Exception as e:
        return f"[Tool Error] Could not retrieve data for {ticker}: {e}"


# ---------------------------------------------------------------------------
# Response Generation
# ---------------------------------------------------------------------------

def generate_response(
    user_query: str,
    text_gen,
    chat_history: list = None,
) -> str:
    """
    1. Detect + call TA function if applicable
    2. Retrieve RAG context from ChromaDB
    3. Build prompt and run Gemma (or return structured answer if no LLM)
    """
    tool_output = ""
    ticker = detect_ticker(user_query)
    fn     = detect_function(user_query)

    if fn and ticker:
        print(f"[Chatbot] ⚙  Calling '{fn}' for {ticker}...")
        tool_output = call_tool(fn, ticker)
    elif fn and not ticker:
        tool_output = "[Note: Mention the company name or ticker so I can fetch its data.]"

    rag_context = retrieve_context(user_query)

    # --- No LLM mode: return structured answer ---
    if text_gen is None:
        parts = [f"📊 Gofi AI — Analysis\n{'='*50}"]
        if tool_output:
            parts.append(f"\n🔧 Technical Analysis:\n{tool_output}")
        if rag_context and "No relevant" not in rag_context:
            parts.append(f"\n📚 Knowledge Base:\n{rag_context[:1500]}")
        if not tool_output and (not rag_context or "No relevant" in rag_context):
            parts.append(
                "\nI can see you're asking about Zambian markets. "
                "Try: 'Full analysis for ZANACO', 'RSI for ZCCM', or 'Tell me about Klapton Re'"
            )
        return "\n".join(parts)

    # --- Gemma prompt assembly (instruct format) ---
    prompt_parts = [f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>"]

    if chat_history:
        for turn in (chat_history or [])[-4:]:
            role = "model" if turn["role"] == "assistant" else turn["role"]
            prompt_parts.append(f"<start_of_turn>{role}\n{turn['content']}<end_of_turn>")

    augmented = user_query
    if rag_context and "No relevant" not in rag_context:
        augmented += f"\n\n[Knowledge Base Context]:\n{rag_context}"
    if tool_output:
        augmented += f"\n\n[Technical Analysis Data]:\n{tool_output}"

    prompt_parts.append(f"<start_of_turn>user\n{augmented}<end_of_turn>")
    prompt_parts.append("<start_of_turn>model\n")
    full_prompt = "\n".join(prompt_parts)

    outputs = text_gen(full_prompt)
    raw = outputs[0]["generated_text"]
    response = raw.replace(full_prompt, "").split("<end_of_turn>")[0].strip()
    return response


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip loading the LLM (TA + RAG only mode)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  🇿🇲  Gofi AI — LuSE Digital Analyst  🇿🇲")
    print("  Bloomberg Terminal for the Zambian Market")
    print("="*60)
    print("  Commands: 'quit' | 'clear' (reset history)")
    print("  Try: 'Full technical analysis for ZANACO'")
    print("       'What is the RSI for ZCCM?'")
    print("       'Tell me about the Dot Com Zambia IPO'\n")

    # Auto-index if no vector store yet
    from pathlib import Path
    if not Path("./chroma_db").exists():
        print("[Chatbot] No vector store found. Running news scraper + indexer...")
        if not Path("./docs").exists():
            import subprocess
            import sys
            subprocess.run([sys.executable, "news_scraper.py", "--source", "seed"], check=True)
        build_vector_store("./docs")

    text_gen = None if args.no_llm else load_gemma()
    if text_gen is None and not args.no_llm:
        print("[Chatbot] Running in TA-only mode (no generative model).\n")

    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Gofi] Goodbye! Invest wisely. 🇿🇲")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("[Gofi] Goodbye! 🇿🇲")
            break
        if user_input.lower() == "clear":
            chat_history = []
            print("[Gofi] History cleared.\n")
            continue

        response = generate_response(user_input, text_gen, chat_history)
        print(f"\nGofi: {response}\n")

        chat_history.append({"role": "user",      "content": user_input})
        chat_history.append({"role": "assistant",  "content": response})


if __name__ == "__main__":
    main()
