# Gofi AI

Gofi AI is a Digital Analyst Chatbot and Financial Terminal built for the Lusaka Securities Exchange (LuSE) in Zambia. It provides technical analysis, real-time data, and RAG-based context using local financial filings and news.

## Features
- **Technical Analysis**: Moving Averages, RSI, MACD, Bollinger Bands, Support/Resistance
- **RAG Pipeline**: Local vector search of scraped LuSE-related news and documents using a pure Python TF-IDF implementation.
- **Digital Analyst**: LLM-powered conversational interface, capable of leveraging local models (like Gemma).
- **News Scraper**: Automated retrieval of financial news from African Markets, Zambia Daily Mail, and ZambiaInvest.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file and add your EODHD API Key:
   ```
   EODHD_API_KEY=your_key_here
   ```

3. Run the Chatbot:
   ```bash
   python chatbot.py
   ```

## Modules
- `chatbot.py`: Main conversational interface.
- `api_Data.py`: Data retrieval via EODHD.
- `technical_analysis.py`: Technical indicator functions.
- `rag_pipeline.py`: Pure-Python vector store for RAG.
- `news_scraper.py`: Financial news scraper.
- `finetune_gemma.py`: Q-LoRA script for fine-tuning models on Zambian data.
