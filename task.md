# Gofi AI - Build Checklist

## Phase 1: Core Modules ✅
- [x] api_Data.py (EODHD source)
- [x] rag_pipeline.py
- [x] technical_analysis.py
- [x] chatbot.py
- [x] finetune_gemma.py
- [x] data/training_data.jsonl

## Phase 2: Fix Dependencies & Runtime Bugs
- [/] Fix LangChain imports (langchain → langchain_community)
- [/] Fix PyTorch broken conda install → reinstall
- [/] Fix all other import issues in chatbot.py
- [ ] Create a lightweight chatbot mode (no torch needed for initial run)

## Phase 3: Knowledge Base Population
- [ ] Scrape LuSE official data / African Markets
- [ ] Scrape BoZ monetary policy data
- [ ] Scrape Lusaka Times / Zambia Daily Mail news
- [ ] Save as .txt files in docs/news/
- [ ] Index into ChromaDB

## Phase 4: End-to-End Run
- [ ] RAG index runs successfully
- [ ] Chatbot responds to queries
- [ ] TA functions work with live EODHD data
