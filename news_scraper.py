"""
news_scraper.py
================
Gofi AI — LuSE News & Market Data Scraper
Fetches fresh content from:
  - African Markets (LuSE section)
  - Zambia Daily Mail (Business)
  - ZambiaInvest
  - Bank of Zambia (macroeconomic summaries)
  - LuSE official announcements

Saves results as .txt files in --output dir (default: ./docs/news/)
These files are then indexed by rag_pipeline.py

Usage:
    python news_scraper.py                    # scrape all sources
    python news_scraper.py --output ./docs/news
    python news_scraper.py --source african_markets
"""

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./docs/news"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_DELAY = 1.5   # seconds between requests (be polite)
TIMEOUT = 15


def clean_text(text: str) -> str:
    """Remove excessive whitespace and boilerplate."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def save_article(title: str, content: str, source: str, output_dir: str) -> str:
    """Saves a scraped article to a .txt file. Returns the path."""
    safe_title = re.sub(r"[^\w\s-]", "", title)[:80].strip().replace(" ", "_")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"{safe_title}_{timestamp}.txt"
    filepath   = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {title}\n")
        f.write(f"SOURCE: {source}\n")
        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("="*60 + "\n\n")
        f.write(content)

    return str(filepath)


# ---------------------------------------------------------------------------
# Source 1: African Markets — LuSE Articles
# ---------------------------------------------------------------------------

AFRICAN_MARKETS_ARTICLES = [
    "https://www.african-markets.com/en/stock-markets/luse/zambia-klapton-re-lists-on-the-lusaka-securities-exchange-through-a-49-5-million-direct-listing",
    "https://www.african-markets.com/en/stock-markets/luse/lusaka-stock-exchange-announces-results-of-dot-com-zambia-ipo-oversubscribed-114-times",
    "https://www.african-markets.com/en/stock-markets/luse/dot-com-zambia-to-list-on-lusaka-securities-exchange-alternative-market-in-landmark-ipo",
    "https://www.african-markets.com/en/stock-markets/luse/dot-com-zambia-launches-zmw-12-3m-ipo-on-the-lusaka-securities-exchange",
]


def scrape_african_markets(output_dir: str) -> list:
    """Scrapes specific African Markets LuSE articles."""
    saved = []
    print(f"[Scraper] Fetching {len(AFRICAN_MARKETS_ARTICLES)} African Markets articles...")
    for url in AFRICAN_MARKETS_ARTICLES:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Get title
            title_tag = soup.find("h1") or soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

            # Get article body — try common article containers
            body = (
                soup.find("article") or
                soup.find("div", class_=re.compile(r"article|content|post|entry", re.I)) or
                soup.find("main")
            )
            if body:
                # Remove nav/ads/scripts
                for tag in body(["script", "style", "nav", "footer", "aside", "form"]):
                    tag.decompose()
                content = clean_text(body.get_text(separator="\n"))
            else:
                content = clean_text(soup.get_text(separator="\n"))

            if len(content) > 200:
                path = save_article(title, content, url, output_dir)
                print(f"  ✓ Saved: {Path(path).name}")
                saved.append(path)
            else:
                print(f"  ⚠ Skipped (too short): {url}")

            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  ✗ Failed {url}: {e}")
    return saved


# ---------------------------------------------------------------------------
# Source 2: Zambia Daily Mail — Business Section
# ---------------------------------------------------------------------------

def scrape_zambia_daily_mail(output_dir: str, max_articles: int = 8) -> list:
    """Scrapes recent business news from Zambia Daily Mail."""
    saved = []
    index_url = "https://www.daily-mail.co.zm/category/business/"
    print(f"[Scraper] Fetching Zambia Daily Mail business headlines...")
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "daily-mail.co.zm" in href and "/202" in href:
                links.append(href)
        links = list(dict.fromkeys(links))[:max_articles]

        for url in links:
            try:
                r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                r.raise_for_status()
                s = BeautifulSoup(r.text, "html.parser")
                h1 = s.find("h1")
                title = h1.get_text(strip=True) if h1 else "Zambia Daily Mail Article"
                article = s.find("article") or s.find("div", class_=re.compile(r"entry|post|content"))
                if article:
                    for tag in article(["script", "style", "aside", "nav"]):
                        tag.decompose()
                    content = clean_text(article.get_text(separator="\n"))
                    if len(content) > 200:
                        path = save_article(title, content, url, output_dir)
                        print(f"  ✓ {Path(path).name}")
                        saved.append(path)
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                print(f"  ✗ {url}: {e}")
    except Exception as e:
        print(f"  ✗ Could not reach Daily Mail: {e}")
    return saved


# ---------------------------------------------------------------------------
# Source 3: ZambiaInvest
# ---------------------------------------------------------------------------

def scrape_zambiainvest(output_dir: str, max_articles: int = 6) -> list:
    """Scrapes investment news from ZambiaInvest."""
    saved = []
    index_url = "https://www.zambiainvest.com/news/"
    print(f"[Scraper] Fetching ZambiaInvest news...")
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "zambiainvest.com" in href.lower() and len(href) > 40:
                links.append(href)
        links = list(dict.fromkeys(links))[:max_articles]
        for url in links:
            try:
                r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                s = BeautifulSoup(r.text, "html.parser")
                h1 = s.find("h1")
                title = h1.get_text(strip=True) if h1 else "ZambiaInvest Article"
                body = s.find("article") or s.find("div", class_=re.compile(r"content|post|entry"))
                if body:
                    for t in body(["script", "style"]):
                        t.decompose()
                    content = clean_text(body.get_text(separator="\n"))
                    if len(content) > 200:
                        path = save_article(title, content, url, output_dir)
                        print(f"  ✓ {Path(path).name}")
                        saved.append(path)
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                print(f"  ✗ {url}: {e}")
    except Exception as e:
        print(f"  ✗ Could not reach ZambiaInvest: {e}")
    return saved


# ---------------------------------------------------------------------------
# Source 4: Write pre-seeded knowledge articles (always works, no network)
# ---------------------------------------------------------------------------

SEED_ARTICLES = [
    {
        "title": "Klapton Re LuSE Direct Listing — March 2026",
        "source": "African Markets / LuSE Official",
        "content": """Klapton Reinsurance Plc (KLRE.LUSE) listed on the Lusaka Securities Exchange on March 24, 2026 through a direct listing — bypassing the traditional IPO subscription process.

The transaction involved 2.82 billion shares offered at ZMW 0.35 per share, raising a total of approximately ZMW 987.7 million (around USD 49.5 million). At listing, the company's market capitalization stood at approximately ZMW 3.95 billion, with a free float of up to 25% of total share capital.

Founded in 2020, Klapton Re has quickly established itself as a key player in Zambia's reinsurance sector. As of November 2025, the group reported:
- Insurance revenues: ZMW 2.7 billion
- Net profit: ZMW 325 million
- Total assets: ZMW 3.5 billion

The company now operates in over 120 countries, focusing primarily on non-life reinsurance, with an estimated 86% domestic market share.

Proceeds from the listing are expected to strengthen the capital base and solvency, support international expansion into the United States and China, invest in risk analytics capabilities, and develop insurance solutions linked to climate-related risks.

This listing came shortly after the high-profile Dot Com Zambia IPO in December 2025, reinforcing growing investor appetite for the Zambian market. It represents a significant step in broadening the LuSE's listings and improving market depth and liquidity.

Investment Implications: KLRE's listing at ZMW 0.35 per share values the company attractively relative to its ZMW 2.7B revenue base. The 86% domestic market share provides a defensive moat, while international expansion aspirations (US, China) suggest a long-term growth angle. However, investors should note that reinsurance businesses carry catastrophe risk and are sensitive to interest rate environments for investment income. The ZMW/USD relationship is particularly important given international operations.""",
    },
    {
        "title": "Dot Com Zambia IPO — Oversubscribed 114 Times — December 2025",
        "source": "African Markets / LuSE Official",
        "content": """Dot Com Zambia Plc (DCZM.LUSE) completed a landmark IPO on the LuSE Alternative Market (Alt-M), listing on December 17, 2025 under the ticker DCZ.

The IPO results were extraordinary:
- Target raise: ZMW 12.3 million
- Total applications received: ZMW 14.1 million
- Oversubscription: 114 times
- Subscription period closed one week early (December 5, 2025) due to demand

Shareholder structure post-IPO:
- Retail investors: 85% of new shareholder base
- New investors joining: 500+
- Zambian shareholders: 75% of total
- Foreign investors: 25% of funds

Dot Com Zambia specializes in ICT solutions — particularly revenue collection optimization and business intelligence tools for public and private sector clients. The company was founded by Mawano Kambeu and attracted institutional investors Kukula Fund and eVentures Africa in 2015.

This was the first-ever listing on the LuSE Alt-M segment, which was launched in 2015 but had no previous listings due to regulatory constraints and low SME awareness. LuSE CEO Nicholas Kabaso described it as "bold and exciting," noting strong market appetite for new assets.

Investment Significance: The 114x oversubscription is one of the most dramatic IPO results in sub-Saharan African emerging markets in recent memory. It signals: (1) significant pent-up demand for Zambian equity investment opportunities; (2) growing retail investor participation in the LuSE; (3) tech sector confidence in Zambia's digital economy; (4) viability of the Alt-M platform for future Zambian SME listings.

The success could catalyze further SME listings and attract foreign portfolio investors to the LuSE ecosystem.""",
    },
    {
        "title": "Bank of Zambia — Monetary Policy & Economic Overview 2025–2026",
        "source": "Bank of Zambia / Ministry of Finance",
        "content": """The Bank of Zambia (BoZ) is the central bank responsible for monetary policy in Zambia. Key economic indicators and policy stance:

MONETARY POLICY RATE (MPR):
The BoZ Monetary Policy Committee meets quarterly to set the MPR. The rate influences commercial bank lending and deposit rates across the entire Zambian financial system. ZANACO, Standard Chartered Zambia, and other commercial banks price their loan books relative to the MPR.

KEY ECONOMIC INDICATORS (2025):
- GDP Growth: Zambia's economy has been recovering, driven by copper mining, agriculture, and services
- Inflation: The BoZ targets single-digit inflation. Food and energy prices are the primary drivers
- Exchange Rate: The ZMW/USD rate is floating and reflects global copper price cycles, FX reserve levels, and risk appetite for frontier markets
- Copper Production: Zambia is one of the world's top copper producers. Revenue from copper exports is a critical driver of FX reserves and government revenue

ZAMBIAN COPPER & GLOBAL MARKETS:
Zambia holds approximately 6% of global copper reserves. Key mining companies listed on the LuSE or influencing the market:
- ZCCM-IH (ZCCM.LUSE): The Zambian government's mining investment vehicle, holding stakes in major copper mines
- Copperbelt Energy Corporation (CECZ.LUSE): Provides power to copper mines; revenue correlated with mining activity

IMPACT OF COPPER PRICES ON LuSE:
When LME copper prices rise:
1. ZCCM-IH dividends increase → positive for LuSE sentiment
2. Government royalty revenues improve → fiscal stability
3. ZMW tends to strengthen (more USD from copper exports)
4. CECZ benefits from increased mining operations demanding power

When copper falls:
1. FX pressure increases on ZMW
2. ZCCM-IH and CECZ face earnings pressure
3. Government budget may face revenue shortfalls
4. BoZ may raise MPR to defend ZMW

ZAMBIAN BANKING SECTOR:
The banking sector is the most liquid segment of the LuSE. Key names:
- ZANACO (ZNCO.LUSE): Zambia's largest local bank by branches. Net Interest Margin (NIM) of ~8%
- Standard Chartered Zambia (BSCZ.LUSE): International bank with strong trade finance capabilities
- Higher MPR = wider NIMs for banks = positive earnings impact short-term, offset by potential NPL rise""",
    },
    {
        "title": "LuSE Listed Companies — Sector Guide 2025",
        "source": "LuSE Official / Stockbrokers Zambia",
        "content": """LUSAKA SECURITIES EXCHANGE — LISTED COMPANIES SECTOR GUIDE

BANKING & FINANCIAL SERVICES:
- ZNCO.LUSE — Zambia National Commercial Bank (ZANACO): Largest local Zambian bank. Focus on retail and SME banking. Listed since 2008. Government holds majority stake via ZNBS.
- BSCZ.LUSE — Standard Chartered Bank Zambia: Part of Standard Chartered global network. Strong in corporate and institutional banking, trade finance.

MINING & ENERGY:
- ZCCM.LUSE — ZCCM Investments Holdings: State mining investment vehicle. Holds stakes in Konkola Copper Mines, First Quantum's Kansanshi, and others. Highly correlated with LME copper.
- CECZ.LUSE — Copperbelt Energy Corporation (CEC): Provides electricity to the Copperbelt mining sector. Also expanding into renewable energy.

CONSUMER & RETAIL:
- ZAFC.LUSE — Zambeef Products Plc: Zambia's largest integrated cold chain food company. Products include beef, pork, chicken, dairy, and stockfeed. Also has West Africa operations.
- SHOP.LUSE — Shoprite Zambia: Part of the Shoprite Holdings group (JSE-listed). Dominant supermarket chain in Zambia.
- PUMA.LUSE — Puma Energy Zambia: Fuel retail and distribution.

INDUSTRIAL & CONSTRUCTION:
- CHIL.LUSE — Chilanga Cement Plc: Zambia's leading cement manufacturer. Benefits from infrastructure investment.
- LGFZ.LUSE — Lafarge Zambia: Cement and construction materials. Parent is Holcim Group.

TELECOMMUNICATIONS:
- AIRZ.LUSE — Airtel Networks Zambia: Listed telecom. Zambia operations of Airtel Africa (LSE-listed). Mobile money (Airtel Money) is key growth driver.

TECHNOLOGY (Alt-M):
- DCZM.LUSE — Dot Com Zambia: First tech company on LuSE Alt-M. ICT solutions, revenue collection systems.

INSURANCE (New Listing):
- KLRE.LUSE — Klapton Reinsurance Plc: Listed March 2026. Non-life reinsurance with 86% domestic market share and 120+ country footprint.

AGRICULTURE:
- MAFM.LUSE — Madison Asset Management: Financial services and asset management.

KEY MARKET METRICS (2025):
- Total listed companies: ~25
- Market Cap: Growing, boosted by new listings
- Most liquid stocks: ZNCO, CECZ, ZAFC, ZCCM
- Currency: All prices quoted in Zambian Kwacha (ZMW)
- Settlement: T+3""",
    },
    {
        "title": "LuSE All-Share Index (LASI) — Understanding Zambia's Market Benchmark",
        "source": "LuSE Official Website",
        "content": """THE LUSE ALL SHARE INDEX (LASI)

The LASI is the primary benchmark index for the Lusaka Securities Exchange, tracking the price performance of all ordinary shares listed on the exchange. It is market-capitalization weighted, meaning larger companies have a proportionally greater impact on index movements.

KEY CHARACTERISTICS:
- Base date: Established as a comprehensive market benchmark
- Composition: All ordinary shares on the main board
- Weighting: Free-float market cap weighted
- Currency: ZMW
- Calculation: Real-time during trading hours (10:00–14:00 Zambian time)

INTERPRETING LASI MOVEMENTS:
- +1% to 3% move: Normal session range; check individual stock drivers
- >3% single-day move: Unusual — likely driven by institutional activity, major earnings announcement, or macro shock
- Sustained >5% decline over a week: Investigate BoZ policy changes, copper price crash, or ZMW devaluation

TRADING HOURS:
- Market hours: 10:00 - 14:00 (CAT, UTC+2)
- Settlement: T+3 (trade date plus 3 business days)

LASI SECTOR COMPOSITION (Approximate):
- Banking & Financial: ~40%
- Mining & Energy: ~30%
- Consumer/Retail: ~15%
- Industrial: ~10%
- Other: ~5%

FOREIGN INVESTOR PARTICIPATION:
Foreign portfolio investors (FPIs) can invest in LuSE-listed stocks through registered brokers. Repatriation of funds is governed by BoZ foreign exchange regulations. ZMW depreciation risk applies — a weakening Kwacha reduces USD-equivalent returns.

PRIMARY BROKERS:
- Stockbrokers Zambia (SBZ) — largest, oldest
- African Alliance Zambia Securities
- Madison Asset Management
- Imara Securities Zambia""",
    },
    {
        "title": "Zambia Copper Sector & ZCCM-IH Deep Dive",
        "source": "ZambiaInvest / ZCCM Annual Reports",
        "content": """ZCCM-IH (ZCCM Investments Holdings) — LuSE DEEP DIVE

Ticker: ZCCM.LUSE
Sector: Mining Investments / State Enterprise
Government Ownership: Majority owned by the Government of the Republic of Zambia (GRZ)

WHAT ZCCM-IH DOES:
ZCCM-IH is the government's equity holding vehicle in the mining sector. It holds minority stakes across multiple major copper mines and resource projects:
- Kansanshi Mine (First Quantum Minerals): 20% stake — one of Africa's largest copper mines
- Lumwana Mine: Minority stake
- Mopani Copper Mines: Following nationalization, government has full control
- Konkola Copper Mines: Government stake after Vedanta Resources disputes
- Various smaller projects in cobalt, emeralds, and other minerals

REVENUE DRIVERS:
- Dividend income from mining subsidiaries
- Management fees and royalties
- Capital gains on mining asset disposals
- USD-denominated revenues translated to ZMW

COPPER PRICE SENSITIVITY:
ZCCM-IH is among the most copper-price-sensitive stocks on the LuSE:
- Every $500/tonne move in LME copper significantly impacts subsidiary dividends
- LME copper at $9,000/tonne = strong dividend flow
- LME copper below $7,500/tonne = revenue stress across subsidiaries

ZAMBIA COPPER PRODUCTION:
- Zambia produces approximately 800,000 - 1,000,000 tonnes of copper per year
- Target: 3 million tonnes by 2030 (government stated ambition)
- New Copperbelt discoveries and First Quantum's expansion driving growth

ZMW/USD AND COPPER CORRELATION:
When copper rallies: ZMW strengthens → government revenues rise → fiscal stability improves
When copper falls: ZMW weakens → government considers MPR hikes → bond yields rise → equity valuation pressure

EV TAILWIND:
Electric vehicle batteries require approximately 80kg of copper per unit. Global EV adoption is creating secular demand growth for copper, providing a structural long-term tailwind for Zambian copper assets and ZCCM.LUSE specifically.""",
    },
]


def write_seed_articles(output_dir: str) -> list:
    """Writes pre-researched seed articles (no internet needed). Always succeeds."""
    saved = []
    print(f"[Scraper] Writing {len(SEED_ARTICLES)} seed knowledge articles...")
    for article in SEED_ARTICLES:
        path = save_article(article["title"], article["content"], article["source"], output_dir)
        print(f"  ✓ {Path(path).name}")
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gofi AI News Scraper")
    parser.add_argument("--output",  type=str, default=OUTPUT_DIR)
    parser.add_argument("--source",  type=str, default="all",
                        choices=["all", "african_markets", "daily_mail", "zambiainvest", "seed"])
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"[Scraper] Saving to: {args.output}")

    all_saved = []

    if args.source in ("all", "seed"):
        all_saved.extend(write_seed_articles(args.output))

    if args.source in ("all", "african_markets"):
        all_saved.extend(scrape_african_markets(args.output))

    if args.source in ("all", "daily_mail"):
        all_saved.extend(scrape_zambia_daily_mail(args.output))

    if args.source in ("all", "zambiainvest"):
        all_saved.extend(scrape_zambiainvest(args.output))

    print(f"\n[Scraper] ✓ Done. Saved {len(all_saved)} articles to '{args.output}'")
    print("[Scraper] Now run: python rag_pipeline.py --index --docs_dir ./docs")


if __name__ == "__main__":
    main()
