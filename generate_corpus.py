import os
import json
import urllib.request
import xml.etree.ElementTree as ET

out_dir = "data/calibration"
out_file = os.path.join(out_dir, "post_cutoff_articles.jsonl")

feeds = [
    # General Global News
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
    "http://rss.cnn.com/rss/edition.rss",
    "http://rss.cnn.com/rss/edition_world.rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/uk/rss",
    "https://www.aljazeera.com/world/rss",
    
    # Business & Finance
    "https://search.cnbc.com/ping/ig/api/v1/search/rss?query=%2B%2B&source=all",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
    "https://www.ft.com/business/rss",
    
    # Technology & Science
    "https://techcrunch.com/rssfeeds/",
    "https://www.wired.com/feed/rss",
    "https://www.theverge.com/feed/rss",
    "https://www.polygon.com/rss/index.xml",
    "https://www.engadget.com/rss.xml",
    "https://www.space.com/feed"
]

articles = []

print("Fetching recent news for Min-K%++ calibration...")
for url in feeds:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            # Extract the description/summary of each article
            for item in root.findall('.//item'):
                desc = item.find('description')
                if desc is not None and desc.text and len(desc.text) > 50:
                    # Clean up basic HTML tags sometimes found in RSS
                    clean_text = desc.text.replace('<p>', '').replace('</p>', '').strip()
                    articles.append(clean_text)
    except Exception as e:
        print(f"  [!] Failed to fetch {url}: {e}")

articles = list(set(articles))

target_n = 300
articles = articles[:target_n]

os.makedirs(out_dir, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    for text in articles:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print(f"Success! Saved {len(articles)} documents to {out_file}")