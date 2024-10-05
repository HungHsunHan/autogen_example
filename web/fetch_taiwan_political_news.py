# filename: fetch_taiwan_political_news.py

from datetime import datetime

import feedparser


def fetch_news():
    # Google News RSS feed for Taiwan political news
    news_url = (
        "https://news.google.com/rss/search?q=台灣+政治&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    )
    feed = feedparser.parse(news_url)

    articles = []
    today_date = datetime.now().strftime("%Y-%m-%d")

    for entry in feed.entries:
        pub_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
        if pub_date == today_date:
            title = entry.title
            link = entry.link
            description = entry.summary
            articles.append(f"### [{title}]({link})\n\n{description}\n\n")

    return articles


def save_to_markdown(articles):
    with open("output.md", "w", encoding="utf-8") as file:
        file.write("# 台灣今日政治新聞\n\n")
        if articles:
            file.write("\n".join(articles))
        else:
            file.write("今天沒有找到相關的政治新聞.")


def main():
    articles = fetch_news()
    save_to_markdown(articles)
    print("新聞已以markdown格式保存至output.md文件中.")


if __name__ == "__main__":
    main()
