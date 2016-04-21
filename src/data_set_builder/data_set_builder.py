import re
import time
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import requests
from sqlalchemy.orm import sessionmaker

# scraping modules
from scrapy.crawler import CrawlerProcess
from demographic_scraper.demographic_scraper.spiders.alexa_spider import AlexaSpider
from scrapy.utils.project import get_project_settings

from models import create_db_tables, db_connect, WebsitesContent
from google import search

MAX_RESULTS_LIMIT = 25


def crawl_for_demographics(url):
    """Fetch demographics from (currently just) Alexa.com"""
    settings = get_project_settings()
    settings.set('ITEM_PIPELINES',
                 {'demographic_scraper.demographic_scraper.pipelines.WebsiteDemographicPipeline': 300})
    process = CrawlerProcess(settings)

    process.crawl(AlexaSpider, url=url)
    process.start()


def get_queries():
    """Returns a list of queries we want to index data.
    Note: for now it is a hardcoded list but it will be replaced with data fetched from Twitter.
    """
    queries = []
    with open('most_popular_words', 'r') as f:
        queries = f.readlines()

    queries = [q.strip() for q in queries]
    queries.reverse()
    return queries


def index_query_data(query, db_session):
    """Fetch google pages for a query and index their content. """
    words = []
    for url in search(query, stop=MAX_RESULTS_LIMIT):
        try:
            words = extract_words_from_url(url)
            db_session.add(WebsitesContent(link=url, words=words))
            db_session.commit()
        except:
            print url

    return words


def extract_words_from_url(url):
    """Use BeautifulSoup to extract visible text from a web page. """
    def visible(element):
        """Method used to filter visible text elements. """
        if element.parent.name in ['style', 'script', '[document]', 'head', 'link']:
            return False
        elif re.match('<!--.*-->', element.encode('utf-8')):
            return False

        value = element.encode('utf-8')
        return not value.isspace()

    html = requests.get(url, verify=False).content
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)

    visible_texts = [text for text in texts if visible(text)]

    words = []
    stop_words = stopwords.words('english')
    for text in visible_texts:
        for word in re.findall(r"[\w']+", text):
            word = word.lower()
            if not word.isdigit() and word not in stop_words:
                words.append(word)


    return words


def main():
    """Based on a list of queries:
     - get the first 50 results from google
     - scrap their text
     - store text in the db
     - get demographics for page
     - store demographics
    """
    engine = db_connect()
    create_db_tables(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    quereies = get_queries()
    for query in quereies:
        print query
        try:
            index_query_data(query, session)
        except Exception as e:
            print 'Failed to fetch query: %s' % query
        time.sleep(60)


    session.close()


if __name__ == '__main__':
    main()
