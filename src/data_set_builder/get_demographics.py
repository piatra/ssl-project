from sqlalchemy.orm import sessionmaker

# scraping modules
from scrapy.crawler import CrawlerProcess, Crawler
from demographic_scraper.demographic_scraper.spiders.alexa_spider import AlexaSpider
from scrapy.utils.project import get_project_settings

from models import db_connect, WebsitesContent, Websites


def main():
    """Index alexa demographics
    """

    engine = db_connect()
    Session = sessionmaker(bind=engine)
    session = Session()

    settings = get_project_settings()
    settings.set('ITEM_PIPELINES',
                 {'demographic_scraper.demographic_scraper.pipelines.WebsiteDemographicPipeline': 300})
    settings.set('EXTENSIONS',
                 {'scrapy.telnet.TelnetConsole': None,})


    process = CrawlerProcess(settings)
    for website in session.query(WebsitesContent).all():
        demographic = list(session.query(Websites).filter_by(link=website.link))
        if len(demographic) is 0:
            url = website.link
            print website.link
            AlexaSpider.name = url
            process.crawl(AlexaSpider, url=url, db_session=session)
    process.start()
    process.stop()

    session.close()


if __name__ == '__main__':
    main()
