from bs4 import BeautifulSoup
import requests
from sqlalchemy.orm import sessionmaker


from data_set_builder import visible
from models import db_connect, WebsitesPhrases, Websites, create_db_tables


def extract_phrases_from_url(url):
    """Use BeautifulSoup to extract visible text from a web page. """
    html = requests.get(url, verify=False).content
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)

    return [text.encode('utf-8') for text in texts if visible(text)]


def main():
    """Index websites content by phrase.
    """

    engine = db_connect()
    create_db_tables(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    import ipdb; ipdb.set_trace()
    for website in session.query(Websites).all():
        url = website.link
        try:
            phrases = extract_phrases_from_url(url)
            session.add(WebsitesPhrases(link=url, phrases=phrases))
            session.commit()
        except:
            session.rollback()
            print url


if __name__ == '__main__':
    main()
