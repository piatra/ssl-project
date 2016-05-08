from collections import defaultdict

import nltk
from sqlalchemy.orm import sessionmaker


from models import db_connect, WebsitesPhrases, Websites


def tag_part_of_speech(phrases):
    pos_list = []
    for phrase in phrases:
         text = nltk.word_tokenize(phrase)
         pos_list.extend(nltk.pos_tag(text))

    return pos_list


def get_pos(websites):
    pos_list = []
    for website in websites:
        pos_list.extend(tag_part_of_speech(website.phrases))

    pos_dict = defaultdict(int)
    for pos in pos_list:
        # pos is a tuple (word, part-of-speech)
        pos_dict[pos[1]] += 1

    return pos_dict


def main():
    engine = db_connect()

    Session = sessionmaker(bind=engine)
    session = Session()

    male_websites = []
    female_websites = []
    for website_content in session.query(WebsitesPhrases).all():
        website = list(session.query(Websites).filter_by(link=website_content.link))[0]
        if len(male_websites) < 10 and website.male_ratio_alexa - website.female_ratio_alexa >= 0.25:
            male_websites.append(website_content)
        elif len(female_websites) < 10 and website.female_ratio_alexa - website.male_ratio_alexa >= 0.25:
            female_websites.append(website_content)


    male_pos = get_pos(male_websites)
    female_pos = get_pos(female_websites)


if __name__ == '__main__':
    main()
