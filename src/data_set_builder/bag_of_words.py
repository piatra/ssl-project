from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.orm import sessionmaker

from models import create_db_tables, db_connect, WebsitesContent, Websites
from data_set_builder import extract_words_from_url

CLASSES = {
    'male': 0,
    'female': 1,
}


def bag_of_words(clean_train_reviews):
    """
        Transform an array of sentences into features.
        The method ASSUMES SENTENCES BEEN CLEANED FOR STOP WORDS.
    """

    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 2000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    return vectorizer, train_data_features


def forrest_classifier(train_data_features, classes):
    forest = RandomForestClassifier(n_estimators = 100)

    return forest.fit(train_data_features, classes)


def classify(training_data, training_classes, test_data):
    vectorizer, features_train_data = bag_of_words(training_data)
    classifier = forrest_classifier(features_train_data, training_classes)

    return classifier.predict(vectorizer.transform(test_data))


def get_history():
    return ['https://www.cars.com']


def main():
    """Build training set and clasify history. """
    engine = db_connect()
    Session = sessionmaker(bind=engine)
    session = Session()

    training_set = []
    labels = []
    for website in session.query(Websites).all():
        website_content = session.query(WebsitesContent).filter(
            WebsitesContent.link==website.link)[0]
        words = website_content.words
        words = ' '.join(words)

        gender = CLASSES['male']
        if website.male_ratio_alexa < website.female_ratio_alexa:
            gender = CLASSES['female']

        training_set.append(words)
        labels.append(gender)

    test_set = []
    for url in get_history():
        words = extract_words_from_url(url)
        words = ' '.join(words)

        test_set.append(words)

    print classify(training_set, labels, test_set)


if __name__ == '__main__':
    main()
