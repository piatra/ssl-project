from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.orm import sessionmaker

from models import db_connect, WebsitesContent, Websites
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

    vectorizer = CountVectorizer(analyzer = "word",    \
                                 tokenizer = None,     \
                                 ngram_range = (1, 3), \
                                 preprocessor = None,  \
                                 stop_words = None,    \
                                 max_features = 600)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    return vectorizer, train_data_features


def forrest_classifier(train_data_features, classes):
    forest = RandomForestClassifier()

    return forest.fit(train_data_features, classes)


def classify(training_data, training_classes, test_data, test_data_classes):
    vectorizer, features_train_data = bag_of_words(training_data)
    classifier = forrest_classifier(features_train_data, training_classes)

    if len(test_data_classes) is 0:
        return classifier.predict(vectorizer.transform(test_data))
    else:
        print classifier.score(vectorizer.transform(test_data), test_data_classes)


def get_history():
    return ['https://www.cars.com']


def main():
    """Build training set and clasify history. """
    engine = db_connect()
    Session = sessionmaker(bind=engine)
    session = Session()

    training_set = []
    labels = []
    websites_query = list(session.query(Websites).filter(
            Websites.male_ratio_alexa > 0)) # some websites do not have demographics
    for website in websites_query:
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
    test_set_classes = []

    if False:
        for url in get_history():
            words = extract_words_from_url(url)
            words = ' '.join(words)

            test_set.append(words)
    else:
        l = len(training_set)
        print "Training set has %i items" % l
        cut = int(l / 4 * 3)
        test_set = training_set[cut:]
        test_set_classes = labels[cut:]
        training_set = training_set[0:cut]
        labels = labels[0:cut]

    print classify(training_set, labels, test_set, test_set_classes)


if __name__ == '__main__':
    main()
