from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.orm import sessionmaker

from models import db_connect, WebsitesContent, Websites
from data_set_builder import extract_words_from_url
from parse_user_history.history_parser import HistoryParser

CLASSES = {
    'male': 0,
    'female': 1,
}

PRODUCTION = True
hp = HistoryParser("../parse_user_history/andrei_history.json")

def bag_of_words(clean_train_reviews):
    """
        Transform an array of sentences into features.
        The method ASSUMES SENTENCES BEEN CLEANED FOR STOP WORDS.
    """

    vectorizer = CountVectorizer(analyzer = "word",    \
                                 tokenizer = None,     \
                                 ngram_range = (1, 2), \
                                 preprocessor = None,  \
                                 stop_words = None,    \
                                 max_features = 600)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    return vectorizer, train_data_features


def forrest_classifier(train_data_features, classes):
    forest = RandomForestClassifier(n_estimators=600, \
                                    max_features="sqrt")

    return forest.fit(train_data_features, classes)


def classify(training_data, training_classes, test_data, test_data_classes):
    vectorizer, features_train_data = bag_of_words(training_data)
    classifier = forrest_classifier(features_train_data, training_classes)

    if len(test_data_classes) is 0:
        return classifier.predict_proba(vectorizer.transform(test_data))
        # return classifier.predict(vectorizer.transform(test_data))
    else:
        print classifier.score(vectorizer.transform(test_data), test_data_classes)


def get_history():
    return hp.unique_links().keys()

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
    crawled_urls = []

    if PRODUCTION:
        for url in get_history():
            words = extract_words_from_url(url)
            if len(words) > 20: # Skip 404/403 pages.
                crawled_urls.append(url)
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

    probabilities = classify(training_set, labels, test_set, test_set_classes)
    # probabilities :: [[male_confidence, female_confidence]]
    # diff between male_confidence and female_confidence.
    # if more likely to be male value will be positive, negative for female.
    gender_prob = [v[0] - v[1] for v in probabilities]

    results = []
    for idx, url in enumerate(crawled_urls):
        r = (gender_prob[idx],
             hp.get_frequency(url), # no of times page was visited
             gender_prob[idx] * hp.get_frequency(url),
             url)
        results.append(r)
        print r

    prediction = [0, 0]
    for x in results:
        if x[2] > 0:
            prediction[0] = prediction[0] + x[2]
        else:
            prediction[1] = prediction[1] + abs(x[2]) # female_confidence is neg

    # normalize and print the prediction
    print [float(x) / sum(prediction) for x in prediction]


if __name__ == '__main__':
    main()
