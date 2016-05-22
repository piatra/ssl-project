import re

from parsel import Selector
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sqlalchemy.orm import sessionmaker
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

# Stemming
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from models import db_connect, WebsitesContent, Websites
from data_set_builder import extract_words_from_url
from parse_user_history.history_parser import HistoryParser

CLASSES = {
    'male': 0,
    'female': 1,
}
PRODUCTION = False
hp = HistoryParser("../parse_user_history/andrei_history.json")


# Stemming
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


parameters = {'vect__ngram_range': [(1, 2)],
              'vect__analyzer': ['word'],
              'vect__tokenizer': [None, LemmaTokenizer()],
              'vect__preprocessor': [None],
              'vect__stop_words': [None],
              'tfidf__use_idf': [True],
              'tfidf__norm': ['l2'],
              'vect__max_features': [1000],
              'vect__binary': [True]
}


def get_alexa_demographics(url):
    url = "http://www.alexa.com/siteinfo/" + url
    response = requests.get(url)

    # We need the decode part because Selector expects unicode.
    selector = Selector(response.content.decode('utf-8'))
    bars = selector.css("#demographics-content .demo-col1 .pybar-bg")
    values = []
    for bar in bars:
        value = bar.css("span::attr(style)").extract()[0]
        value = int(re.search(r'\d+', value).group())
        values.append(value)

    male_ratio = 0
    female_ratio = 0
    if sum(values) == 0:
        return male_ratio, female_ratio

    return float(values[0] + values[1]) / sum(values), float(values[2] + values[3]) / sum(values)

def sgdc_parameters():
    clf_params = {
        'sgdcclf__penalty': ['l2'],
        'sgdcclf__alpha': [0.001, 0.002],
        'sgdcclf__loss': ['hinge'],
        'sgdcclf__n_iter': [5]
    }

    return dict(clf_params, **parameters)


def randomForest_parameters():
    clf_params = {
            'forest__n_estimators': [800, 1000],
            'forest__max_features': ['sqrt'],
            }

    return dict(clf_params, **parameters)


def svc_parameters():
    clf_params = {
            'svcclf__kernel': ['rbf'],
            'svcclf__gamma': [1e-3, 1e-4],
            'svcclf__C': [1, 10, 100, 1000]
            }

    return dict(clf_params, **parameters)


def multinomial_pipeline():
    if PRODUCTION:
        return Pipeline([('vect', CountVectorizer(ngram_range=(1,2),
                                  analyzer='word', tokenizer=LemmaTokenizer(),
                                  max_features=800, binary=True)),
                         ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
                         ('nbclf', MultinomialNB()),
                        ])

    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('nbclf', MultinomialNB()),
                    ])


def sgdc_pipeline():
    if PRODUCTION:
        return Pipeline([('vect', CountVectorizer(ngram_range=(1,2),
                                  analyzer='word', tokenizer=LemmaTokenizer(),
                                  max_features=1000, binary=True)),
                         ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
                         ('sgdcclf', SGDClassifier(alpha=0.001, loss='hinge',
                                     n_iter=5, penalty='l2')),
                        ])

    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('sgdcclf', SGDClassifier()),
                    ])


def randomForest_pipeline():
    if PRODUCTION:
        return False

    return Pipeline([('vect', CountVectorizer(ngram_range=(1,2),
                                  analyzer='word', tokenizer=LemmaTokenizer(),
                                  max_features=1000, binary=True)),
                     ('forest', RandomForestClassifier(max_features='sqrt',
                                n_estimators=800))
                    ])


def svc_pipeline():
    if PRODUCTION:
        return Pipeline([('vect', CountVectorizer(ngram_range=(1,2),
                                  analyzer='word', tokenizer=LemmaTokenizer(),
                                  max_features=1000, binary=True)),
                         ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),
                         ('svcclf', SVC(kernel='rbf', gamma='0.001', C='1000')),
                        ])

    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('svcclf', SVC()),
                    ])


def pipeline_classify(training_data, training_classes, test_data, test_data_classes):
    classifier = multinomial_pipeline()
    # classifier = sgdc_pipeline()
    # parameters = sgdc_parameters()
    # classifier = randomForest_pipeline()
    # parameters = randomForest_parameters()
    # classifier = svc_pipeline()
    # parameters = svc_parameters()
    gs_clf = GridSearchCV(classifier, parameters, n_jobs=-1)

    if PRODUCTION:
        classifier.fit_transform(training_data, training_classes)
        return classifier.predict_proba(test_data)
    else:
        gs_clf.fit(training_data, training_classes)

        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        print score

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

    history = get_history()
    if PRODUCTION:
        for url in history:
            words = extract_words_from_url(url)
            if len(words) > 20: # Skip 404/403 pages.
                crawled_urls.append(url)
                words = ' '.join(words)
                test_set.append(words)
    else:
        l = len(training_set)
        print "Training set has %i items" % l
        cut = int(l / 2 * 1)
        test_set = training_set[cut:]
        test_set_classes = labels[cut:]
        training_set = training_set[0:cut]
        labels = labels[0:cut]

    probabilities = pipeline_classify(training_set, labels, test_set, test_set_classes)

    if PRODUCTION:
        # probabilities :: [[male_confidence, female_confidence]]
        # diff between male_confidence and female_confidence.
        # if more likely to be male value will be positive, negative for female.
        gender_prob = [v[0] - v[1] for v in probabilities]

        results = []
        for idx, url in enumerate(crawled_urls):
            alexa_male_confidence, alexa_female_confidence = get_alexa_demographics(url)
            r = (gender_prob[idx],
                 hp.get_frequency(url), # no of times page was visited
                 gender_prob[idx] * hp.get_frequency(url),
                 url,
                 alexa_male_confidence,
                 alexa_female_confidence)
            results.append(r)
            print r

        prediction = [0, 0]
        alexa_result = [0, 0]
        for x in results:
            if x[2] > 0:
                prediction[0] += x[2]
                alexa_result[0] += x[4]
            else:
                prediction[1] += abs(x[2]) # female_confidence is neg
                alexa_result[0] += x[4]

        # normalize and print the prediction
        print 'Prediction:'
        print [float(x) / sum(prediction) for x in prediction]

        print 'Alexa:'
        print [float(x) / sum(alexa_result) for x in alexa_result]


if __name__ == '__main__':
    main()
