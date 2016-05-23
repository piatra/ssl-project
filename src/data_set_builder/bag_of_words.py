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

# roc curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from models import db_connect, WebsitesContent, Websites, WebsitesCache
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
              'tfidf__use_idf': [True, False],
              'tfidf__norm': ['l2', 'l1'],
              'vect__max_features': [1000],
              'vect__binary': [True, False]
}


def get_alexa_demographics(url, db_session=False):
    if db_session is not False:
        result = list(db_session.query(WebsitesCache).filter_by(link=url))
        if len(result) > 0 and result[0].male_ratio_alexa >= 0:
            return float(result[0].male_ratio_alexa), float(result[0].female_ratio_alexa)
        else:
            return 0.0, 0.0

    orig_url = url
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

    male_ratio = 0.0
    female_ratio = 0.0
    if sum(values) == 0:
        print "No alexa rating for " + url
    else:
        male_ratio = float(values[0] + values[1]) / sum(values)
        female_ratio = float(values[2] + values[3]) / sum(values)
        print url
        print values
        print male_ratio, female_ratio

    # Do we want to cache the result?
    if db_session is not False:
        try:
            db_session.query(WebsitesCache).filter(WebsitesCache.link==orig_url) \
                      .update({
                          'male_ratio_alexa': male_ratio,
                          'female_ratio_alexa': female_ratio
                       })
            db_session.commit()
        except:
            print "Could not update " + url

    return male_ratio, female_ratio

def sgdc_parameters():
    clf_params = {
        'sgdcclf__penalty': ['l2'],
        'sgdcclf__alpha': [0.001, 0.002],
        'sgdcclf__loss': ['modified_huber'],
        'sgdcclf__n_iter': [5]
    }

    return dict(clf_params, **parameters)


def sgdc_pipeline():
    if PRODUCTION:
        return Pipeline([('vect', CountVectorizer(ngram_range=(1,3),
                                  analyzer='word', tokenizer=LemmaTokenizer(),
                                  max_features=1000, binary=True)),
                         ('tfidf', TfidfTransformer(use_idf=True, norm='l1')),
                         ('sgdcclf', SGDClassifier(loss='log',
                                                   n_iter=1000,
                                                   penalty='l1'))])

    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('sgdcclf', SGDClassifier()),
                    ])


def randomForest_parameters():
    clf_params = {
            'forest__n_estimators': [800, 1200],
            'forest__max_features': ['sqrt'],
            'forest__min_samples_leaf': [100, 200, 500],
            }

    return dict(clf_params, **parameters)


def randomForest_pipeline():
    if PRODUCTION:
        return Pipeline([('vect', CountVectorizer(ngram_range=(2,3),
                                      analyzer='word', tokenizer=LemmaTokenizer(),
                                      max_features=1000, binary=True)),
                         ('tfidf', TfidfTransformer(use_idf=True, norm='l1')),
                         ('forest', RandomForestClassifier(max_features='sqrt',
                                    n_estimators=1000))
                        ])

    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('forest', RandomForestClassifier())])


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
    # classifier = multinomial_pipeline()
    # classifier = sgdc_pipeline()
    # parameters = sgdc_parameters()
    classifier = randomForest_pipeline()
    parameters = randomForest_parameters()
    # classifier = svc_pipeline()
    # parameters = svc_parameters()

    if PRODUCTION:
        classifier.fit(training_data, training_classes)
        return classifier.predict_proba(test_data)
    else:
        gs_clf = GridSearchCV(classifier, parameters, n_jobs=6)
        gs_clf.fit(training_data + test_data, training_classes + test_data_classes)

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
            words = extract_words_from_url(url, session)
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
    alexa_probabilities = []

    if PRODUCTION:
        # probabilities :: [[male_confidence, female_confidence]]
        # diff between male_confidence and female_confidence.
        # if more likely to be male value will be positive, negative for female.
        gender_prob = [v[0] - v[1] for v in probabilities]

        results = []
        for idx, url in enumerate(crawled_urls):
            alexa_male_confidence, alexa_female_confidence = get_alexa_demographics(url,session)
            r = (gender_prob[idx],
                 hp.get_frequency(url), # no of times page was visited
                 gender_prob[idx] * hp.get_frequency(url),
                 url,
                 alexa_male_confidence,
                 alexa_female_confidence,
                 (alexa_male_confidence - alexa_female_confidence) * hp.get_frequency(url))
            alexa_probabilities.append([alexa_male_confidence, alexa_female_confidence])
            results.append(r)

        prediction = [0, 0]
        alexa_result = [0, 0]
        for x in results:
            if x[6] > 0:
                alexa_result[0] += x[6]
            else:
                alexa_result[1] += abs(x[6])

            if x[2] > 0:
                prediction[0] += x[2]
            else:
                prediction[1] += abs(x[2]) # female_confidence is neg

        # normalize and print the prediction
        print 'Prediction:'
        print [float(x) / sum(prediction) for x in prediction]

        print 'Alexa:'
        print [float(x) / sum(alexa_result) for x in alexa_result]

        for i in range(len(alexa_probabilities)):
            if alexa_probabilities[i][0] > alexa_probabilities[i][1]:
                alexa_probabilities[i] = CLASSES['male']
            else:
                alexa_probabilities[i] = CLASSES['female']

        my_probs = []
        for i in range(len(probabilities)):
            if probabilities[i][0] > probabilities[i][1]:
                my_probs.append(CLASSES['male'])
            else:
                my_probs.append(CLASSES['female'])

        print "my_probs"
        print my_probs
        print "alexa_probabilities"
        print alexa_probabilities

        fpr, tpr, _ = roc_curve(my_probs, alexa_probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('out_forest.png')


if __name__ == '__main__':
    main()
