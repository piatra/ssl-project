from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

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


# Example usage
# training_data = ["ana are mere", "gigi are pere", "stefan are mere", "cristi are pere"]
# training_classes = [0, 1, 0, 1]
# test_data = ["nicu are mere", "alex are pere"]
# print classify(training_data, training_classes, test_data)
