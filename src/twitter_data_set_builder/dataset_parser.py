# import nltk
# nltk.download()
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

cachedStopWords = stopwords.words("english")

def clean(text):
    # 0. Remove urls
    text = re.sub(r"http\S+", "", text)
    # 1. Remove HTML
    review_text = BeautifulSoup(text, "lxml").get_text()
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in cachedStopWords]
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

if __name__ == "__main__":
    frequency = {}
    with open("out2") as f:
        content = f.readlines()
        for line in content:
            line = clean(line)
            for word in line.split():
                if word in frequency:
                    frequency[word] = frequency[word] + 1
                else:
                    frequency[word] = 1
    top = []
    for word in frequency:
        if len(top) < 50:
            top.append((word, frequency[word]))
        else:
            for i in range(0, len(top) + 1):
                if i == len(top):
                    top[i - 1] = (word, frequency[word])
                    break
                if top[i][1] <= frequency[word]:
                    continue
                elif i > 0 and top[i - 1][1] < frequency[word]:
                    top[i - 1] = (word, frequency[word])
                    break
                else:
                    break
    for w in top:
        print w[0]
