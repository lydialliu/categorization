import os
import re
import string
import math
import random
import numpy as np
import csv
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')



#Reference: https://pythonmachinelearning.pro/text-classification-tutorial-with-naive-bayes/


def get_data(data_path, seed = 123):

    cat_data_path = os.path.join(data_path, 'cat')
    data = []

    """
    csv format:
    id, name, cb_desc, pb_desc, cat, cat_id
    """

    with open(os.path.join(cat_data_path, "final_data_view.csv")) as datafile:
        i = -1  # b/c first row isn't data
        reader = csv.reader(datafile, quoting=csv.QUOTE_ALL)
        for row in reader:  # each row is a list
            data.append(row)

    data = data[1:]
    random.seed(seed)
    random.shuffle(data)

    train_texts = []
    train_labels = []
    for row in data:
        train_texts.append(row[2])
        train_labels.append(row[5])


    return data, train_texts, train_labels


class NaiveModel(object):
    """Implementation of Naive Bayes for classification"""

    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)

    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def fit(self, X, Y):
        """Fit our classifier
        Arguments:
            X {list} -- list of document contents
            y {list} -- correct labels
        """

        self.num_descs = {}
        self.log_class_priors = {}
        self.word_counts = {}
        # word_counts is {class_label : {word1 : count, word2 : count}, class_label : {word1 ...

        for label in set(Y):
            self.num_descs[label] = 0
            self.word_counts[label] = {}

        self.vocab = set()
        n = len(X)

        for label in Y:
            self.num_descs[label] += 1

        for label in Y:
            self.log_class_priors[label] = math.log(self.num_descs[label] / n)

        stop_words = set(stopwords.words('english'))

        for x, y in zip(X, Y):
            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                if (word not in self.vocab) and (word not in stop_words):
                    self.vocab.add(word)
                if word not in self.word_counts[y]:
                    self.word_counts[y][word] = 0.0
                if word not in stop_words:
                    self.word_counts[y][word] += count

        #removes words that only appear once
        for label in self.word_counts:
            for word in self.word_counts[label]:
                if (self.word_counts[label][word] < 2):
                    self.word_counts[label][word] = 0


    def predict(self, X):
        result = []
        for x in X:
            counts = self.get_word_counts(self.tokenize(x))
            print(counts)
            scores = {}
            for label in set(y):
                scores[label] = 0
            print(self.num_descs)
            for word, _ in counts.items():
                if word not in self.vocab: continue
                for label in scores:
                    # add Laplace smoothing
                    log_w_given_label = math.log(
                        (self.word_counts[label].get(word, 0.0) + 1) / (self.num_descs[label] + len(self.vocab)))
                    scores[label] += log_w_given_label


            for label in scores:
                scores[label] += self.log_class_priors[label]
            """
            for label in scores:
                result.append(scores[label])
            """
            max_score_label = max(scores, key=scores.get)
            result.append(max_score_label)
        return result


if __name__ == '__main__':
    data, X, y = get_data('', 11)
    MNB = NaiveModel()
    MNB.fit(X[10:], y[10:])

    test_texts = X[:10]
    pred = MNB.predict(test_texts)
    true = y[:10]

    accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))
