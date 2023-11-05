# models.py

from sentiment_data import *
from utils import *
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import math
# import matplotlib.pyplot as plt


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, ex_words: List[str], add_to_indexer: bool) -> List[int]:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param ex_words: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return:
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.Indexer = Indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, ex_words: List[str], add_to_indexer: bool) -> List[int]:
        features_of_str = np.zeros(self.indexer.__len__())
        for ele in ex_words:
            if self.indexer.contains(ele.lower()):
                features_of_str[self.indexer.index_of(ele.lower())] += 1
        return features_of_str


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, ex_words: List[str], add_to_indexer: bool) -> List[int]:
        features_of_str = np.zeros(self.indexer.__len__(), dtype=int)
        for i in range(0, len(ex_words) - 1):
            bigram = ex_words[i] + ' ' + ex_words[i + 1]
            if self.indexer.contains(bigram.lower()):
                index = self.indexer.index_of(bigram.lower())
                features_of_str[index] += 1
        return features_of_str


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, ex_words: List[str], add_to_indexer: bool) -> List[int]:
        features_of_str = np.zeros(self.indexer.__len__())
        for ele in ex_words:
            if self.indexer.contains(ele.lower()):
                features_of_str[self.indexer.index_of(ele.lower())] += 1
        return features_of_str


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, ex_words: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        features_of_str = self.feat_extractor.extract_features(ex_words, False)
        expo = math.exp(np.dot(self.weights, features_of_str))
        possibility = expo / (1 + expo)
        if possibility > 0.5:
            return 1
        return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    raise Exception("Must be implemented")


def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexer = feat_extractor.get_indexer()
    weights = np.transpose(np.zeros(indexer.__len__(), dtype=int))
    learning_rate = 0.1
    for i in range(15):
        for ex in train_exs:
            features_of_str = feat_extractor.extract_features(ex.words, False)
            expo = math.exp(np.dot(weights, features_of_str))
            possibility = expo / (1 + expo)
            gradient_of_w = np.dot(ex.label - possibility, features_of_str)
            weights = np.add(weights, np.dot(learning_rate, gradient_of_w))
    return LogisticRegressionClassifier(weights, feat_extractor)

    # Methods for plotting average training loss

    # x = np.arange(0, 14)
    # # learning_rate = 1
    # indexer = feat_extractor.get_indexer()
    # weights = np.transpose(np.zeros(indexer.__len__(), dtype=int))
    # avrg_losses = np.zeros(14)
    # for i in range(15):
    #     for ex in train_exs:
    #         features_of_str = feat_extractor.extract_features(ex.words, False)
    #         expo = math.exp(np.dot(weights, features_of_str))
    #         possibility = expo / (1 + expo)
    #         gradient_of_w = np.dot(ex.label - possibility, features_of_str)
    #         weights = np.add(weights, gradient_of_w)
    #     loss = 0
    #     for ex in train_exs:
    #         features_of_str = feat_extractor.extract_features(ex.words, False)
    #         expo = math.exp(np.dot(weights, features_of_str))
    #         possibility = expo / (1 + expo)
    #         loss += -(ex.label * math.log(possibility) + (1 - ex.label) * math.log(1 - possibility))
    #     avrg_losses[i - 1] = loss / train_exs.__len__()
    # plt.plot(x, avrg_losses)
    #
    # # learning_rate = 0.01
    # weights = np.transpose(np.zeros(indexer.__len__(), dtype=int))
    # learning_rate = 0.01
    # avrg_losses = np.zeros(14)
    # for i in range(15):
    #     for ex in train_exs:
    #         features_of_str = feat_extractor.extract_features(ex.words, False)
    #         expo = math.exp(np.dot(weights, features_of_str))
    #         possibility = expo / (1 + expo)
    #         gradient_of_w = np.dot(ex.label - possibility, features_of_str)
    #         weights = np.add(weights, np.dot(learning_rate, gradient_of_w))
    #     loss = 0
    #     for ex in train_exs:
    #         features_of_str = feat_extractor.extract_features(ex.words, False)
    #         expo = math.exp(np.dot(weights, features_of_str))
    #         possibility = expo / (1 + expo)
    #         loss += -(ex.label * math.log(possibility) + (1 - ex.label) * math.log(1 - possibility))
    #     avrg_losses[i - 1] = loss / train_exs.__len__()
    # plt.plot(x, avrg_losses)
    #
    # # learning_rate = 0.1
    # weights = np.transpose(np.zeros(indexer.__len__(), dtype=int))
    # learning_rate = 0.1
    # avrg_losses = np.zeros(14)
    # for i in range(15):
    #     for ex in train_exs:
    #         features_of_str = feat_extractor.extract_features(ex.words, False)
    #         expo = math.exp(np.dot(weights, features_of_str))
    #         possibility = expo / (1 + expo)
    #         gradient_of_w = np.dot(ex.label - possibility, features_of_str)
    #         weights = np.add(weights, np.dot(learning_rate, gradient_of_w))
    #     loss = 0
    #     for ex in train_exs:
    #         features_of_str = feat_extractor.extract_features(ex.words, False)
    #         expo = math.exp(np.dot(weights, features_of_str))
    #         possibility = expo / (1 + expo)
    #         loss += -(ex.label * math.log(possibility) + (1 - ex.label) * math.log(1 - possibility))
    #     avrg_losses[i - 1] = loss / train_exs.__len__()
    # plt.plot(x, avrg_losses)
    # plt.xlabel('Epochs')
    # plt.ylabel('Average Training Loss')
    # plt.legend(['step size 1', 'step size 0.01', 'step size 0.1'], loc='upper left')
    # plt.show()
    # return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    indexer = Indexer()
    stop_words = set(stopwords.words('english'))
    punkt = (',', '.', '...', '?', '\'', '\'\'', '!', ':', ';')
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Generate vocabulary
        for ex in train_exs:
            for word in ex.words:
                if word.lower() not in stop_words and word.lower() not in punkt:
                    indexer.add_and_get_index(word.lower())
        feat_extractor = UnigramFeatureExtractor(indexer)
    elif args.feats == "BIGRAM":
        # Generate vocabulary
        for ex in train_exs:
            for i in range(0, len(ex.words) - 1):
                if stop_words.__contains__(ex.words[i]) and stop_words.__contains__(ex.words[i + 1]) or (
                        punkt.__contains__(ex.words[i]) or punkt.__contains__(ex.words[i + 1])):
                    continue
                bigram = ex.words[i] + ' ' + ex.words[i + 1]
                indexer.add_and_get_index(bigram.lower())
        feat_extractor = BigramFeatureExtractor(indexer)
    elif args.feats == "BETTER":
        # Generate vocabulary
        cnt = Counter()
        for ex in train_exs:
            cnt.update(
                word.lower() for word in ex.words if word.lower() not in stop_words and word.lower() not in punkt)
        cnt = dict(cnt.most_common(int(cnt.__len__() * 0.75)))
        for keys in cnt.keys():
            indexer.add_and_get_index(keys)
        feat_extractor = BetterFeatureExtractor(indexer)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model

# Methods for plotting dev acc
# Need to run model.py

# def get_dev_acc(golds: List[int], predictions: List[int]) -> float:
#     num_correct = 0
#     num_pos_correct = 0
#     num_pred = 0
#     num_gold = 0
#     num_total = 0
#     if len(golds) != len(predictions):
#         raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
#     for idx in range(0, len(golds)):
#         gold = golds[idx]
#         prediction = predictions[idx]
#         if prediction == gold:
#             num_correct += 1
#         if prediction == 1:
#             num_pred += 1
#         if gold == 1:
#             num_gold += 1
#         if prediction == 1 and gold == 1:
#             num_pos_correct += 1
#         num_total += 1
#     return float(num_correct) / num_total
#
#
# def train_better_for_plot(epoch: int, learning_r: float, train_exs: List[SentimentExample]) -> SentimentClassifier:
#     indexer = Indexer()
#     stop_words = set(stopwords.words('english'))
#     punkt = (',', '.', '...', '?', '\'', '\'\'', '!', ':', ';')
#     # Generate vocabulary
#     cnt = Counter()
#     for ex in train_exs:
#         cnt.update(
#             word.lower() for word in ex.words if word.lower() not in stop_words and word.lower() not in punkt)
#     cnt = dict(cnt.most_common(int(cnt.__len__() * 0.75)))
#     for keys in cnt.keys():
#         indexer.add_and_get_index(keys)
#     feat_extractor = BetterFeatureExtractor(indexer)
#     # train
#     indexer = feat_extractor.get_indexer()
#     weights = np.transpose(np.zeros(indexer.__len__(), dtype=int))
#     learning_rate = learning_r
#     for i in range(epoch):
#         for ex in train_exs:
#             features_of_str = feat_extractor.extract_features(ex.words, False)
#             expo = math.exp(np.dot(weights, features_of_str))
#             possibility = expo / (1 + expo)
#             gradient_of_w = np.dot(ex.label - possibility, features_of_str)
#             weights = np.add(weights, np.dot(learning_rate, gradient_of_w))
#     return LogisticRegressionClassifier(weights, feat_extractor)
#
#
# if __name__ == '__main__':
#     train_exs = read_sentiment_examples('data/train.txt')
#     dev_exs = read_sentiment_examples('data/dev.txt')
#     dev_acc_lr1 = np.zeros(14)
#     dev_acc_lr0_1 = np.zeros(14)
#     dev_acc_lr0_01 = np.zeros(14)
#     x = np.arange(0, 14)
#     for i in range(15):
#         # learning rate = 1
#         model = train_better_for_plot(i, 1, train_exs)
#         dev_acc = get_dev_acc([ex.label for ex in dev_exs], [model.predict(ex.words) for ex in dev_exs])
#         dev_acc_lr1[i - 1] = dev_acc
#         # learning rate = 0.1
#         model = train_better_for_plot(i, 0.1, train_exs)
#         dev_acc = get_dev_acc([ex.label for ex in dev_exs], [model.predict(ex.words) for ex in dev_exs])
#         dev_acc_lr0_1[i - 1] = dev_acc
#         # learning rate = 0.01
#         model = train_better_for_plot(i, 0.01, train_exs)
#         dev_acc = get_dev_acc([ex.label for ex in dev_exs], [model.predict(ex.words) for ex in dev_exs])
#         dev_acc_lr0_01[i - 1] = dev_acc
#     plt.plot(x, dev_acc_lr1)
#     plt.plot(x, dev_acc_lr0_1)
#     plt.plot(x, dev_acc_lr0_01)
#     plt.xlabel('Epochs')
#     plt.ylabel('Dev Accuracy')
#     plt.legend(['step size 1', 'step size 0.01', 'step size 0.1'], loc='upper left')
#     plt.show()
