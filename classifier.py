import pickle
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
import re

class classifier_engine:
    def __init__(self):
        self.corpus = None
        self.tags = None
        self.lexicons = None
        self.vectorizer = None
        self.model = None
        self.metrics = None
    
    # Takes a list of strings and converts to lowercase, removes non alphabetic characters, removes stopwords and stems each word
    def _simplify(self, corpus):
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        def clean(text):
            text = re.sub("[^a-zA-Z]", " ", text)
            words = [stemmer.stem(w) for w in word_tokenize(text.lower()) if w not in stop_words] 
            return " ".join(words)
        return [clean(text) for text in corpus]

    # Trains a model using Bag of Words
    def train_using_bow(self):
        corpus = self._simplify(self.corpus)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)

        bag_of_words = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(bag_of_words, self.tags, test_size=0.2, stratify=self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)

    def train_using_tfidf(self):
        corpus = self._simplify(self.corpus)
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)

        word_vectors = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(word_vectors, self.tags, test_size=0.2, stratify=self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)

    class CustomVectorizer:
        # Extracts features from text and vectorizes them
        def __init__(self, lexicons):
            self.lexicons = lexicons

        # Returns a numpy array of word vectors
        def transform(self, corpus):
            word_vectors = []
            for text in corpus:
                features = []
                for _, v in self.lexicons.items():
                    features.append(len([w for w in word_tokenize(text) if w in v]))
                word_vectors.append(features)
            return numpy.array(word_vectors)

    # returns the set containing every word in file in path
    def _get_lexicon(self, path):
        words = set()
        with open(path) as file:
            for line in file:
                words.update(line.strip().split(" "))
        return words


    def train_using_custom(self):
        corpus = self._simplify(self.corpus)
        self.vectorizer = self.CustomVectorizer(self.lexicons)
        
        word_vectors = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(word_vectors, self.tags, test_size=0.2, stratify=self.tags)

        self.model = SVC(gamma = "auto")
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)
    
    def predict(self, corpus):
        x = self.vectorizer.transform(self._simplify(corpus))
        return self.model.predict(x)

    # Take data and returns a dictionary of metrics
    def _model_metrics(self, features, tags):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        predictions = self.model.predict(features)
        for r in zip(predictions, tags):
            if r[0] == 1 and r[1] == 1:
                tp += 1
            elif r[0] == 1 and r[1] == 0:
                fp += 1
            elif r[0] == 0 and r[1] == 1:
                fn += 1
            else:
                tn += 1
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return {
            'precision': precision,
            'recall': recall,
            'f1': (2 * precision * recall) / (precision + recall)
        }

    def evaluate(self):
        return self.metrics

    # Takes pandas dataframe and extracts tagged corpus
    def load_corpus(self, path, corpus_col, tag_col):
        data = pandas.read_pickle(path)[[corpus_col, tag_col]].values
        self.corpus = [row[0] for row in data]
        self.tags = [row[1] for row in data]

    # Loads words from a file
    def load_lexicon(self, fname):
        if self.lexicons is None:
            self.lexicons = {}
        
        self.lexicons[fname] = self._get_lexicon('./data/' + fname + '.txt')

    # loads previously created model
    def load_model(self, model_name):
        self.model = pickle.load(open('./models/' + model_name + '_ml_model.pkl', 'rb'))
        self.vectorizer = pickle.load(open('./models/' + model_name + '_vectorizer.pkl', 'rb'))
        self.metrics = pickle.load(open('./models/' + model_name + '_metrics.pkl', 'rb'))

    def save_model(self, model_name):
        pickle.dump(self.model, open('./models/' + model_name + '_ml_model.pkl', 'wb'))
        pickle.dump(self.vectorizer, open('./models/' + model_name + '_vectorizer.pkl', 'wb'))
        pickle.dump(self.metrics, open('./models/' + model_name + '_metrics.pkl', 'wb'))
