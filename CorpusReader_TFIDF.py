import nltk
import math
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict

class CorpusReader_TFIDF():
    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=False, ignoreCase=True):
        self._corpus = corpus
        self._tf = tf
        self._idf = idf
        self.stopwords = set()
        self.toStem = toStem
        self.ignoreCase = ignoreCase

        if stopWord == "standard":
            self.stopwords = set(stopwords.words('english'))
        elif stopWord != "none":
            with open(stopWord, 'r') as f:
                for line in f:
                    self.stopwords.add(line.strip())

    #  Returns dictionary of tf for each word in doc
    def get_tf(self, doc) -> dict:
        tf = defaultdict(int)
        if self.toStem:
            stemmer = SnowballStemmer("english")
            doc = [stemmer.stem(word) for word in doc]
        for word in doc:
            if self.ignoreCase:
                word = word.lower()
            if word not in self.stopwords:
                tf[word] += 1
        if self._tf == "raw":
            return tf
        elif self._tf == "log":
            return {word: 1 + math.log(tf[word]) if tf[word] > 0 else 0 for word in tf}

    #  Returns dictionary of idf for each word in doc
    def get_idf(self) -> dict:
        doc_freq = defaultdict(int)
        for doc in self._corpus:
            if self.toStem:
                stemmer = SnowballStemmer("english")
                doc = [stemmer.stem(word) for word in doc]
            for word in set(doc):
                if self.ignoreCase:
                    word = word.lower()
                if word not in self.stopwords:
                    doc_freq[word] += 1
        if self._idf == "base":
            return {word: math.log(len(self._corpus) / doc_freq[word]) for word in doc_freq}
        elif self._idf == "smooth":
            return {word: math.log(1 + len(self._corpus) / (1 + doc_freq[word])) for word in doc_freq}

    #  Returns dictionary of tfidf for each word in doc
    def get_tfidf(self, doc) -> dict:
        tf = self.get_tf(doc)
        idf = self.get_idf()
        tfidf = {word: tf[word] * idf[word] for word in tf.keys() & idf.keys()}
        return tfidf

    #  Returns array of tfidf, one for each doc
    def get_tfidf_all_docs(self):
        tfidf_all_docs = []
        for doc in self.corpus:
            tfidf = self.get_tfidf(doc)
            tfidf_all_docs.append(tfidf)
        return tfidf_all_docs

    #  Specific Methods
    def tfidf(self, fileid, returnZero=False):
        """
        Return the TF-IDF for the specific document in the corpus (specified 
        by fileid). The vector is represented by a dictionary/hash in python. The keys are the terms, and the 
        values are the tf-idf value of the dimension. If returnZero is true, then the dictionary will contains 
        terms that have 0 value for that vector, otherwise the vector will omit those terms
        """
        pass

    def tfidfAll(self, returnZero=False):
        """
        Return the TF-IDF for all documents in the corpus. It will be returned as a 
        dictionary. The key is the fileid of each document, for each document the value is the tfidf of that 
        document (using the same format as above). 
        """
        pass

    def tfidfNew(self, words):
        """
        return the tf-idf of a “new” document, represented by a list of words. You should 
        honor the various parameters (ignoreCase, toStem etc.) when preprocessing the new document. 
        Also, the idf of each word should not be changed (i.e. the “new” document should not be treated as 
        part of the corpus). 
        """
        pass

    def idf(self):
        """
        Return the idf of each term as a dictionary : keys are the terms, and values are the idf
        """
        pass

    def cosine_sim(self, fileid1, fileid2):
        """
        Return the cosine similarity between two documents in the corpus
        """
        pass

    def cosine_sim_new(self, words, fileid):
        """
        return the cosine similary between a “new” document (as if 
        specified like the tfidf_new() method) and the documents specified by fileid. 
        """
        pass

    #  Shared Methods
    def fileids(self):
        """
        Return a list of file identifiers for the files that make up this corpus.
        """
        return self._corpus.fileids()
    def raw(self, fileids=None):
        """
        Returns the concatenation of the raw text of the specified files, if specified
        """
        return self._corpus.raw(fileids)

    def words(self, fileids=None):
        """
        Returns the words in the specified file(s).
        """
        return self._corpus.words(fileids)
