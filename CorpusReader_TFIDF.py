import nltk
import math
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict


class CorpusReader_TFIDF:
    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=False, ignoreCase=True):
        self._corpus = corpus
        self._tf = tf
        self._idf = idf
        self.stopwords = set()
        self.toStem = toStem
        self.ignoreCase = ignoreCase

        if stopWord == "standard":
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))
        elif stopWord != "none":
            with open(stopWord, 'r') as f:
                for line in f:
                    self.stopwords.add(line.strip())

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

    #  Returns dictionary of idf for each word in each doc
    def get_idf(self) -> dict:
        doc_freq = defaultdict(int)
        # Preprocessing
        for doc in self._corpus.fileids():
            if self.toStem:
                stemmer = SnowballStemmer("english")
                doc = [stemmer.stem(word) for word in self._corpus.words(doc)]
            for word in set(self._corpus.words(doc)):
                if self.ignoreCase:
                    word = word.lower()
                if word not in self.stopwords:
                    doc_freq[word] += 1

        if self._idf == "base":
            return {word: math.log(len(self._corpus.fileids()) / doc_freq[word]) for word in doc_freq}
        elif self._idf == "smooth":
            return {word: math.log(1 + len(self._corpus.fileids()) / (1 + doc_freq[word])) for word in doc_freq}

    #  Returns dictionary of tfidf for each word in doc    
    def get_tfidf(self, doc, return_zero=False) -> dict:
        tf = self.get_tf(doc)
        idf = self.get_idf()
        tfidf = {}
        for term, freq in tf.items():
            tfidf[term] = freq * idf[term]
        if return_zero:
            for term in doc:
                if term not in tfidf:
                    tfidf[term] = 0
        return tfidf

    #  Returns dictionary of tfidf dicts, one for each doc
    def get_tfidf_all_docs(self, returnZero=False) -> dict[dict]:
        tfidf_all_docs = defaultdict(dict)
        for doc in set(self._corpus.fileids()):
            tfidf = self.get_tfidf(list(self._corpus.words(doc)), return_zero=returnZero)
            tfidf_all_docs[doc] = tfidf
        return tfidf_all_docs

    #  Specific Methods
    def tfidf(self, fileid, returnZero=False) -> dict:
        """
        Returns the TF-IDF for the specific document in the corpus (specified by fileid). The vector is represented
        by a dictionary/hash in python. The keys are the terms, and the values are the tf-idf value of the dimension.
        If returnZero is true, then the dictionary will contain terms that have 0 value for that vector,
        otherwise the vector will omit those terms.
        :param fileid: A file ID contained within the object's corpus
        :param returnZero: Boolean to control omission of terms that have 0 value for the given vector
        :return: A dictionary of TF-IDF values for each word in the document
        """
        doc = list(self._corpus.words(fileid))
        return self.get_tfidf(doc, return_zero=returnZero)

    def tfidfAll(self, returnZero=False) -> dict[dict]:
        """
        Returns the TF-IDF for all documents in the corpus as a dictionary. The key is the fileid of each document,
        for each document the value is the tfidf of that.
        document (using the same format as above).
        :param returnZero: Boolean to control omission of terms that have 0 value for the given vector
        :return: A dictionary of dictionaries of TF-IDF values, one for each document
        """
        return self.get_tfidf_all_docs(returnZero)

    def tfidfNew(self, words) -> dict:
        """
        Returns the tf-idf of a “new” document, represented by a list of words. Honors the various parameters
        (ignoreCase, toStem etc.) of the object when preprocessing the new document.
        Also, the idf of each word is static (i.e. the “new” document is not treated as part of the corpus).
        :param words: A list of words constituting a new/simulated document
        return: A dictionary of TF-IDF values for each word in the simulated document
        """
        return self.get_tfidf(words, return_zero=False)

    def idf(self) -> dict:
        """
        :return: The idf of each term as a dictionary : keys are the terms, and values are the idf
        """
        return self.get_idf()

    def cosine_sim(self, fileid1, fileid2) -> float:
        """
        :param fileid1: A file ID contained within the object's corpus
        :param fileid2: A file ID contained within the object's corpus
        :return: The cosine similarity between two documents in the corpus
        """
        return self.cosine_sim_new(list(self._corpus.words(fileid1)), fileid2)

    def cosine_sim_new(self, words, fileid) -> float:
        """
        :param words: A list of words constituting a new/simulated document
        :param fileid: A file ID contained within the object's corpus
        :return: the cosine similarity between a “new” document (as if
        specified like the tfidf_new() method) and the documents specified by fileid. 
        """

        # Preprocessing
        all_words = set(words + list(self._corpus.words(fileid)))
        if self.toStem:
            stemmer = SnowballStemmer("english")
            all_words = [stemmer.stem(word) for word in all_words]
        if self.ignoreCase:
            all_words = [word.lower() for word in all_words]
        if len(self.stopwords):
            all_words = [word for word in all_words if word not in self.stopwords]

        numerator = 0
        doc1_tf = self.get_tf(words)
        doc2_tf = self.get_tf(self._corpus.words(fileid))
        # A * B
        for word in all_words:
            numerator = numerator + (doc1_tf[word] * doc2_tf[word])
        # Substituting the formal definition sqrt(X**2) for abs(X) since they do the same thing
        denominator = abs(sum(doc1_tf.values())) * abs(sum(doc2_tf.values()))
        return numerator / denominator

    # Bonus Method
    def query(self, words) -> list[tuple[str, float]]:
        """
        :param words: A list of words constituting a new/simulated document
        :return: A list of (document, cosine_sim) tuples comparing each document against words, sorted descending
        against cosine_sim
        """
        query_results = []
        for doc in self.fileids():
            query_results.append((doc, self.cosine_sim_new(words, doc)))
        query_results.sort(key=lambda tup: tup[1], reverse=True)
        return query_results

    #  Shared Methods
    def fileids(self) -> list:
        """
        :return: A list of file identifiers for the files that make up this corpus.
        """
        return list(self._corpus.fileids())

    def raw(self, fileids=None) -> list:
        """
        :param fileids: A list of file IDs contained within the object's corpus
        :return: The concatenation of the raw text of the specified file(s)
        """
        return list(self._corpus.raw(fileids))

    def words(self, fileids=None) -> list:
        """
        :param fileids: A list of file IDs contained within the object's corpus
        :return: The list containing the words in the specified file(s).
        """
        return list(self._corpus.words(fileids))
