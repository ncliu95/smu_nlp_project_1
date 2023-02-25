from nltk.corpus import inaugural
from CorpusReader_TFIDF import *

nltk.download('inaugural')
nltk.download('punkt')

print(len(inaugural.words()))
print(inaugural.sents())
print(len(inaugural.sents()))
print(inaugural.fileids())
print(inaugural.sents(['1789-washington.txt']))

myCorpus = CorpusReader_TFIDF(inaugural)
print(myCorpus.tfidf(['1789-Washington.txt']))

print("-----\n")

q = myCorpus.tfidfAll()
for x in q:
    print(x, q[x])

print("-----\n")

print(myCorpus.cosine_sim('1789-Washington.txt', '2021-Biden.txt'))

print("-----\n")

print(myCorpus.cosine_sim_new(['citizens', 'economic', 'growth', 'economic'], '2021-Biden.txt'))

print(myCorpus.words())

print(myCorpus.words(["2021-Biden.txt"]))

print(myCorpus.query(["Representatives", "Senate"]))

#  This is for testing your own corpus
#
#  create a set of text files, store them in a directory specified from 'rootDir' variable
#
#  
'''

rootDir = '/myhomedirectory'   # change that to the directory where the files are
newCorpus = PlaintextCorpusReader(rootDir, '*')
tfidfCorpus = CorpusReader_TFIDF(newCorpus)

q = tfidfCorpus.tfidfAll()
for x in q:
   print(x, q[x])

print("-----\n")

'''
