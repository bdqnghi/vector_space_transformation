import gensim
import os
import codecs
from sklearn.manifold import TSNE
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec, LineSentence 
import numpy as np
# import matplotlib.pyplot as plt

CURRENT_DIR = os.getcwd()

def load_embeddings(file_name):
	with codecs.open(file_name,"r","utf-8") as f:
		next(f)
		vocabulary, wv = zip(*[line.strip().split(" ",1) for line in f])
	wv = np.loadtxt(wv)
	return wv,vocabulary


code_sentences = list()
with open("new_sentence.txt","r") as f:
	data = f.readlines()
	for line in data:
		code_sentences.append(line.rstrip().split(" "))


model = Word2Vec.load("oldmodel")

model.build_vocab(code_sentences,update=True)
model.train(code_sentences,total_examples=model.corpus_count,epochs=model.iter)
model.save("newmodel")
try:
	print(model.most_similar("function"))
except KeyError as e:
	print(e)

old_model = Word2Vec.load("oldmodel")
new_model = Word2Vec.load("newmodel")
print(len(old_model.wv.vocab))
print(len(new_model.wv.vocab))