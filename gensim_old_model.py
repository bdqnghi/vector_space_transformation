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
with open("SENTENCE/cs_sentences.txt","r") as f:
	data = f.readlines()
	for line in data:
		code_sentences.append(line.rstrip().split(" "))

print("Building Word2Vec model.......")
model = gensim.models.Word2Vec(code_sentences,min_count=1,size=100,workers=20,iter=20,sg=1,window=7)

model.wv.save_word2vec_format("embeddings/cs_vectors_all_tokens.txt",fvocab=None,binary=False)
# model.save("oldmodel")
# try:
# 	print(model.most_similar("function"))
# except KeyError as e:
# 	print(e)



# model = KeyedVectors.load_word2vec_format("gensim_small_sentence.txt",binary=False)
# model = gensim.models.Word2Vec.load_word2vec_format("gensim_small_sentence.txt")

# model.build_vocab(code_sentences,update=True)
# model.train(code_sentences)
# model.save("gensim_new_model.txt")



# print model.similarity("private","public")