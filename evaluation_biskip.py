import gensim
import os
import codecs
from sklearn.manifold import TSNE
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

cs_vectors = KeyedVectors.load_word2vec_format("./embeddings/cs_biskip.txt",binary=False)
java_vectors = KeyedVectors.load_word2vec_format("./embeddings/java_biskip.txt",binary=False)

print(cs_vectors.similar_by_vector(java_vectors["int"], topn=30))
# print cs_vectors.similar_by_vector(java_vectors["package"], topn=30)