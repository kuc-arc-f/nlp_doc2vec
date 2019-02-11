# encoding: utf-8
from gensim.models import word2vec
import numpy as np

model = word2vec.Word2Vec.load("./book.model")
#
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

ret2 =cos_sim(model.docvecs[1], model.docvecs[2])
ret3 =cos_sim(model.docvecs[1], model.docvecs[3])
print("1-2:", ret2 )
print("1-3:", ret3 )
