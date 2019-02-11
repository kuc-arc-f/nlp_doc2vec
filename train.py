# encoding: utf-8
# 2019/02/11 16:27:学習処理の修正。
#
#
#import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer
import numpy as np
#
def tokenize(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    word = []
    for token in tokens:
        part_of_speech = token.part_of_speech.split(",")[0]
        if part_of_speech == "名詞":
            word.append(token.surface)
        if part_of_speech == "動詞":
            word.append(token.base_form)
        if part_of_speech == "形容詞":
            word.append(token.base_form)
        if part_of_speech == "形容動詞":
            word.append(token.base_form)        
    return word

#
words1=tokenize("女ははじめて関西線で四日市の方へ行くのだということを三四郎に話した。")
words2=tokenize("例の女はすうと立って三四郎の横を通り越して車室の外へ出て行った。")
words3=tokenize("けれども暑い時分だから町はまだ宵の口のようににぎやかだ。")
print(1, words1 )
print(2, words2 )
print(3, words3 )
#quit()
#
training_docs = []
#quit()
sent1 = TaggedDocument(words=words1 ,  tags=[1])
sent2 = TaggedDocument(words=words2 ,  tags=[2])
sent3 = TaggedDocument(words=words3 ,  tags=[3])

training_docs.append(sent1)
training_docs.append(sent2)
training_docs.append(sent3)
#quit()

model = Doc2Vec(documents=training_docs, dm=1,
                vector_size=300, window=8, min_count=1, workers=4)
#model = Doc2Vec(documents=training_docs, dm=1,
#                size=100, window=8, min_count=1, workers=4)

model.train(training_docs, total_examples=model.corpus_count, epochs=50)
model.save("./book.model")
#print(model.docvecs[1])
#print(model.docvecs.most_similar(1) )
#quit()
#cos類似度
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#    return 0
#
ret2 =cos_sim(model.docvecs[1], model.docvecs[2])
ret3 =cos_sim(model.docvecs[1], model.docvecs[3])
print("1-2:", ret2 )
print("1-3:", ret3 )
