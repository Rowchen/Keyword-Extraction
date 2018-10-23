import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
import re
import jieba.analyse
import gensim
from gensim.models import word2vec
TaggededDocument = gensim.models.doc2vec.TaggedDocument
from gensim.models import doc2vec

alldoc = pd.read_csv('../data/all_docs.txt',sep='\001',header=None)
alldoc.columns = ['id','title','doc']

train2 = pd.read_csv('../data/train_docs_keywords.txt',sep='\t',header=None)
train2.columns = ['id','label']
alldoc = pd.merge(alldoc,train2,on=['id'],how='left')

train_idx=list(alldoc[~alldoc['label'].isnull()].index)
test_idx=list(alldoc[alldoc['label'].isnull()].index)

train=alldoc.iloc[train_idx].copy().reset_index(drop=True)

tf=np.load('../lsdata/tf.npy')


w2v_train_doc=list(pd.read_csv('../data/w2v_train_doc.txt',header=None)[0].values)

print (len(w2v_train_doc))
w2v_train_doc2=[]
for i in range(len(w2v_train_doc)//2):
    w2v_train_doc2.append(w2v_train_doc[i]+w2v_train_doc[i+len(w2v_train_doc)//2])
print (len(w2v_train_doc2))

print ('now train word2vec')


import time
a=time.time()
class sentences_generator():
    def __init__(self, doc):
        self.doc = doc
    def __iter__(self):
        for i,line in enumerate(self.doc):
            sentence = TaggededDocument(line.split(), tags=[i])
            yield sentence
sents=sentences_generator(w2v_train_doc2)
model = doc2vec.Doc2Vec(sents,dm=0,vector_size=300,window=13,min_count=1,hs=1,workers=12,epochs=30,
                        dbow_words=1)
b=time.time()
print (b-a)

model.save("../data/d2v_dbow.txt")

#model=gensim.models.Doc2Vec.load("../data/d2v_dbow.txt")


def predict_proba(oword, iword):
    if oword==iword:
        return 0
    iword_vec = model[iword]#获取输入词的词向量
    oword = model.wv.vocab[oword]
    oword_l = model.trainables.syn1[oword.point].T
    dot = np.dot(iword_vec, oword_l)
    lprob = -sum(np.logaddexp(0, -dot) + oword.code*dot) 
    return lprob


# import operator
def get_doc2vec_score2(data,tf):
    DS=[]
    for row in tqdm(range(data.shape[0])):
        d2v_score={}
        doc_vec=model[row]
        doc_vec/=np.sqrt(sum(doc_vec**2))
        for w in tf[row]:
            try:
                ls=model.wv.word_vec(w)
            except:
                continue
            d2v_score[w]=np.sum(ls*doc_vec)/np.sqrt(sum(ls**2))
        DS.append(d2v_score)
    return DS
D2V=get_doc2vec_score2(alldoc,tf)
np.save('../lsdata/D2V2.npy',D2V)

import operator
def get_word2vec_score2(data,tf):
    WS=[]
    for row in tqdm(range(data.shape[0])):
        w2v_score={}
        for w in tf[row]:
            w2v_score[w]=sum([predict_proba(u, w)*tf[row][u] for u in tf[row]])/sum([tf[row][u] for u in tf[row]])
        WS.append(w2v_score)
    return WS
W2V=get_word2vec_score2(alldoc,tf)
np.save('../lsdata/W2V2.npy',W2V)



















