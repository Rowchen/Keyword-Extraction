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
tf=np.load('../lsdata2/tf.npy')

w2v_train_doc=list(pd.read_csv('../data/w2v_train_doc.txt',header=None)[0].values)

print (len(w2v_train_doc))
w2v_train_doc2=[]
for i in range(len(w2v_train_doc)//2):
    w2v_train_doc2.append(w2v_train_doc[i]+w2v_train_doc[i+len(w2v_train_doc)//2])
print (len(w2v_train_doc2))

w2v_train_doc=w2v_train_doc2
import gensim
train_doc=[]
for i,ls in enumerate(w2v_train_doc):
    tmp=ls.split()
    tmp2=[]
    for w in tmp:
        if w in tf[i]:
            tmp2.append(w)
    train_doc.append(tmp2)
    
word_count_dict = gensim.corpora.Dictionary(train_doc)
bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in train_doc]

import gc
gc.collect()

lda_model=gensim.models.ldamulticore.LdaMulticore(corpus=bag_of_words_corpus, num_topics=250, id2word=word_count_dict, workers=10, chunksize=2000, passes=30, alpha='symmetric', eta=None, decay=0.6, offset=5.0, eval_every=0, iterations=50, gamma_threshold=0.001, random_state=None)

import operator
def get_lda_score(data,train_doc):
    LS=[]
    for row in tqdm(range(data.shape[0])):
        ls_score={}
        _,_,docterm=lda_model.get_document_topics(word_count_dict.doc2bow(train_doc[row]),
                                                  per_word_topics=True)
        for i in docterm:
            ls_score[word_count_dict.id2token[i[0]]]=i[1][0][1]
        LS.append(ls_score)
    return LS

xxx=get_lda_score(alldoc,train_doc)
np.save('../lsdata2/lda.npy',xxx)




















