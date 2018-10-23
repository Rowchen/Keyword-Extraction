import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import operator
import time

alldoc = pd.read_csv('../data/all_docs.txt',sep='\001',header=None)
alldoc.columns = ['id','title','doc']
train = pd.read_csv('../data/train_docs_keywords.txt',sep='\t',header=None)
train.columns = ['id','label']
alldoc = pd.merge(alldoc,train,on=['id'],how='left')
train_idx=list(alldoc[~alldoc['label'].isnull()].index)
test_idx=list(alldoc[alldoc['label'].isnull()].index)
import gc
gc.collect()

def normalize(data,fixdata,method='mean'):
    for row in tqdm(range(data.shape[0])):
        tmp=np.array(list(fixdata[row].values()))
        if method=='mean':
            mean=np.mean(tmp)
            std=np.std(tmp)+1e-7
            tmp=(tmp-mean)/(std)
        if method=='sum':
            sums=np.sum(tmp)+1e-7
            tmp=tmp/sums
        fixdata[row]=dict(zip(list(fixdata[row].keys()),tmp))
    return fixdata

tfidf1=np.load('../lsdata/tfidf1.npy')
tfidf2=np.load('../lsdata/tfidf2.npy')
TR1=np.load('../lsdata/TR1.npy')
TR2=np.load('../lsdata/TR2.npy')
tf=np.load('../lsdata/tf.npy')
ifxing=np.load('../lsdata/ifxing.npy')
TE1=np.load('../lsdata/TE1.npy')
TE2=np.load('../lsdata/TE2.npy')
FR=np.load('../lsdata/FR.npy')
LR=np.load('../lsdata/LR.npy')
POS_jieba=np.load('../lsdata/POS_jieba.npy')
POS_hanlp=np.load('../lsdata/POS_hanlp.npy')
D2V2=np.load('../lsdata/D2V2.npy')
W2V2=np.load('../lsdata/W2V2.npy')
gc.collect()

tfidf1=normalize(alldoc,tfidf1)
tfidf2=normalize(alldoc,tfidf2)
TR1=normalize(alldoc,TR1,method='sum')
TR2=normalize(alldoc,TR2,method='sum')
tf=normalize(alldoc,tf,method='sum')
D2V2=normalize(alldoc,D2V2,method='sum')


def combine_feat(data,tfidf,tfidf2,TR1,TR2,TE1,TE2,FP,LP,tf,POS_jieba,POS_hanlp,ifxing,W2V2,D2V2,training=True,KT=None):
    feat=[]
    cnt=0
    for row in tqdm(range(data.shape[0])):
        if training:
            if row in test_idx:
                continue
            tag=data.loc[row,'label'].split(',')
        else:
            if row in train_idx:
                continue
        for w in tfidf[row]:
            ls=[]
            ls.append(tfidf[row][w])
            ls.append(tfidf2[row][w])
            ls.append(TR1[row][w])
            ls.append(TR2[row][w])
            ls.append(TE1[row][w])
            ls.append(TE2[row][w])
            ls.append(FP[row][w])
            ls.append(LP[row][w])
            ls.append(tf[row][w])
            ls.append(POS_jieba[row][w])
            ls.append(POS_hanlp[row][w])
            ls.append(w)
            ls.append(cnt)
            if training:
                if w in tag:
                    ls.append(1)
                else:
                    ls.append(0)
            else:
                ls.append(0)
            ls.append(ifxing[row][w])
            ls.append(W2V2[row][w])
            ls.append(D2V2[row][w])
            
            if KT is not None:
                ls.append(KT[row][w])
                
            feat.append(ls)
        cnt+=1
    return feat

train_com_feat=combine_feat(alldoc,tfidf1,tfidf2,TR1,TR2,TE1,TE2,FR,LR,tf,POS_jieba,POS_hanlp,
                            ifxing,W2V2,D2V2,training=True)
train_com=pd.DataFrame(train_com_feat)
train_com.columns=['tfidf1','tfidf2','tr1','tr2','te1','te2','fr','lr','tf','jpos','hpos',
                   'word','row','tag','ifxing','w2v','d2v']


test_com_feat=combine_feat(alldoc,tfidf1,tfidf2,TR1,TR2,TE1,TE2,FR,LR,tf,POS_jieba,POS_hanlp,
                           ifxing,W2V2,D2V2,training=False)
test_com=pd.DataFrame(test_com_feat)
test_com.columns=['tfidf1','tfidf2','tr1','tr2','te1','te2','fr','lr','tf','jpos','hpos',
                  'word','row','tag','ifxing','w2v','d2v']


for col in ['jpos','hpos']:
    dummies = pd.get_dummies(train_com.loc[:, col], prefix=col ) 
    train_com = pd.concat( [train_com, dummies], axis = 1 )
for col in ['jpos','hpos']:
    dummies = pd.get_dummies(test_com.loc[:, col], prefix=col ) 
    test_com = pd.concat( [test_com, dummies], axis = 1 )
feature_list=[col for col in train_com.columns if col not in ['jpos','hpos','row','word','tag','row2','proba']]

cvcnt=0
train_com['row2']=train_com['row']//200
for row2 in range(6):
    filter_t=(train_com['row2']!=row2)
    x_train=train_com.loc[filter_t,feature_list].values
    y_train=train_com.loc[filter_t,'tag'].values
    
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression(random_state=2018,penalty='l2',C=1)
    clf.fit(x_train,y_train)

    train_com['proba']=clf.predict_proba(train_com.loc[:,feature_list])[:,1]
    test_com['proba_%d'%row2]=clf.predict_proba(test_com.loc[:,feature_list])[:,1]
    
    hit_num=0
    for row in range(0,1000):
        if (row//200)!=row2:
            continue
        tag=alldoc.loc[train_idx[row],'label'].split(',')[:2]
        ls=train_com.loc[train_com['row']==row,['proba','word']]
        ls.sort_values('proba',inplace=True,ascending=False)
        words=list(ls.iloc[:2]['word'].values)
        for w in words:
            if w in tag:
                hit_num+=1
    cvcnt+=hit_num
    print (x_train.shape[0],hit_num)
print (cvcnt)

test_com['proba']=0.0
for row2 in range(6):
    test_com['proba']+=test_com['proba_%d'%row2]
    

from tqdm import tqdm
label1=[]
label2=[]
test_id=[]
allkeylist=[]
for row in tqdm(range(len(test_idx))):
    ls=test_com.loc[test_com['row']==row,['proba','word']]
    ls.sort_values('proba',inplace=True,ascending=False)
    words=list(ls.iloc[:2]['word'].values)
    keylist=list(ls.iloc[:3]['word'].values)
    allkeylist.append(keylist)
    
    test_id.append(test_idx[row])
    if(len(words)==2):
        label1.append(words[0])
        label2.append(words[1])
    elif (len(words)==1):
        label1.append(words[0])
        label2.append(words[0])
    else:
        label1.append('')
        label2.append('')
        
        
allkeylist2=[]
for se in allkeylist:
    try: 
        allkeylist2.append(",".join(se))
    except:
        ls=[]
        for w in se:
            if w is not None:
                ls.append(str(w))
        allkeylist2.append(",".join(ls))
        print (",".join(ls))
        

id_list=alldoc['id'].values
keyword2 = pd.DataFrame()
keyword2['id'] = id_list[test_id]
keyword2['top3'] = allkeylist2
keyword2.to_csv('../data/top3.csv',index=None)

keyword1=pd.read_csv('../data/train_docs_keywords.txt',sep='\t',header=None)
keyword1.columns=['id','top3']
keyword=pd.concat([keyword1,keyword2],axis=0).reset_index(drop=True)

key_times={}
keywordlist=keyword['top3'].values
for row in tqdm(range(keywordlist.shape[0])):
    ls=keywordlist[row].split(',')[:2]
    for w in ls:
        key_times[w]=key_times.get(w,0)+1
        
alldoc2=alldoc.merge(keyword,on='id',how='left')
def get_keytime(tfidf,key_times):
    KT=[]
    for row in range(len(tfidf)):
        kt={}
        for w in tfidf[row]:
            kt[w]=key_times.get(w,0)
        tag=alldoc2.loc[row,'top3'].split(',')[:2]
        for w in tag:
            try:
                kt[w]-=1
            except:
                pass
        KT.append(kt)
    return KT

KT=get_keytime(tfidf1,key_times)
np.save('../lsdata/KT.npy',KT)

train_com_feat=combine_feat(alldoc,tfidf1,tfidf2,TR1,TR2,TE1,TE2,FR,LR,tf,POS_jieba,POS_hanlp,
                            ifxing,W2V2,D2V2,training=True,KT=KT)
train_com=pd.DataFrame(train_com_feat)
train_com.columns=['tfidf1','tfidf2','tr1','tr2','te1','te2','fr','lr','tf','jpos','hpos',
                   'word','row','tag','ifxing','w2v','d2v','kt']


test_com_feat=combine_feat(alldoc,tfidf1,tfidf2,TR1,TR2,TE1,TE2,FR,LR,tf,POS_jieba,POS_hanlp,
                           ifxing,W2V2,D2V2,training=False,KT=KT)
test_com=pd.DataFrame(test_com_feat)
test_com.columns=['tfidf1','tfidf2','tr1','tr2','te1','te2','fr','lr','tf','jpos','hpos',
                  'word','row','tag','ifxing','w2v','d2v','kt']


train_com['kt']=train_com['kt'].apply(lambda x:np.log(x+1))
test_com['kt']=test_com['kt'].apply(lambda x:np.log(x+1))
for col in ['jpos','hpos']:
    dummies = pd.get_dummies(train_com.loc[:, col], prefix=col ) 
    train_com = pd.concat( [train_com, dummies], axis = 1 )
for col in ['jpos','hpos']:
    dummies = pd.get_dummies(test_com.loc[:, col], prefix=col ) 
    test_com = pd.concat( [test_com, dummies], axis = 1 )
feature_list=[col for col in train_com.columns if col not in ['jpos','hpos','row','word','tag','row2','proba']]
print (feature_list)

cvcnt=0
train_com['row2']=train_com['row']//200
for row2 in range(6):
    filter_t=(train_com['row2']!=row2)
    x_train=train_com.loc[filter_t,feature_list].values
    y_train=train_com.loc[filter_t,'tag'].values
    
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression(random_state=2018,penalty='l2',C=1)
    clf.fit(x_train,y_train)

    train_com['proba']=clf.predict_proba(train_com.loc[:,feature_list])[:,1]
    test_com['proba_%d'%row2]=clf.predict_proba(test_com.loc[:,feature_list])[:,1]
    
    hit_num=0
    for row in range(0,1000):
        if (row//200)!=row2:
            continue
        tag=alldoc.loc[train_idx[row],'label'].split(',')[:2]
        ls=train_com.loc[train_com['row']==row,['proba','word']]
        ls.sort_values('proba',inplace=True,ascending=False)
        words=list(ls.iloc[:2]['word'].values)
        for w in words:
            if w in tag:
                hit_num+=1
    cvcnt+=hit_num
    print (x_train.shape[0],hit_num)
print (cvcnt)

test_com['proba']=0.0
for row2 in range(6):
    test_com['proba']+=test_com['proba_%d'%row2]
    

from tqdm import tqdm
label1=[]
label2=[]
test_id=[]
allkeylist=[]
for row in tqdm(range(len(test_idx))):
    ls=test_com.loc[test_com['row']==row,['proba','word']]
    ls.sort_values('proba',inplace=True,ascending=False)
    words=list(ls.iloc[:2]['word'].values)
    keylist=list(ls.iloc[:3]['word'].values)
    allkeylist.append(keylist)
    
    test_id.append(test_idx[row])
    if(len(words)==2):
        label1.append(words[0])
        label2.append(words[1])
    elif (len(words)==1):
        label1.append(words[0])
        label2.append(words[0])
    else:
        label1.append('')
        label2.append('')
        
        
allkeylist2=[]
for se in allkeylist:
    try: 
        allkeylist2.append(",".join(se))
    except:
        ls=[]
        for w in se:
            if w is not None:
                ls.append(str(w))
        allkeylist2.append(",".join(ls))
        print (",".join(ls))
        

id_list=alldoc['id'].values
keyword2 = pd.DataFrame()
keyword2['id'] = id_list[test_id]
keyword2['top3'] = allkeylist2
keyword2.to_csv('../data/top3.csv',index=None)




















