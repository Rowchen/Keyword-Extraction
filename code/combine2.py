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


tfidf1=np.load('../lsdata2/tfidf1.npy')
w2v_train_doc3=list(pd.read_csv('../data/w2v_train_doc.txt',header=None)[0].values)
w2v_train_doc4=[]
for i in range(len(w2v_train_doc3)//2):
    w2v_train_doc4.append(w2v_train_doc3[i]+w2v_train_doc3[i+len(w2v_train_doc3)//2])
print (len(w2v_train_doc4))
DF={}
for ls in tqdm(w2v_train_doc4):
    for w in set(ls.split()):
        DF[w]=DF.get(w,0)+1
IDF=[]
for row in range(len(tfidf1)):
    df={}
    for w in tfidf1[row]:
        df[w]=DF[w]
    IDF.append(df)
np.save('../lsdata2/IDF.npy',IDF)

keyword2=pd.read_csv('../data/top3.csv')
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

key_times={}
keywordlist=keyword['top3'].values
for row in tqdm(range(keywordlist.shape[0])):
    ls=keywordlist[row].split(',')
    for w in ls:
        key_times[w]=key_times.get(w,0)+1
alldoc2=alldoc.merge(keyword,on='id',how='left')
def get_keytime(tfidf,key_times):
    KT=[]
    for row in range(len(tfidf)):
        kt={}
        for w in tfidf[row]:
            kt[w]=key_times.get(w,0)
        tag=alldoc2.loc[row,'top3'].split(',')
        for w in tag:
            try:
                kt[w]-=1
            except:
                pass
        KT.append(kt)
    return KT
KT2=get_keytime(tfidf1,key_times)

tfidf2=np.load('../lsdata2/tfidf2.npy')
gc.collect()
TR1=np.load('../lsdata2/TR1.npy')
TR2=np.load('../lsdata2/TR2.npy')
gc.collect()
TE1=np.load('../lsdata2/TE1.npy')
TE2=np.load('../lsdata2/TE2.npy')
gc.collect()
FR=np.load('../lsdata2/FR.npy')
LR=np.load('../lsdata2/LR.npy')
gc.collect()
tf=np.load('../lsdata2/tf.npy')
POS_jieba=np.load('../lsdata2/POS_jieba.npy')
POS_hanlp=np.load('../lsdata2/POS_hanlp.npy')
ifxing=np.load('../lsdata2/ifxing.npy')
D2V=np.load('../lsdata2/D2V.npy')
W2V2=np.load('../lsdata2/W2V2.npy')
D2V2=np.load('../lsdata2/D2V2.npy')
LDA=np.load('../lsdata2/lda.npy')

gc.collect()
tfidf1=normalize(alldoc,tfidf1)
tfidf2=normalize(alldoc,tfidf2)
TR1=normalize(alldoc,TR1,method='sum')
TR2=normalize(alldoc,TR2,method='sum')
tf=normalize(alldoc,tf,method='sum')
D2V=normalize(alldoc,D2V,method='sum')
D2V2=normalize(alldoc,D2V2,method='sum')

def combine_feat(data,tfidf,tfidf2,TR1,TR2,TE1,TE2,FP,LP,tf,POS_jieba,POS_hanlp,ifxing,
                 KT,D2V,W2V2,D2V2,IDF,LDA,KT2,training=True):
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
            ls.append(KT[row][w])
            ls.append(D2V[row][w])
            ls.append(W2V2[row][w])
            ls.append(D2V2[row][w])
            ls.append(IDF[row][w])
            ls.append(LDA[row][w])
            ls.append(KT2[row][w])
            
            feat.append(ls)
        cnt+=1
    return feat

train_com_feat=combine_feat(alldoc,tfidf1,tfidf2,TR1,TR2,TE1,TE2,FR,LR,tf,  
                             POS_jieba,POS_hanlp,ifxing,KT,D2V,W2V2,D2V2,IDF,LDA,KT2)
train_com=pd.DataFrame(train_com_feat)
train_com.columns=['tfidf1','tfidf2','tr1','tr2','te1','te2','fr','lr','tf','jpos',
                   'hpos','word','row','tag','ifxing','kt','d2v','w2v2','d2v2','idf','lda','kt2']
train_com.iloc[0:4]


test_com_feat=combine_feat(alldoc,tfidf1,tfidf2,TR1,TR2,TE1,TE2,FR,LR,tf,
                           POS_jieba,POS_hanlp,ifxing,KT,D2V,W2V2,D2V2,IDF,LDA,KT2,training=False)
test_com=pd.DataFrame(test_com_feat)
test_com.columns=['tfidf1','tfidf2','tr1','tr2','te1','te2','fr','lr','tf','jpos','hpos','word',
                  'row','tag','ifxing','kt','d2v','w2v2','d2v2','idf','lda','kt2']
test_com.iloc[0:4]


print(train_com.columns)
print(test_com.columns)


for col in ['jpos','hpos']:
    dummies = pd.get_dummies(train_com.loc[:, col], prefix=col ) 
    train_com = pd.concat( [train_com, dummies], axis = 1 )
for col in ['jpos','hpos']:
    dummies = pd.get_dummies(test_com.loc[:, col], prefix=col ) 
    test_com = pd.concat( [test_com, dummies], axis = 1 )
    
    
import numpy as np
train_com['kt']=train_com['kt'].apply(lambda x:np.log(x+1))
test_com['kt']=test_com['kt'].apply(lambda x:np.log(x+1))
train_com['kt2']=train_com['kt2'].apply(lambda x:np.log(x+1))
test_com['kt2']=test_com['kt2'].apply(lambda x:np.log(x+1))
train_com['idf2']=train_com['idf'].apply(lambda x:(x-1)**(1/4))
test_com['idf2']=test_com['idf'].apply(lambda x:(x-1)**(1/4))

feature_list=[col for col in train_com.columns if col not in ['jpos','hpos','row','idf',
                                                              'word','tag','row2','proba']]


cvcnt=0
train_com['row2']=train_com['row']//200
for row2 in range(6):
    filter_t=(train_com['row2']!=row2)
    x_train=train_com.loc[filter_t,feature_list].values
    y_train=train_com.loc[filter_t,'tag'].values
    
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=0.0001,
                    hidden_layer_sizes=(24,), random_state=4000,early_stopping=False)
    clf.fit(x_train,y_train)

    train_com['proba']=clf.predict_proba(train_com.loc[:,feature_list])[:,1]
    test_com['proba_%d'%row2]=clf.predict_proba(test_com.loc[:,feature_list])[:,1]
    
    clf = MLPClassifier(solver='lbfgs', alpha=0.003,
                    hidden_layer_sizes=(22,), random_state=0,early_stopping=False)
    clf.fit(x_train,y_train)
    train_com['proba']+=clf.predict_proba(train_com.loc[:,feature_list])[:,1]
    test_com['proba_%d'%row2]+=clf.predict_proba(test_com.loc[:,feature_list])[:,1]
    
    clf = MLPClassifier(solver='lbfgs', alpha=0.0001,
                    hidden_layer_sizes=(22,), random_state=4000,early_stopping=False)
    clf.fit(x_train,y_train)
    train_com['proba']+=clf.predict_proba(train_com.loc[:,feature_list])[:,1]
    test_com['proba_%d'%row2]+=clf.predict_proba(test_com.loc[:,feature_list])[:,1]
    
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
        
        
id_list=alldoc['id'].values
result = pd.DataFrame()
result['id'] = id_list[test_id]
result['label1'] = label1
result['label1'] = result['label1'].replace('','nan')
result['label2'] = label2
result['label2'] = result['label2'].replace('','nan')

result.to_csv('../submit/last1216.csv',index=None)
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
 




    


































































































