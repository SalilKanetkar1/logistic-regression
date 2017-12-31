'''
tokenize the train data
'''
from __future__ import division
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import codecs
import numpy as np
from sklearn.model_selection import KFold



train=open(r"tokenized.txt","r")
cat=open(r"cat.txt","r")
v1=train.readlines()
v=v1[0:len(v1)-1]
token=[]
for h in range(len(v)):
    token=token+RegexpTokenizer(r'\w+').tokenize(v[h])
'''
t=codecs.open(r"tokenized1.txt","w","utf-8")
t1=open(r"cat1.txt","w")
u=[]

for k in range(1,len(v)):
    print k
    y=v[k].split(" ")
    u.append(" ".join([word for word in y if word not in stopwords.words('english')]).strip("\n").strip(",,,"))
    t1.write(y[0].split(",")[0]+"\n")
    t.write(u[k-1])
'''

corpus=[]
for t in range(len(v)):
    corpus.append(v[t].strip("\n"))
dictionary=set(token)
cv =CountVectorizer(dictionary)

X=cv.fit_transform(corpus)
'''
transformer=TfidfTransformer(smooth_idf=False,norm='l2')
X=transformer.fit_transform(X)
pca=PCA(n_components=724,whiten=True)
X_pca=pca.fit_transform(X.toarray())
eigvectors=pca.components_
np.save("X_pca",X_pca)
np.save("eigenvectors",eigvectors)
'''
X_pca=np.load("X_pca.npy")
eigvectors=np.load("eigvectors.npy")


'''
logistic regression
'''

categories=cat.readlines()
classe=[]
for k in range(len(categories)-1):
    classe.append(categories[k].strip("\n"))
classes1=[]
for k in range(len(categories)-1):
    if classe[k]=='ham':
        classes1.append(0)
    else:
        classes1.append(1)
kf = KFold(n_splits=10)
kf.get_n_splits(X)
lamb=[0.001,0.01,0.1,1,10]
wt_f=[]
wt_noreg_f=[]
wt_ec1_f=[]
wt_eg_f=[]
accuracy_f=[]
accuracy_noreg_f=[]
accuracy_ec1_f=[]
accuracy_eg_f=[]

for u in range(len(lamb)):
    wt=[]
    wt_noreg=[]
    wt_ec1=[]
    wt_eg=[]
    
    
    for a,b in kf.split(X):
        
        train_out=[]
        val_out=[]
        train_inp=X_pca[a]
        val_inp=X_pca[b]
        
        
        for k in a:
            train_out.append(classes1[k])
        for k in b:    
            val_out.append(classes1[k])
    
        weights=np.random.normal(0,10**-5,len(X_pca[1]))
        
        ita=1
        
        '''
        train the weights
        '''
        weights=np.random.normal(0,10**-5,len(X_pca[1]))
            #print weights
        for k in range(1,20):
            temp_cost=[0]*len(weights)
            it=ita*k**(-0.9)
            for y in range(len(train_inp)):
                dot_p=np.dot(weights,train_inp[y])
                temp_cost=temp_cost + train_inp[y]*(train_out[y]-1/(1+np.exp(-np.dot(weights,train_inp[y]))))
            weights=weights+it*temp_cost-weights*it*lamb[u]
        
        wt.append(weights)
        
        '''
        
        
        train the weights for no regularization
        '''
        weights=np.random.normal(0,10**-5,len(X_pca[1]))
            #print weights
        for k in range(1,20):
            weight_pr=np.append([0],weights[1::])
            temp_cost=[0]*len(weights)
            it=ita*k**(-0.9)
            for y in range(len(train_inp)):
                dot_p=np.dot(weights,train_inp[y])
                temp_cost=temp_cost + train_inp[y]*(train_out[y]-1/(1+np.exp(-np.dot(weights,train_inp[y]))))
            weights=weights+it*temp_cost-weight_pr*it*lamb[u]
       
        wt_noreg.append(weights)
        
        '''
        
        train the weights for extra credit
        '''
        weights=np.random.normal(0,10**-5,len(X_pca[1]))
            #print weights
        for k in range(1,20):
            temp_cost=[0]*len(weights)
            it=ita*k**(-0.9)
            for y in range(len(train_inp)):
                dot_p=np.dot(weights,train_inp[y])
                temp_cost=temp_cost + train_inp[y]*(train_out[y]-1/(1+np.exp(-np.dot(weights,train_inp[y]))))
            weights=weights+it*temp_cost-it*lamb[u]*np.array([1]*len(weights))
        
        wt_ec1.append(weights)   
        '''
        
        train the weights for extra credit (EG)
        '''
        
        
        weights_positive=np.random.normal(0,10**-7,len(X_pca[1]))
        weights_negative=np.random.normal(0,10**-7,len(X_pca[1]))
          #print weights
        for k in range(1,20):
            temp_cost_positive=[0]*len(weights)
            temp_cost_negative=[0]*len(weights)
            it=ita*k**(-1.5)
            for y in range(len(train_inp)):
                #dot_p=np.dot(weights,train_inp[y])
                temp_cost_positive=temp_cost_positive + train_inp[y]*(train_out[y]-1/(1+np.exp(-np.dot(weights,train_inp[y]))))
                temp_cost_negative=temp_cost_negative + np.negative(train_inp[y])*(train_out[y]-1/(1+np.exp(-np.dot(weights,np.negative(train_inp[y])))))
            weights_positive=weights_positive*np.exp(it*temp_cost_positive)+lamb[u]*weights_positive
            weights_negative=weights_negative*np.exp(it*temp_cost_negative)/np.sum(weights_negative)
            weights_eg=weights_positive  
        
        wt_eg.append(weights_eg/np.sum(weights_eg))
        
    
        
    sum_n=np.array([0]*len(weights))
    sum_noreg=np.array([0]*len(weights))
    sum_ec1=np.array([0]*len(weights))
    sum_eg=np.array([0]*len(weights_eg))
    for i in wt:
        sum_n=sum_n+np.array(i)
    for i in wt_noreg:
        sum_noreg=sum_noreg+np.array(i)
    for i in wt_ec1:
        sum_ec1=sum_ec1+np.array(i)
    for i in wt_eg:
        sum_eg=sum_eg+np.array(i)
    wt_f.append(sum_n/10)
    wt_noreg_f.append(sum_noreg/10)
    wt_ec1_f.append(sum_ec1/10)
    wt_eg_f.append(sum_eg/10)
        

     
for u in range(len(lamb)):
    sum=0
    sum_noreg=0
    sum_ec1=0
    sum_eg=0
    for a,b in kf.split(X):
    
        '''
        calculate the accuracy on validation set
        '''

        accuracy=[]
        results=[]
        
        
        for t in range(len(val_inp)):
            k=0
            if 1/(1+np.exp(-np.dot(wt_f[u],val_inp[t])))>=0.99:
                k=1
            results.append(k)
            if val_out[t]==k :
                sum=sum+1
        accuracy.append(sum/len(val_out))
    
    
        '''
        calculate the accuracy on validation set for no regularization
        '''

        accuracy_noreg=[]
        results_noreg=[]
        
        
        for t in range(len(val_inp)):
            k=0
            if 1/(1+np.exp(-np.dot(wt_noreg_f[u],val_inp[t])))>=0.99:
                k=1
            results_noreg.append(k)
            if val_out[t]==k :
                sum_noreg=sum_noreg+1
        accuracy_noreg.append(sum_noreg/len(val_out))
        #print accuracy  
        
        '''
        calculate the accuracy on validation set for extra credit
        '''
        accuracy_ec1=[]
        results_ec1=[]
        
        
        for t in range(len(val_inp)):
            k=0
            
            
            if 1/(1+np.exp(-np.dot(wt_ec1_f[u],val_inp[t])))>=0.99:
                k=1
            results_ec1.append(k)
            if val_out[t]==k :
                    sum_ec1=sum_ec1+1
        accuracy_ec1.append(sum_ec1/len(val_out))
        #print "extra credit="+str(accuracy_ec1)
    
    
        
        '''
        calculate the accuracy on validation set for extra credit(EG)
        '''
        accuracy_eg=[]
        results_eg=[]
        
        
        for t in range(len(val_inp)):
            k=0
            val_inp_neg=np.array([-r for r in val_inp[t]])
            val_inp_eg=np.append(val_inp[t],val_inp_neg)
            if 1/(1+np.exp(-np.dot(weights_positive,val_inp[t])))>=0.99:
                k=1
            results_eg.append(k)
            if val_out[t]==k :
                sum_eg=sum_eg+1
        accuracy_eg.append(sum_eg/len(val_out))
        #print "extra credit(EG) ="+str(accuracy_eg)
    
        
    accuracy_f.append(np.sum(accuracy)/10)
    accuracy_noreg_f.append(np.sum(accuracy_noreg)/10)
    accuracy_ec1_f.append(np.sum(accuracy_ec1)/10)
    accuracy_eg_f.append(np.sum(accuracy_eg)/10)
    

print (accuracy_f)
print ("extra credit(noreg)="+str(accuracy_noreg_f))
print ("extra credit="+str(accuracy_ec1_f))
print ("extra credit(EG) ="+str(accuracy_eg_f))




'''
weights calculated on the entire  training data
'''
weights_final=np.random.normal(0,10**-5,len(X_pca[1]))
for k in range(1,100):
    temp_cost=[0]*len(weights_final)
    it=ita*k**(-0.9)
    for y in range(len(X_pca)):
        dot_p=np.dot(weights_final,X_pca[y])
        temp_cost=temp_cost + X_pca[y]*(classes1[y]-1/(1+np.exp(-np.dot(weights_final,X_pca[y]))))
    weights_final=weights_final+it*temp_cost-weights_final*it*lamb[accuracy_f.index(max(accuracy_f))]
    
'''
weights calculated on the entire  training data for no regularization
'''
weights_final_noreg=np.random.normal(0,10**-5,len(X_pca[1]))
for k in range(1,100):
    temp_cost=[0]*len(weights_final_noreg)
    it=ita*k**(-0.9)
    weights_final_noreg_pr=np.append([0],weights_final_noreg[1::])
    for y in range(len(X_pca)):
        dot_p=np.dot(weights_final,X_pca[y])
        temp_cost=temp_cost + X_pca[y]*(classes1[y]-1/(1+np.exp(-np.dot(weights_final_noreg,X_pca[y]))))
    weights_final_noreg=weights_final_noreg+it*temp_cost-weights_final_noreg_pr*it*lamb[accuracy_noreg_f.index(max(accuracy_noreg_f))]

'''
weights calculated on the entire  training data for extra credit
'''   
weights_final_ec1=np.random.normal(0,10**-5,len(X_pca[1]))
for k in range(1,100):
    temp_cost=[0]*len(weights_final_ec1)
    it=ita*k**(-0.9)
    for y in range(len(X_pca)):
        dot_p=np.dot(weights_final_ec1,X_pca[y])
        temp_cost=temp_cost + X_pca[y]*(classes1[y]-1/(1+np.exp(-np.dot(weights_final_ec1,X_pca[y]))))
    weights_final_ec1=weights_final_ec1+it*temp_cost-it*lamb[accuracy_ec1_f.index(max(accuracy_ec1_f))]*np.array([1]*len(weights_final_ec1))


'''
weights calculated on the entire  training data for extra credit(EG)
'''   
weights_final_eg_positive=np.random.normal(0,10**-5,len(X_pca[1]))
weights_final_eg_negative=np.random.normal(0,10**-5,len(X_pca[1]))
for k in range(1,100):
    temp_cost_pos=[0]*len(weights_final_eg_positive)
    temp_cost_neg=[0]*len(weights_final_eg_negative)
    it=ita*k**(-1.5)
    for y in range(len(X_pca)):
        
        
        #X_eg=np.append(X_pca[y],np.negative(X_pca[y]))
        
        temp_cost_pos=temp_cost_pos + X_pca[y]*(classes1[y]-1/(1+np.exp(-np.dot(weights_final_eg_positive,X_pca[y]))))
        #temp_cost_neg=temp_cost_neg + np.negative(X_pca[y])*(classes[y]-1/(1+np.exp(-np.dot(weights_final_eg_negative,np.negative(X_pca[y])))))
    
    weights_final_eg_positive=weights_final_eg_positive*np.exp(it*temp_cost_pos)+lamb[3]*weights_final_eg_positive
weights_final_eg_positive=weights_final_eg_positive/np.sum(weights_final_eg_positive)
    #weights_final_eg_negative=weights_final_eg_negative*np.exp(it*temp_cost_neg)/np.sum(weights_final_eg_negative)
#weights_final_eg=np.append(weights_final_eg_positive)
#weights_final_eg_positive=weights_final_eg_positive/np.sum(weights_final_eg_positive)

       
'''
calculate the accuracy on test set
'''
test=open(r"tokenized_test.txt","r")
cat_test=open(r"cat_test.txt","r")
categories=cat_test.readlines()
v=test.readlines()
#v=v1[0:len(v1)-1]
token=[]
for h in range(len(v)):
    token=token+RegexpTokenizer(r'\w+').tokenize(v[h])
'''
t=open(r"tokenized_test.txt","w")

t1=open(r"tokenized_test.txt","r")
v=t1.readlines()
u=[]
for k in range(1,len(v)):
    print k
    y=v[k].split(" ")
    u.append(" ".join([word for word in y if word not in stopwords.words('english')]).strip("\n").strip(",,,"))
    cat_test.write(u[0].split(",")[0]+"\n")
    #t.write(u[k-1]+"\n")
'''
corpus=[]
for t in range(len(v)):
    corpus.append(v[t].strip("\n"))
dictionary=set(token)
#cv =CountVectorizer(dictionary)
X=cv.transform(corpus)
classe=[]
for k in range(len(categories)):
    classe.append(categories[k].strip("\n"))
classes=[]
for k in range(len(categories)):
    if classe[k]=='ham':
        classes.append(0)
    else:
        classes.append(1)

accuracy_test=[]
results_test=[]

X=np.matmul((X.toarray()-np.mean(X.toarray())),np.transpose(eigvectors))
sum=0
sum_tp=0
sum_tn=0
for t in range(len(classes)-10):
    k=0
    if 1/(1+np.exp(-np.dot(weights_final,X[t])))>=0.99:
        k=1
    results_test.append(k)
    if classes[t]==k :
        sum=sum+1
    if classes[t]==1 and results_test[t]==1:
        sum_tp=sum_tp+1
    if classes[t]==0 and results_test[t]==0:
        sum_tn=sum_tn+1
    
print (sum/len(classes))    
print (sum_tp/330)
print (sum_tn/2242)


'''
calculate accuracy on test set for no regularization
'''
sum_tp_noreg=0
sum_tn_noreg=0
sum_noreg=0
accuracy_test_noreg=[]
results_test_noreg=[]
for t in range(len(classes)):
    k=0
    if 1/(1+np.exp(-np.dot(weights_final_noreg,X[t])))>=0.99:
        k=1
    results_test_noreg.append(k)
    if classes[t]==k :
        sum_noreg=sum_noreg+1
    if classes[t]==1 and results_test_noreg[t]==1:
        sum_tp_noreg=sum_tp_noreg+1
    if classes[t]==0 and results_test_noreg[t]==0:
        sum_tn_noreg=sum_tn_noreg+1
    
print ("extra credit(Nonreg) "+str(sum_noreg/len(classes)))    
print ("extra credit(Nonreg) "+str(sum_tp_noreg/330))
print ("extra credit(Nonreg) "+str(sum_tn_noreg/2242))

'''
calculate accuracy on test set for extra credit
'''
sum_tp_ec1=0
sum_tn_ec1=0
sum_ec1=0
accuracy_test_ec1=[]
results_test_ec1=[]
for t in range(len(classes)):
    k=0
    if 1/(1+np.exp(-np.dot(weights_final_ec1,X[t])))>=0.99:
        k=1
    results_test_ec1.append(k)
    if classes[t]==k :
        sum_ec1=sum_ec1+1
    if classes[t]==1 and results_test_ec1[t]==1:
        sum_tp_ec1=sum_tp_ec1+1
    if classes[t]==0 and results_test_ec1[t]==0:
        sum_tn_ec1=sum_tn_ec1+1
    
print ("extra credit "+str(sum_ec1/len(classes)) )   
print ("extra credit "+str(sum_tp_ec1/330))
print ("extra credit "+str(sum_tn_ec1/2242))

'''
calculate accuracy on test set for extra credit(EG)
'''
sum_tp_eg=0
sum_tn_eg=0
sum_eg=0
accuracy_test_eg=[]
results_test_eg=[]
for t in range(len(classes)):
    k=0
    X_neg=np.array([-y for y in X[t]])
    X_eg=np.append(X[t],X_neg)
    if 1/(1+np.exp(-np.dot(weights_final_eg_positive,X[t])))>=0.99:
        k=1
    results_test_eg.append(k)
    if classes[t]==k :
        sum_eg=sum_eg+1
    if classes[t]==1 and results_test_eg[t]==1:
        sum_tp_eg=sum_tp_eg+1
    if classes[t]==0 and results_test_eg[t]==0:
        sum_tn_eg=sum_tn_eg+1
    
print ("extra credit(EG) "+str(sum_eg/len(classes)))    
print ("extra credit(EG) "+str(sum_tp_eg/330))
print ("extra credit(EG) "+str(sum_tn_eg/2242))


'''
calculating loss
'''
loss=[]
for k in range(len(lamb)):
    sum1=0
    for y in range(len(X_pca)):
        if classes1[y]==0 and (np.dot(wt_f[k],X_pca[y]))<100 :
            sum1=sum1+(1-classes1[y])*np.log((1-classes1[y])*(1+np.exp(np.dot(wt_f[k],X_pca[y]))))
        elif classes1[y]==0 and (np.dot(wt_f[k],X_pca[y]))>100 :
            sum1=sum1+1000
        elif classes1[y]==1 and (np.dot(wt_f[k],X_pca[y]))>-100:
            sum1=sum1+classes1[y]*np.log(classes1[y]*(1+np.exp(-np.dot(wt_f[k],X_pca[y]))))
        elif classes1[y]==1 and (np.dot(wt_f[k],X_pca[y]))<-100:
            sum1=sum1+1000
    sum1=sum1/np.log(2)
    loss.append(sum1) 

'''
calculating loss for no regularization
'''
loss_noreg=[]
for k in range(len(lamb)):
    sum1=0
    for y in range(len(X_pca)):
        if classes1[y]==0 and (np.dot(wt_noreg_f[k],X_pca[y]))<100 :
            sum1=sum1+(1-classes1[y])*np.log((1-classes1[y])*(1+np.exp(np.dot(wt_noreg_f[k],X_pca[y]))))
        elif classes1[y]==0 and (np.dot(wt_noreg_f[k],X_pca[y]))>100 :
            sum1=sum1+1000
        elif classes1[y]==1 and (np.dot(wt_noreg_f[k],X_pca[y]))>-100:
            sum1=sum1+classes1[y]*np.log(classes1[y]*(1+np.exp(-np.dot(wt_noreg_f[k],X_pca[y]))))
        elif classes1[y]==1 and (np.dot(wt_noreg_f[k],X_pca[y]))<-100:
            sum1=sum1+1000
            
    sum1=sum1/np.log(2)
    loss_noreg.append(sum1) 

'''
calculating loss for extra credit
'''
loss_ec1=[]
for k in range(len(lamb)):
    sum1=0
    for y in range(len(X_pca)):
        if classes1[y]==0 and (np.dot(wt_ec1_f[k],X_pca[y]))<100 :
            sum1=sum1+(1-classes1[y])*np.log((1-classes1[y])*(1+np.exp(np.dot(wt_ec1_f[k],X_pca[y]))))
        elif classes1[y]==0 and (np.dot(wt_ec1_f[k],X_pca[y]))>100 :
            sum1=sum1+1000
        elif classes1[y]==1 and (np.dot(wt_ec1_f[k],X_pca[y]))>-100:
            sum1=sum1+classes1[y]*np.log(classes1[y]*(1+np.exp(-np.dot(wt_ec1_f[k],X_pca[y]))))
        elif classes1[y]==1 and (np.dot(wt_ec1_f[k],X_pca[y]))<-100:
            sum1=sum1+1000
    sum1=sum1/np.log(2)
    loss_ec1.append(sum1) 

'''
calculating loss (EG)
'''
loss_eg=[]
for k in range(len(lamb)):
    sum1=0
    for y in range(len(X_pca)):
        if classes1[y]==0 and (np.dot(wt_eg_f[k],X_pca[y]))<300 :
            sum1=sum1+(1-classes1[y])*np.log((1-classes1[y])*(1+np.exp(np.dot(wt_eg_f[k],X_pca[y]))))
        elif classes1[y]==0 and (np.dot(wt_eg_f[k],X_pca[y]))>300 :
            sum1=sum1+1
        elif classes1[y]==1 and (np.dot(wt_eg_f[k],X_pca[y]))>-300:
            sum1=sum1+classes1[y]*np.log(classes1[y]*(1+np.exp(-np.dot(wt_eg_f[k],X_pca[y]))))
        elif classes1[y]==1 and (np.dot(wt_eg_f[k],X_pca[y]))<-300:
            sum1=sum1+1
    sum1=sum1/np.log(2)
    loss_eg.append(sum1) 


print ("loss= "+str(loss))
print ("loss(noreg)= "+str(loss_noreg))
print ("loss(extra credit)= "+str(loss_ec1))
print ("loss(EG)= "+str(loss_eg))