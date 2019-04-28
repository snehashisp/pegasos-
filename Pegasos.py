#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as svm



#to get the data for a set of labels
def getData(data,labels,sublabels):
    sdata = []
    for l in sublabels:
        sdata += [data[np.where(labels == l)]]
        
    return np.array(sdata)

def randomInit(shape):
    return np.random.randn(shape[0],shape[1])




#linear kernel (simply inner product of vectors)
def linearKernel(x,x_):
    return x.reshape(1,-1).dot(x_.reshape(-1,1))
#get a polynomial kernel of degree d
def getPolyKernel(d):

    def PolyKernel(x,x_):
        return float((1 + x.reshape(1,-1).dot(x_.reshape(-1,1)))**d)
    return PolyKernel

#get a gaussian kernel with standard deviation of std
def getGaussianKernel(std):
    
    def GausianKernel(x,x_):
        x = x.reshape(1,-1)
        x_ = x.reshape(1,-1)
        return np.exp(-1*(np.linalg.norm(x - x_)**2)/2*std**2)
    return GausianKernel




class pegasos_solver:
    
    def init(self,data,labels,weight_vector,bias = None):
        
        self.data = data #training data
        self.label = labels #training lables
        self.bias = bias #bias
        self.weights = np.array(weight_vector).reshape(-1,1) #weight vector
        
        self.supportVec = None
    
    def printWeights(self):
        print(self.weights)
    
    def loss(self,batchData,batchlabel):
        prod = np.max((np.zeros([batchData.shape[0],1]),                         1-((batchData.dot(self.weights) + (self.bias if self.bias is not None else 0))                         * batchlabel)),axis=0)        
        return np.sum(prod)/batchData.shape[0]
    
    def predict(self,batchData):
        return batchData.dot(self.weights) + (self.bias if self.bias is not None else 0)
    
    def predictKernel(self,kernel,batchData):
        
        res = []
        #get non-zero support vectors
        nzsv = np.where(self.supportVec != 0)[0]
        for d in batchData:
            #predict the label using the dual form of w
            res += [sum([self.supportVec[j] * self.label[j] * kernel(d,self.data[j])                            for j in nzsv])]
        return np.array(res).reshape(-1,1)
            
    #training using subgradient descent
    def train(self,B,L,T):
        
        errs = []
        
        #B = Batch Size
        #L = Lambda / Regularizing Constant
        #T = Total number of iterations      
        for t in range(1,T+1):
            
            #updating learning rate and getting a random set of data
            lrate = 1/(L*t)
            perm = np.random.permutation(self.data.shape[0])[:B]
            subData,sublabl = self.data[perm],self.label[perm]
            
            #calculating loss and the set of data that conribute to loss
            prod = (subData.dot(self.weights) + (self.bias if self.bias is not None else 0)) * sublabl #errors
            updateBatch = np.where(prod < 1)[0] #find non zero erros
            #errs += [self.loss(subData,sublabl)]
            subData,sublabl = subData[updateBatch],sublabl[updateBatch] #data that contribute to error
            
            
            #updating weights and bias by gradient of loss
            if sublabl.shape[0]:
                self.weights = (1 - 1/t)*self.weights +                     (lrate/sublabl.shape[0]) * subData.T.dot(sublabl)
                #update bias 
                if self.bias is not None:
                    self.bias -= (lrate/sublabl.shape[0]) * np.sum(sublabl)
                
        return errs
    
    #training using mercer kernel functions
    def kernelTrain(self,L,T,kernel):
        
        self.supportVec = np.zeros([self.data.shape[0],1])
        for t in range(1,T+1):
            #update learning rate and get a random sample from data
            lrate = 1/(L*t)
            r = np.random.randint(0,self.data.shape[0])
            d,l = self.data[r],self.label[r]
            
            #get non-zero support vectors
            nzsv = np.where(self.supportVec != 0)[0]

            #compute the support vectors 
            kres = sum([self.supportVec[j] * self.label[j] * kernel(d,self.data[j])                            for j in nzsv])
            if l*lrate*kres < 1:
                self.supportVec[r] += 1
            
            


def trainTestSvm(data,labels,lblvec,B,L,T,split = 0.8,initializer = np.zeros,use_bias = True,kernel = None):
    
    #sectioning data into only two labels and shuffling them
    data1,data2 = getData(data,labels,lblvec)
    label1,label2 = np.ones([data1.shape[0],1])*1,np.ones([data2.shape[0],1])*-1
    data12 = np.concatenate((data1,data2))
    label12 = np.concatenate((label1,label2))
    perm = np.random.permutation(data12.shape[0])
    data12 = data12[perm]
    label12 = label12[perm]
    
    #split into training and testing datasets
    sp = int(split * data12.shape[0])
    train,trainlbl,test,testlbl = data12[:sp],label12[:sp],data12[sp:],label12[sp:]
    
    #create and train the svm on the training set
    svm = pegasos_solver()
    bias = initializer([1]) if use_bias else None
    svm.init(train,trainlbl,initializer([1,data12.shape[1]]),bias)
    
    #train and test the svm either with primal subgradient descent or mercer kernels 
    if kernel is None:
        errs = svm.train(B,L,T)
        tres = svm.predict(test)
    else:
        svm.kernelTrain(L,T,kernel)
        tres = svm.predictKernel(kernel,test)
        
    tp,fp,tn,fn = 0,0,0,0
    for i,j in zip(tres,testlbl):
        if i > 0:
            if j > 0:
                tp += 1
            else:
                fp += 1
        else:
            if j < 0:
                tn += 1
            else:
                fn += 1
          
    print("Accuracy",(tp + tn)/(tres.shape[0]),"TPR",tp/(tp + fn),           "FPR",fp/(tn + fp))
    return (tp + tn)/(tres.shape[0])



#driver code
try:
    dataLoc = input("Enter the location of Data : ")
    labelName = input("Enter the name of the label column : ")
    data = pd.read_csv(dataLoc)
    label = data[labelName].values
    data = data.loc[:, data.columns != labelName].values.astype(np.float)
except:
    print("Incorrect Information Entered")


normalize = int(input("Normalize Data 1 - yes 0 - no : "))
if normalize == 1:
    for i in range(data.shape[1]):
        data[:,i] = data[:,i].astype(np.float)/float(np.max(data[:,i]))

labels = np.unique(label)
print("Possible labels for binary classification ")
for i in range(labels.shape[0]):
    print("Index ",i,"label",labels[i])
print("Enter the 2 labels for classification\n")        

try:
    L1 = labels[int(input("Enter Label 1 index from list : "))]
    L2 = labels[int(input("Enter Label 2 index from list : "))]
    L = float(input("Enter Lambda Value : "))
    T = int(input("Enter Iterations : "))
    S = float(input("Enter training - testing split eg for 80-20 enter 0.8 : "))
    B = 100
    PD = int(input("Enter 1 for primal 2 for dual version : "))
    if PD == 1:
        B = int(input("Enter Batch Size "))
        b = True if int(input("Use Bias 1 - True 0 - False : ")) == 1 else 0
        print(B,L,T,S,b)
        print("Labels",L1,"and",L2,"Accuracy True positive rate and False positive rate ")
        trainTestSvm(data,label,[L1,L2],B,L,T,split=S,use_bias=b)
    elif PD == 2:
        kernel = None
        kType = int(input("1 - Polynomial 2 - Gaussian : " ))
        if kType == 1:
            degree = int(input("Enter the degree of the polynomial : "))
            kernel = getPolyKernel(degree)
        elif kType == 2:
            std = float(input("Enter the standard deviation of the gaussian : "))
            kernel = getGaussianKernel(std)
        print("Labels",L1,"and",L2,"Accuracy True positive rate and False positive rate")
        trainTestSvm(data,label,[L1,L2],B,L,T,split=S,use_bias=False,kernel=kernel)
except:
    print("Error entering some data")
