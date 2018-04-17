## -*- coding:utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,datasets,cross_validation

def load_classification_data():
     digits=datasets.load_digits()
     X_train=digits.data
     y_train=digits.target
     return cross_validation.train_test_split(X_train,y_train,test_size=0.25,
     random_state=0,stratify=y_train)
     
def create_regression_data(n):
     X=5*np.random.rand(n,1)
     y=np.sin(X).ravel()
     y[::5]+=1*(0.5-np.random.rand(int(n/5)))
     return cross_validation.train_test_split(X,y,test_size=0.25,random_state=0)
     
def test_KNeighborClassifier(*data):
     X_train,y_train,X_test,y_test=data
     clf=neighbors.KNeighborsClassifier()
     clf.fit(X_train,y_train)
     print("Training score:%f"%clf.score(X_train,y_train))
     print("Testing score:%f"%clf.score(X_test,y_test))
     
#考察k值（选择多少个临近的点）和投票策略（每个点所贡献的权重）对于分类的影响
def test_KNeighborClassifier_k_w(*data):
     X_train,y_train,X_test,y_test=data
     Ks=np.linspace(1,y_train.size,num=100,endpoint=False,dtype='int')#创造等差数列
     weights=['uniform','distance']

     fig=plt.figure()
     ax=fig.add_subplot(1,1,1)
     for weight in weights:
          training_scores=[]
          testing_scores=[]
          for k in Ks:
               clf=neighbors.KNeighborsClassifier(weights=weight,n_neighbors=k)
               clf.fit(X_train,y_train)
               testing_scores.append(clf.score(X_test,y_test))
               training_scores.append((clf.score(X_train,y_train)))
          
          ax.plot(Ks,testing_scores,label="testing score:weight=%s"%weight)
          ax.plot(Ks,training_scores,label="training score:weight=%s"%weight)
     ax.legend(loc='best')
     ax.set_xlabel('K')
     ax.set_ylabel('score')
     ax.set_ylim(0,1.05)
     ax.set_title("KNeighborClassiffier")
     plt.show()
#下面考察p值，就是距离函数的一般形式对于分类预测的影响
def test_KNeighborClassifier_k_p(*data):
     X_train,y_train,X_test,y_test=data
     Ps=[1,2,10]
     Ks=np.linspace(1,y_train.size,endpoint=False,dtype='int')
     
     fig=plt.figure()
     ax=fig.add_subplot(1,1,1)
     for p in Ps:
          training_scores=[]
          testing_scores=[]
          for k in Ks:
               clf=neighbors.KNeighborsClassifier(p=p,n_neighbors=k)
               clf.fit(X_train,y_train)
               testing_scores.append(clf.score(X_test,y_test))
               training_scores.append(clf.score(X_train,y_train))
          ax.plot(Ks,testing_scores,label='testing score:p=%d'%p)
          ax.plot(Ks,training_scores,label='training score:p=%d'%p)
     ax.legend(loc='best')
     ax.set_xlabel('K')
     ax.set_ylabel('score')
     ax.set_ylim(0,1.05)
     ax.set_title('KNeighborClassiffier')
     plt.show()
     
#关于回归模型，KNeighborsRegressor提供回归来分类预测，不同的是分类是多数加权表决
#回归是取平均值

def test_kNeighborRegressor(*data):
     X_train,X_test,y_train,y_test=data
     regr=neighbors.KNeighborsRegressor()
     regr.fit(X_train,y_train)
     print "Training score:%f"%(regr.score(X_train,y_train))
     print "Testing score:%f"%(regr.score(X_test,y_test))
     

def test_kNeighborRegressor_k_w(*data):
     X_train,X_test,y_train,y_test=data
     Ks=np.linspace(1,y_train.size,endpoint=False,dtype='int')
     weights=['uniform','distance']
     fig=plt.figure()
     ax=fig.add_subplot(1,1,1)
     for weight in weights:
          train_scores=[]
          test_scores=[]
          for k in Ks:
               regr=neighbors.KNeighborsRegressor(weights=weight,n_neighbors=k)
               regr.fit(X_train,y_train)
               train_scores.append(regr.score(X_train,y_train))
               test_scores.append(regr.score(X_test,y_test))
          ax.plot(Ks,test_scores,label="testing scores:weights=%s"%weight)
          ax.plot(Ks,train_scores,label="training scores:weight=%s"%weight)
     ax.legend(loc="best")
     ax.set_xlabel('K')
     ax.set_ylabel('score')
     ax.set_ylim(0,1.05)
     ax.set_title("KNeighbor Regressor")
     plt.show()
     
def test_kNeighborRegressor_k_p(*data):
     X_train,X_test,y_train,y_test=data
     Ps=[1,2,10]
     Ks=np.linspace(1,y_train.size,endpoint=False,dtype='int')
     fig=plt.figure()
     ax=fig.add_subplot(1,1,1)
     for p in Ps:
          train_scores=[]
          test_scores=[]
          for k in Ks:
               regr=neighbors.KNeighborsRegressor(p=p,n_neighbors=k)
               regr.fit(X_train,y_train)
               train_scores.append(regr.score(X_train,y_train))
               test_scores.append(regr.score(X_test,y_test))
          ax.plot(Ks,train_scores,label="train scores:p=%d"%p)
          ax.plot(Ks,test_scores,label="test scores:p=%d"%p)
     ax.legend(loc='best')
     ax.set_xlabel('X')
     ax.set_ylabel('score')
     ax.set_ylim(0,1.05)
     ax.set_title("KNeighbor Regressor")
     plt.show()
     
     
X_train,X_test,y_train,y_test=create_regression_data(1000)
#test_KNeighborClassifier_k_p(X_train,X_test,y_train,y_test)
test_kNeighborRegressor_k_p(X_train,X_test,y_train,y_test)