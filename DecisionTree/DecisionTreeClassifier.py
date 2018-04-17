 # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation 
#将生成的树输出做png或者pdf文件
from sklearn.tree import export_graphviz

def load_data():
     #自带数据集，花花的数据
     iris = datasets.load_iris()
     X_train = iris.data
     y_train = iris.target
     return cross_validation.train_test_split(X_train,y_train,test_size = 0.25,
     random_state=0,stratify=y_train)
     
def test_DecisionTreeClassifier(*data):
     X_train,X_test,y_train,y_test = data
     clf = DecisionTreeClassifier()
     clf.fit(X_train,y_train)
     
     print ("Training socre %f"%clf.score(X_train,y_train))
     print ("Testing socre %f"%clf.score(X_test,y_test))
     
     
def test_DecisionTreeClassifier_criterion(*data):
     X_train,X_test,y_train,y_test = data
     criterion = ['gini','entropy']
     for cri in criterion:
          clf = DecisionTreeClassifier(criterion=cri)
          clf.fit(X_train,y_train)
          print ("in %s"%cri)
          print ("Training socre %f"%clf.score(X_train,y_train))
          print ("Testing socre %f"%clf.score(X_test,y_test))

def test_DecisionTreeClassifier_splitter(*data):
     X_train,X_test,y_train,y_test = data
     splitters = ['best','random']
     for splitter in splitters:
          clf=DecisionTreeClassifier(splitter=splitter)
          clf.fit(X_train,y_train)
          
          print("Splitter:%s"%splitter)
          print("Training Score:%f"%clf.score(X_train,y_train))
          print("Testing Score:%f"%clf.score(X_test,y_test))
          
def test_DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,maxdepth):
     depths=np.arange(1,maxdepth)
     training_scores=[]
     testing_scores=[]
     for depth in depths:
          clf=DecisionTreeClassifier(max_depth=depth)
          clf.fit(X_train,y_train)
          training_scores.append(clf.score(X_train,y_train))
          testing_scores.append(clf.score(X_test,y_test))
          
     #绘图开始
     fig=plt.figure()
     ax=fig.add_subplot(1,1,1)
     ax.plot(depths,training_scores,label='training score',marker='o')
     ax.plot(depths,testing_scores,label='testing score',marker='*')
     ax.set_xlabel('maxdepth')
     ax.set_ylabel('scores')
     ax.set_title('Decison Tree Classification')
     ax.legend(framealpha=0.5,loc='best')
     plt.show()


X_train,X_test,y_train,y_test = load_data()
#test_DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,maxdepth=10)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
export_graphviz(clf,"/Users/apple/Desktop/DecisionTreeClassifierPic")






