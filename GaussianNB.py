## -*- coding:utf-8 -*- 
from sklearn import datasets,cross_validation,naive_bayes
import numpy as np
import matplotlib.pyplot as plt

def show_digits():
     digits=datasets.load_digits()
     fig=plt.figure()
     print('vector from image 0:',digits.data[0])
     for i in range(25):
          ax=fig.add_subplot(5,5,i+1)
          ax.imshow(digits.images[i],cmap=plt.cm.gray_r,interpolation='nearest')
     plt.show()
          #上述函数为展示我们训练和测试所用数据，手写识别数字的数据库，官方自带
#show_digits()

def load_data():
     digits=datasets.load_digits()
     return cross_validation.train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)
     
def test_GaussianNB(*data):
     X_train,X_test,y_train,y_test=data
     cls=naive_bayes.GaussianNB()
     cls.fit(X_train,y_train)
     print("Training score %f"%(cls.score(X_train,y_train)))
     print("Testing score %f"%(cls.score(X_test,y_test)))
     
     
def test_MultinomialNB(*data):
     X_train,X_test,y_train,y_test=data
     cls=naive_bayes.MultinomialNB()#这里不指定alpha，原函数中定义alpha默认为1
     cls.fit(X_train,y_train)
     print('Training score: %f'%(cls.score(X_train,y_train)))
     print('Testing score: %f'%(cls.score(X_test,y_test)))
     
#接下来验证不同的alpha对于多项式朴素贝叶斯分类器的影响
###
###alpha如果过大，，条件概率趋近于n/1，忽略了特征之间的差别
###
def test_MultinomialNB_alpha(*data):#这里没有把alphs写进函数的参数里
     X_train,X_test,y_train,y_test=data
     alphas=np.logspace(-2,5,num=200)#这里logspace函数用来创建等比数列，
     #这里的-2，5都是10的幂指数，num参数是表示创建的等比数列的长度
     train_scores=[]
     test_scores=[]
     for alpha in alphas:
          cls=naive_bayes.MultinomialNB(alpha=alpha)
          cls.fit(X_train,y_train)
          train_scores.append(cls.score(X_train,y_train))
          test_scores.append(cls.score(X_test,y_test))
     #下面又她妈的开始绘图了
     fig=plt.figure()
     ax=fig.add_subplot(1,1,1)
     ax.plot(alphas,train_scores,label='Training score')
     ax.plot(alphas,test_scores,label='Testing score')
     ax.set_xlabel(r"$\alpha$")#表示alpha字符的变量，显示为希腊字母
     ax.set_ylabel("score")
     ax.set_ylim(0,1.0)
     ax.set_title("MultinomialNB")
     ax.set_xscale("log")
     plt.show()
#测试伯努利贝叶斯分类对于的性能     
def test_BernoulliNB(*data):
     X_train,X_test,y_train,y_test=data
     cls=naive_bayes.BernoulliNB()
     cls.fit(X_train,y_train)
     print('Training score:%f'%(cls.score(X_train,y_train)))
     print('Testing score:%f'%(cls.score(X_test,y_test)))

#然后测试不同的alpha的影响，真是无聊。。。
def test_BernoulliNB_alpha(*data):
     X_train,X_test,y_train,y_test=data
     alphas=np.logspace(-2,5,num=200)
     training_scores=[]
     testing_scores=[]
     for alpha in alphas:
          cls=naive_bayes.BernoulliNB(alpha=alpha)
          cls.fit(X_train,y_train)
          training_scores.append(cls.score(X_train,y_train))
          testing_scores.append(cls.score(X_test,y_test))
     #科科
     fig=plt.figure()
     ax=fig.add_subplot(1,1,1)
     ax.plot(alphas,training_scores,label='train score')
     ax.plot(alphas,testing_scores,label='test score')
     ax.set_xlabel(r'$\alpha$')
     ax.set_ylabel('score')
     ax.set_ylim(0,1.0)
     ax.set_title("BernoulliNB")
     ax.set_xscale('log')
     ax.legend(loc='best')
     plt.show()
     
     
# 这里测试二元化对于伯努利贝叶斯分类器分类效果的影响
#二元化的主要作用就是设定一个阈值，将输入的特征分为两部分，一部分为0，一部分为1，也就是将伯努利概率分布做成特殊的二项01分布,这里如果此项参数为空，默认数据已经二元化

def test_BernoulliNB_binarize(*data):
     #将二元化的阈值上下限加0.1，为了能出现全0或者全1的情况，好观察断崖
     min_x=min(min(X_train.ravel()),min(X_test.ravel()))-0.1
     max_x=max(max(X_train.ravel()),max(X_test.ravel()))+0.1
     binarizes=np.linspace(min_x,max_x,endpoint=True,num=100)
     train_scores=[]
     test_scores=[]
     for binarize in binarizes:
          cls=naive_bayes.BernoulliNB(binarize=binarize)
          cls.fit(X_train,y_train)
          test_scores.append(cls.score(X_test,y_test))
          train_scores.append(cls.score(X_train,y_train))
          
     #又开始了 
     fig=plt.figure()
     ax=fig.add_subplot(1,1,1)
     ax.plot(binarizes,train_scores,label='train score') 
     ax.plot(binarizes,test_scores,label='test score')
     ax.set_xlabel('Binarize')
     ax.set_ylabel('score')
     ax.set_ylim(0,1.0)
     ax.set_xlim(min_x-1,max_x+1)
     ax.set_title("BernoulliNB")
     ax.legend(loc='best')
     plt.show()    
###执行部分     
X_train,X_test,y_train,y_test=load_data()
digits=datasets.load_digits()
#test_GaussianNB(X_train,X_test,y_train,y_test)
#test_MultinomialNB(X_train,X_test,y_train,y_test)
#test_MultinomialNB_alpha(X_train,X_test,y_train,y_test)
#test_BernoulliNB_alpha(X_train,X_test,y_train,y_test)
#test_BernoulliNB_binarize(X_train,X_test,y_train,y_test)
cls=naive_bayes.BernoulliNB()
cls.fit(X_train,y_train)
print cls.predict(digits.data[9])
