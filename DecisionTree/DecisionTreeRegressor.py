 # -*- coding: utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz

#给出随机产生的数据集
def creat_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n,1)
    y = np.sin(X).ravel()
    noise_num = (int)(n/5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))
    return  cross_validation.train_test_split(X,y,test_size = 0.25,random_state = 1 )
    #cross_validation.train_test_split将数据分成两部分，一部分作为训练，一部分作为测试
    
def test_DecisionTreeRegressor(*data):
    X_train,X_test,y_train,y_test = data
    regr = DecisionTreeRegressor(max_depth = 21)
    regr.fit(X_train,y_train)#训练决策树的方法
    print ("Training score : %f"%(regr.score(X_train,y_train)))
    print ("Testing score : %f"%(regr.score(X_test,y_test)))
##接下来开始绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    X = np.arange(0.0,5.0,0.01)[:,np.newaxis]#0到5，步长为0.01
    Y = regr.predict(X)
    ax.scatter(X_train,y_train,label = "train sample", c='b')
    ax.scatter(X_test,y_test,label = "test sample", c='y')
    ax.plot(X,Y,label ="predict_value",linewidth = 2,alpha = 0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decicison Tree Regeressor")
    ax.legend(framealpha = 0.5)
    plt.show()
##绘图结束（这里将图像放在函数内部）

#考察随机划分和最优划分对于抉择树的影响
def test_DecisionTreeRegressor_splitter(*data):
     #python中*arg表示接受一个元组作为输入
     X_train,X_test,y_train,y_test = data
     #这一行制定 arg中的成员
     splitters = ['best','random']
     for splitter in splitters:
          regr = DecisionTreeRegressor(splitter = splitter)
          regr.fit(X_train,y_train)
          print("Splitter %s"%splitter)
          print("Training score %f"%(regr.score(X_train,y_train)))
          print("Testing score %f"%(regr.score(X_test,y_test)))
          

#考察树的深度对于决策树分类的影响
def test_DecisionTreeRegressor_depth(X_train,X_test,y_train,y_test, maxdepth):
     #X_train,X_test,y_train,y_test = data
     #上面写法在前面是*data，后面报错。
     depths = np.arange(1,maxdepth)
     training_scores = []
     testing_scores = []
     for depth in depths:
          regr = DecisionTreeRegressor(max_depth = depth)
          regr.fit(X_train,y_train)
          training_scores.append(regr.score(X_train,y_train))
          testing_scores.append(regr.score(X_test,y_test))
     ##绘图
     fig = plt.figure()
     ax = fig.add_subplot(1,1,1)
     ax.plot(depths,training_scores,label="training score")
     ax.plot(depths,testing_scores,label="testing score")
     ax.set_xlabel('maxdepth')
     ax.set_ylabel('score')
     ax.set_title('Decision Tree Regressor')
     ax.legend(framealpha = 0.5)
     plt.show()



X_train,X_test,y_train,y_test = creat_data(100)   
rgr = DecisionTreeRegressor()
rgr.fit(X_train,y_train)
export_graphviz(rgr,'this')
#test_DecisionTreeRegressor(X_train,X_test,y_train,y_test) 
#test_DecisionTreeRegressor_depth(X_train,X_test,y_train,y_test, maxdepth = 20)
    