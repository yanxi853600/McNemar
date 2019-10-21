#(1008 homework)研究方法
#CART、knn做McNemar，並比較準確性是否有顯著差異

import pandas as pd
from sklearn import model_selection  
from sklearn.metrics import confusion_matrix #计算混淆矩阵，主要来评估分类的准确性
#from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

#download dataset
dataset=pd.read_csv("avocado.csv")

# Split out validation dataset
array = dataset.values #將數據庫->數組
X = array[:,1:7] 
Y = array[:,10] 
validation_size = 0.20 #驗證集規模
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) #分割数据集

scoring = 'accuracy'

models = [] #建立列表
models.append(('CART', DecisionTreeClassifier()))

results = []
names = []
for name, model in models: #將算法名稱與算法函數分別讀取
	kfold = model_selection.KFold(n_splits=10, random_state=seed) #建立10倍交叉驗證
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) #每一个算法模型作為其中的參數，算每一模型的精度得分
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
	print(msg)


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation) #預測驗證集
print(confusion_matrix(Y_validation, predictions)) 
#混淆矩中的i,j指的是觀察數目i，預測為j

#卡方檢定(比較觀測值與期望值是否顯著差異)
rows = 2
columns =2
df = (rows-1)*(columns-1)
print("自由度:", df)

crit = stats.chi2.ppf(q = 0.95, df=df)
print("臨界區: ", crit)


# McNemar Test
table = [[177, 42], [88, 1746]]
result = mcnemar(table, exact=True)
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

alpha = 0.05
if result.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')


