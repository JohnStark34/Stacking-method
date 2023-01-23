from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor as GBDT
from sklearn.ensemble import ExtraTreesRegressor as ET
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import AdaBoostRegressor as ADA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn. model_selection import train_test_split

df = pd.read_csv("data.csv")
X = df.iloc[:, :19]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_cols = X_train.columns
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values

models = [GBDT(n_estimators=100),
          RF(n_estimators=100),
          ET(n_estimators=100),
          ADA(n_estimators=100)]

X_train_stack = np.zeros((X_train.shape[0], len(models)))
X_test_stack = np.zeros((X_test.shape[0], len(models)))

# 10折stacking
n_folds = 10
kf = KFold(n_splits=n_folds)

for i, model in enumerate(models):
    X_stack_test_n = np.zeros((X_test.shape[0], n_folds))

    for j, (train_index, test_index) in enumerate(kf.split(X_train)):
        tr_x = X_train[train_index]
        tr_y = y_train[train_index]
        model.fit(tr_x, tr_y)

        # 生成stacking训练数据集
        X_train_stack[test_index, i] = model.predict(X_train[test_index])
        X_stack_test_n[:, j] = model.predict(X_test)

    # 生成stacking测试数据集
    X_test_stack[:, i] = X_stack_test_n.mean(axis=1)

    model_second = LinearRegression()
    model_second.fit(X_train_stack, y_train)
    pred = model_second.predict(X_test_stack)
    print("R1:", r2_score(y_test, pred))

    # GBDT
    model_1 = models[0]
    model_1.fit(X_train, y_train)
    pred_1 = model_1.predict(X_test)
    print("R2:", r2_score(y_test, pred_1))

    # RF
    model_2 = models[1]
    model_2.fit(X_train, y_train)
    pred_2 = model_2.predict(X_test)
    print("R3:", r2_score(y_test, pred_2))

    # ET
    model_3 = models[2]
    model_3.fit(X_train, y_train)
    pred_3 = model_1.predict(X_test)
    print("R4:", r2_score(y_test, pred_3))

    # ADA
    model_4 = models[3]
    model_4.fit(X_train, y_train)
    pred_4 = model_4.predict(X_test)
    print("R5:", r2_score(y_test, pred_4))

    print('stacking model')
    print("loss is {}".format(mean_squared_error(y_test, pred)))
    plt.scatter(np.arange(len(pred)), pred)
    plt.plot(np.arange(len(y_test)), y_test)
    plt.show()

    from sklearn import metrics

    fpr, tpr, _ = metrics.roc_curve(y_test, pred)
    # y_test_one_hot:one-hot  测试集(编码后)的真实值y
    # y_test_one_hot_hat: one-hot  测试集(编码后)的预测值y
from sklearn import metrics

print ('Micro AUC:\t', metrics.auc(fpr, tpr))    # AUC ROC意思是ROC曲线下方的面积(Area under the Curve of ROC)
print( 'Micro AUC(System):\t', metrics.roc_auc_score(y_test, pred, average='micro'))
auc = metrics.roc_auc_score(y_test, pred, average='macro')
print ('Macro AUC:\t', auc)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 7), dpi=80, facecolor='w')    # dpi:每英寸长度的像素点数；facecolor 背景颜色
plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))  #绘制刻度
plt.yticks(np.arange(0, 1.1, 0.1))
plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)  # 绘制AUC 曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc='lower right')    # 设置显示标签的位置
plt.xlabel('False Positive Rate', fontsize=14)   #绘制x,y 坐标轴对应的标签
plt.ylabel('True Positive Rate', fontsize=14)
plt.grid(b=True, ls=':')  # 绘制网格作为底板;b是否显示网格线；ls表示line style
plt.title('ROC curve And  AUC', fontsize=18)  # 打印标题
plt.show()
