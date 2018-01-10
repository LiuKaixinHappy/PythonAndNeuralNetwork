import numpy as np
import pandas as pd
# TODO：读取SleepQuality.csv中的数据
data = pd.read_csv('SleepQuality.csv')

# TODO：选择你认为对预测睡眠质量有用的特征，必要时通过pandas的get_dummies方法将非数值型变量变成数值型变量
data = pd.concat([data, pd.get_dummies(data['Sex'], prefix='Sex')], axis=1)
data = data.drop('Sex', axis=1)
data = data.drop('Source', axis=1)
data = data.drop('Number', axis=1)

# 特征归一化
for field in ['Age', 'Reliability','Psychoticism','Nervousness','Character']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std

print data.head()
    
# 随机取10%的数据作为测试集
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

    
# 将训练集和测试集都分为features 和 targets 两部分
features, targets = data.drop('Sleep quality', axis=1), data['Sleep quality']
features_test, targets_test = test_data.drop('Sleep quality', axis=1), test_data['Sleep quality']

# TODO:定义一个合适的激活函数
def activate_function(x):
    return 1 / (1 + np.exp(-x))

n_records, n_features = features.shape
last_loss = None

# TODO：初始化权重矩阵
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# TODO：设置代数和学习率
epochs = 1000
learnrate = 0.8

# TODO：实现神经网络模型
for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        output = activate_function(np.dot(x, weights))
       
        error = y - output

        error_term = error * output * (1 - output)

        
        del_w += error_term * x
      
    weights += learnrate * del_w / len(x)

    # 输出训练集的MSE
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))