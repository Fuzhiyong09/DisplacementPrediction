import numpy as np
import  pandas as pd
from sklearn.ensemble import  RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn import preprocessing

import matplotlib.pyplot as plt
import pandas as pd


OriginData = pd.read_excel(r'C:\Users\ZhiYong\Desktop\UCStest.xlsx')
FeatureData = OriginData.iloc[0:71, :4].values
LabelData = OriginData.iloc[0:71, 4:5].values
TransFeatureData = OriginData.iloc[0:35, 6:10].values
TransLabelData = OriginData.iloc[0:35, 10:11].values

# 构造数据
TTrainX = TransFeatureData[:25, :]
TTrainY = TransLabelData[:25, :].ravel()

TTestX = TransFeatureData[25:, :]
TTestY = TransLabelData[25:, :].ravel()

# 建立随机森林算法
rfa = RandomForestRegressor(n_estimators=500)
bpa = MLPRegressor(hidden_layer_sizes= [5, 5], activation= 'relu', solver= 'adam', learning_rate_init = 0.001, max_iter = 80000)

rfa.fit(TTrainX, TTrainY)
bpa.fit(TTrainX, TTrainY)

rf1_pre = rfa.predict(TTestX)
bp1_pre = bpa.predict(TTestX)

# 测试结果
PreResurf = mean_squared_error(rf1_pre, TTestY)
PreResubp = mean_squared_error(bp1_pre, TTestY)

print(PreResurf, PreResubp)
print(bp1_pre, rf1_pre, TTestY)


