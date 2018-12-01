import mglearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

################# matplotlib 한글 구현 #############################
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
####################################################################

citibike = mglearn.datasets.load_citibike()

print(citibike.shape)
print(type(citibike))
print("시티바이크 데이터:\n{}".format(citibike.head()))

# 타깃값 추출
y = citibike.values
# POSIX 시간을 10**9로 나누어 변환
X= citibike.index.astype("int64").values.reshape(-1,1)// 10**9
print(X[:5], y[:5])
print(X.shape, y.shape)


plt.figure(figsize=(10,7))
xticks = pd.date_range(start=citibike.index.min(), end = citibike.index.max(), freq='D')
week = ["일","월","화","수","목","금","토"]
xticks_name = [week[int(w)]+d for w, d in zip(xticks.strftime("%w"), xticks.strftime(" %m-%d"))]
plt.xticks(xticks, xticks_name, rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("날짜")
plt.ylabel("대여횟수")
# plt.show()

n_train = 184
def eval_on_features(features, target, regressor):
    # 훈련 세트와 테스트 세트로 나눕니다.
    X_train, X_test = features[:n_train], features[n_train:]
    # 타깃값도 나눕니다.
    y_train, y_test = target[:n_train], target[n_train:]
    print("=================================")
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print("=================================")
    regressor.fit(X_train, y_train)
    print("테스트 세트 R^2 : {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize=(10,3))
    plt.xticks(range(0, len(X), 8), xticks_name, rotation=90, ha="left")
    plt.plot(range(n_train), y_train, label="훈련")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="테스트")
    plt.plot(range(n_train), y_pred_train, '--', label="훈련 예측")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="테스트 예측")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("날짜")
    plt.ylabel("대여횟수")
    plt.show()

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# eval_on_features(X, y, regressor)

X_hour = citibike.index.hour.values.reshape(-1,1)
# print(X_hour)
# print(X_hour.shape)
# eval_on_features(X_hour, y, regressor)

X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1),    # 요일 정보 추가
                         citibike.index.hour.values.reshape(-1,1) ])
# print(X_hour_week)
print(X_hour_week.shape)

# eval_on_features(X_hour_week, y, regressor)   # 0.84

# 랜포 -> LinearRegression 으로 변경
from sklearn.linear_model import LinearRegression
# eval_on_features(X_hour_week, y, LinearRegression())        # 0.13

# OneHotEncoder 사용
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
print(X_hour_week_onehot[:5])
print(X_hour_week_onehot.shape)

eval_on_features(X_hour_week_onehot, y, LinearRegression())        # 0.62

