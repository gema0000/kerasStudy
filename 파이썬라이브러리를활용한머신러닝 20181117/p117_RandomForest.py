from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd

################# matplotlib 한글 구현 #############################
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
####################################################################



from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons                 # two_moon 데이터셋

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2, n_jobs=-1)     # tree 5개로 구성 // n_jobs=-1 컴퓨터의 모든 코어 사용
aaa = forest.fit(X_train, y_train)

print("훈련 세트 점수: {:.2f}".format(aaa.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(aaa.score(X_test, y_test)))

print("===================================")

print(y_train)
print(X_train.shape)
print(y_train.shape)

# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify = cancer.target, random_state=42)
# forest = RandomForestClassifier(n_estimators=100, random_state=2)       # tree 100개인가? ㄷㄷ
# forest.fit(X_train, y_train)

# print("훈련 세트 점수: {:.2f}".format(forest.score(X_train, y_train)))
# print("테스트 세트 점수 : {:.2f}".format(forest.score(X_test, y_test)))

# plot_feature_importances_cancer(forest)

# plt.show()