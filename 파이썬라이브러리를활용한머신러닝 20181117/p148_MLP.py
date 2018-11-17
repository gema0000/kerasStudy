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

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)



# 신경망 튜닝

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
#                                                     random_state=42)
# print(X_train)
# print(X_train.shape)    # 75, 2

# mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.xlabel("특성 0")
# plt.ylabel("특성 1")

# plt.show()

print("유방암 데이터의 특성별 최대값:\n{}".format(cancer.data.max(axis=0)))

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(mlp.score(X_train, y_train)))   # 0.91
print("테스트 세트 정확도: {:.2f}".format(mlp.score(X_test, y_test)))   # 0.88

# 훈련 세트 각 특성의 평균을 계산합니다
mean_on_train = X_train.mean(axis=0)
# 훈련 세트 각 특성의 표준 편차를 계산합니다
std_on_train = X_train.std(axis=0)

# 데이터에서 평균을 빼고 표준 편차로 나누면
# 평균 0, 표준 편차 1 인 데이터로 변환됩니다.
X_train_scaled = (X_train - mean_on_train) / std_on_train
# (훈련 데이터의 평균과 표준 편차를 이용해) 같은 변환을 테스트 세트에도 합니다
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)

print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled, y_train)))    # 0.991
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled, y_test)))    # 0.965

############# 최대반복횟수가 도달햇다는 경고가 출력되어 adam알고리즘의 반복횟수를 늘린다.
mlp = MLPClassifier(max_iter=1000, random_state=0)  # 반복횟수를 늘림.
mlp.fit(X_train_scaled, y_train)

print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled, y_train)))    # 0.993
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled, y_test)))    # 0.972

################ 가중치를 더 강하게 규제 alpha=0.0001 -> 1 로 변경 
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled, y_train)))    # 0.993
print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled, y_test)))      # 0.972


mlp.coefs_[0].std(axis=1), mlp.coefs_[0].var(axis=1)
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("은닉 유닛")
plt.ylabel("입력 특성")
plt.colorbar()
plt.show()