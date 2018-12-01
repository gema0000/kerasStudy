
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 데이터 적재와 분할
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# 훈련 데이터의 최솟값, 최댓값을 계산합니다.
scaler = MinMaxScaler().fit(X_train)
scaler2 = StandardScaler().fit(X_train)

# 훈련 데이터의 스케일을 조정합니다.
X_train_scaled = scaler.transform(X_train)
X_train_scaled2 = scaler2.transform(X_train)

svm = SVC()
# 스케일 조정된 훈련 데이터에 SVM을 학습시킵니다.
svm.fit(X_train_scaled, y_train)
# 테스트 데이터의 스케일을 조정하고 점수를 계산합니다.
X_test_scaled = scaler.transform(X_test)
print("테스트 점수 : {:.3f}".format(svm.score(X_test_scaled, y_test)))  # 정규화 : 0.951

svm.fit(X_train_scaled2, y_train)
X_test_scaled2 = scaler2.transform(X_test)
print("테스트 점수 : {:.3f}".format(svm.score(X_test_scaled2, y_test)))  # 표준화 : 0.965

# 정규화후 표준화
scaler3 = StandardScaler().fit(X_train_scaled)
X_train_scaled3 = scaler3.transform(X_train_scaled)

svm.fit(X_train_scaled3, y_train)
X_test_scaled3 = scaler3.transform(X_test)
print("테스트 점수 : {:.3f}".format(svm.score(X_test_scaled3, y_test)))  # 정규화후 표준화 : 0.371   # 망하네 ㅋㅋㅋ



