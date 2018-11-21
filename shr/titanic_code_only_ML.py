
# coding: utf-8

# In[2]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[3]:


import os
os.getcwd()


# In[4]:


train_df = pd.read_csv("C:\\kerasStudy\\kaggle\\titanic\\train.csv")
test_df = pd.read_csv("C:\\kerasStudy\\kaggle\\titanic\\test.csv")


# In[5]:


print(train_df.describe())
print("**********************************")
print(train_df.info())


# In[6]:


train_df.head(10)


# In[7]:


total = train_df.isnull().sum().sort_values(ascending=False)


# In[8]:


train_df.isnull().count()


# In[9]:


missing_percent = train_df.isnull().sum()/train_df.isnull().count()*100
missing_percent
missing_percent = (round(missing_percent,1)).sort_values(ascending = False)


# In[10]:


missing_data = pd.concat([total, missing_percent], axis=1, keys=['Total', '%']) #axis =1 => 열방향으로 붙인다
missing_data.head(5)


# In[11]:


train_df.columns.values 


# ## 1. Sex   VS  Survived

# In[12]:


train_df.groupby(['Sex','Survived'])['Survived'].count()


# In[13]:


sns.barplot(x = 'Sex', y = 'Survived', data = train_df)


# In[14]:


survived = 'survived'
not_survived = 'dead'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']

# 왼쪽 그래프_여자
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False) # 여자중에 살았으면_파란
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=18, label = not_survived, ax = axes[0], kde =False) # 여자중에 죽었으면
ax.legend()
ax.set_title('Female')

# 오른쪽 그래프_남자
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False) # 남자중에 살았으면_파란
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False) # 남자 중에 죽었으면
ax.legend()
_ = ax.set_title('Male')


# In[15]:


genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# ## 2. Pclass  VS	Survived

# In[16]:


train_df.groupby(['Pclass','Embarked','Survived'])['Survived'].count()


# In[17]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[18]:


sns.barplot(x='Pclass', y='Survived', data=train_df) #경제력이 높을 수록 더 살았다


# In[19]:


g = sns.factorplot(data = train_df, x='Pclass', y = 'Survived', hue = 'Sex', ci=95.0, col = 'Embarked', size= 3, aspect= 1)
#Embarked 승선장소 : C 세르부르, Q 퀸스톤, S 사우샘프턴


# In[20]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
#Pclass가 높을수록 즉, 경제력이 낮을수록 20~40대가 더 많이 죽었고, Pclass가 낮을수록 더 많이 살았다
# 경제력과 생존은 연관관계가 있다


# ## 3. Name  VS  Survived

# In[21]:


train_df = train_df.drop(['PassengerId'], axis=1)  #'PassengerId'열 삭제


# In[22]:


train_df['Name'].unique()


# In[23]:


test_df['Name'].unique()


# In[24]:


import re

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data = [train_df, test_df]


# In[25]:


dataset.info()


# In[26]:


for dataset in data:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


# In[27]:


data


# ## 4. Parch + SibSp   VS  Survived

# In[28]:


data = [train_df, test_df]

for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0,  'Alone'] = 0  #혼자가 아님
    dataset.loc[dataset['relatives'] == 0, 'Alone'] = 1  #혼자임
    dataset['Alone'] = dataset['Alone'].astype(int)


# In[29]:


axes = sns.barplot('relatives','Survived',data = train_df)


# In[30]:


data


# ## 5. Embarked   VS  Survived

# In[31]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))

ax = sns.barplot(x = 'Embarked',y = 'Survived', hue = 'Sex', data = train_df, ax = axes[1] ) 

ax = sns.barplot(x = 'Embarked',y = 'Survived', data = train_df, ax = axes[0]) 


# In[32]:


common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


# In[33]:


train_df.isnull().sum()


# In[34]:


port = {"S" : 0, "C" : 1, "Q" : 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(port)


# ## 6. Cabin   

# In[35]:


train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# In[36]:


train_df.describe()


# In[37]:


test_df.describe()


# ## 7. Age  VS  Survived 

# In[38]:


data = [train_df, test_df] # PassengerID, Cabin 없는

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()

    #np.random.randint(시작, 끝, 반환갯수) => 지정된 범위에서 숫자를 반환한다
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    age_slice = dataset["Age"].copy() #복사
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)

print(train_df["Age"].isnull().sum())


# In[39]:


data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 25), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 35), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[40]:


data = [train_df, test_df]
    
for dataset in data:
    dataset['Age_Class'] = dataset['Age']* dataset['Pclass']


# In[41]:


train_df.info()


# ## 8. Fare   VS  Survievd

# In[42]:


data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[43]:


for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[44]:


for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)


# ## 9. Ticket   VS  Survived

# In[45]:


train_df['Ticket'].describe()


# In[46]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# In[47]:


train_df.info()


# In[48]:


train_df.info()


# ## 10. train-test split

# In[51]:


from sklearn.model_selection import train_test_split

predictors = train_df.drop(['Survived'], axis=1)
target = train_df["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# ## 10.Modeling

# ### A) Random Forest

# In[53]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(x_train, y_train)

Y_prediction = random_forest.predict(x_val)

random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# ### B) Logistic Regression

# In[54]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

Y_pred = logreg.predict(x_val)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)
print(round(acc_log,2,), "%")


# ### D) SVM

# In[56]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# ### E) Linear SVC

# In[57]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# ### F)  Gaussian Naive Bayes

# In[59]:


from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# ### G) Decision Tree

# In[60]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# ### H) Gradient Boosting Classifier

# In[61]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
y_pred = gbc.predict(x_val)
acc_gbc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbc)


# ## 정리 

# In[63]:


using_models = pd.DataFrame({
    'Model': ['Random Forest','Logistic Regression','Support Vector Machines','Linear SVC',
              'Gaussian Naive Bayes', 'Decision Tree', 'Gradient Boosting Classifier'],
    'Score': [acc_random_forest,acc_log, acc_svc, acc_linear_svc,
              acc_gaussian, acc_decisiontree,acc_gbc]})
using_models.sort_values(by='Score', ascending=False)


# ##  Creating Submission File

# In[64]:


#set ids as PassengerId and predict survival 
ids = test_df['PassengerId']
predictions = random_forest.predict(test_df.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

