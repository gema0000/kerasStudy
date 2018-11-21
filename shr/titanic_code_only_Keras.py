
# coding: utf-8

# In[1]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[2]:


import os
os.getcwd()


# In[3]:


train_df = pd.read_csv("C:\\kerasStudy\\kaggle\\titanic\\train.csv")
test_df = pd.read_csv("C:\\kerasStudy\\kaggle\\titanic\\test.csv")


# In[4]:


print(train_df.describe())
print("**********************************")
print(train_df.info())


# In[5]:


train_df.head(10)


# In[6]:


total = train_df.isnull().sum().sort_values(ascending=False)


# In[7]:


train_df.isnull().count()


# In[8]:


missing_percent = train_df.isnull().sum()/train_df.isnull().count()*100
missing_percent
missing_percent = (round(missing_percent,1)).sort_values(ascending = False)


# In[9]:


missing_data = pd.concat([total, missing_percent], axis=1, keys=['Total', '%']) #axis =1 => 열방향으로 붙인다
missing_data.head(5)


# In[10]:


train_df.columns.values 


# ## 1. Sex   VS  Survived

# In[11]:


train_df.groupby(['Sex','Survived'])['Survived'].count()


# In[12]:


sns.barplot(x = 'Sex', y = 'Survived', data = train_df)


# In[13]:


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


# In[14]:


genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# ## 2. Pclass  VS	Survived

# In[15]:


train_df.groupby(['Pclass','Embarked','Survived'])['Survived'].count()


# In[16]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[17]:


sns.barplot(x='Pclass', y='Survived', data=train_df) #경제력이 높을 수록 더 살았다


# In[18]:


g = sns.factorplot(data = train_df, x='Pclass', y = 'Survived', hue = 'Sex', ci=95.0, col = 'Embarked', size= 3, aspect= 1)
#Embarked 승선장소 : C 세르부르, Q 퀸스톤, S 사우샘프턴


# In[19]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
#Pclass가 높을수록 즉, 경제력이 낮을수록 20~40대가 더 많이 죽었고, Pclass가 낮을수록 더 많이 살았다
# 경제력과 생존은 연관관계가 있다


# ## 3. Name  VS  Survived

# In[20]:


train_df = train_df.drop(['PassengerId'], axis=1)  #'PassengerId'열 삭제


# In[21]:


train_df['Name'].unique()


# In[22]:


test_df['Name'].unique()


# In[23]:


import re

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data = [train_df, test_df]


# In[24]:


dataset.info()


# In[25]:


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


# In[26]:


data


# ## 4. Parch + SibSp   VS  Survived

# In[27]:


data = [train_df, test_df]

for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0,  'Alone'] = 0  #혼자가 아님
    dataset.loc[dataset['relatives'] == 0, 'Alone'] = 1  #혼자임
    dataset['Alone'] = dataset['Alone'].astype(int)


# In[28]:


axes = sns.barplot('relatives','Survived',data = train_df)


# In[29]:


data


# ## 5. Embarked   VS  Survived

# In[30]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))

ax = sns.barplot(x = 'Embarked',y = 'Survived', hue = 'Sex', data = train_df, ax = axes[1] ) 

ax = sns.barplot(x = 'Embarked',y = 'Survived', data = train_df, ax = axes[0]) 


# In[31]:


common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


# In[32]:


train_df.isnull().sum()


# In[33]:


port = {"S" : 0, "C" : 1, "Q" : 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(port)


# ## 6. Cabin   

# In[34]:


train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# In[35]:


train_df.describe()


# In[36]:


test_df.describe()


# ## 7. Age  VS  Survived 

# In[37]:


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


# In[38]:


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


# In[39]:


data = [train_df, test_df]
    
for dataset in data:
    dataset['Age_Class'] = dataset['Age']* dataset['Pclass']


# In[40]:


train_df.info()


# ## 8. Fare   VS  Survievd

# In[41]:


data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[42]:


for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[43]:


for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)


# ## 9. Ticket   VS  Survived

# In[44]:


train_df['Ticket'].describe()


# In[45]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# In[46]:


train_df.info()


# ## 10. train-test split

# In[56]:


# from sklearn.model_selection import train_test_split

# predictors = train_df.drop(['Survived'], axis=1)
# target = train_df["Survived"]
# x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# ## 10. 5-Fold Validation Modeling

# In[87]:


from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# In[48]:


seed = 1113
np.random.seed(seed)
tf.set_random_seed(seed)


# In[49]:


train_df.head()


# In[60]:


dataset = train_df.values


# In[61]:


X = dataset[:,1:]


# In[62]:


Y_obj = dataset[:,1]


# In[69]:


e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)


# In[70]:


n_fold = 5
skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = seed)


# In[71]:


accuracy = []


# In[76]:


for train, val in skf.split(X,Y):
    model= Sequential()
    model.add(Dense(10,input_dim = 12, activation = "relu"))
    model.add(Dense(1, activation = "softmax"))
    model.compile(loss = 'binary_crossentropy',
                 optimizer = 'adam',
                 metrics = ['accuracy'])
    model.fit(X[train],Y[train], epochs = 20, batch_size = 5)
    k_accuracy = "%.4f" %(model.evaluate(X[val],Y[val])[1])
    
    accuracy.append(k_accuracy)


# In[77]:


print("/n %.f fold accuracy : " %n_fold, accuracy)


# ## 11. Basic Modeling

# In[78]:


X_train = train_df.drop(["Survived"],axis=1).values
y_train= train_df["Survived"].values


# In[83]:


X_test = test_df.drop(["PassengerId"],axis = 1).values.astype(np.float64, copy = False)


# In[84]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[88]:


model= Sequential()
model.add(Dense(10,input_dim = 12, activation = "relu"))
model.add(Dense(1, activation = "softmax"))
model.add(Dropout(0.20))
model.compile(loss = 'binary_crossentropy',
                 optimizer = 'adam',
                 metrics = ['accuracy'])


# In[97]:


model.fit(X_train, y_train, epochs = 100, batch_size = 5)
scores1 = model.evaluate(X_train, y_train, batch_size=5)
print("%s: %.2f%%" % (model.metrics_names[1], scores1[1]*100))


# In[96]:


model.fit(X_train, y_train, epochs = 50, batch_size = 30)
scores2 = model.evaluate(X_train, y_train, batch_size=30)
print("%s: %.2f%%" % (model.metrics_names[1], scores2[1]*100))

