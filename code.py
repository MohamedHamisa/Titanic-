# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()

## statistical info
train.describe()

## datatype info
train.info()

## categorical attributes
sns.countplot(train['Survived'])

sns.countplot(train['Pclass'])

sns.countplot(train['Sex'])

sns.countplot(train['SibSp'])

sns.countplot(train['Parch'])

sns.countplot(train['Embarked'])

## numerical attributes
sns.distplot(train['Age'])

sns.distplot(train['Fare'])

class_fare = train.pivot_table(index='Pclass', values='Fare')
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Avg. Fare')
plt.xticks(rotation=0)
plt.show()

class_fare = train.pivot_table(index='Pclass', values='Fare', aggfunc=np.sum)
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Total Fare')
plt.xticks(rotation=0)
plt.show()

sns.barplot(data=train, x='Pclass', y='Fare', hue='Survived')

sns.barplot(data=train, x='Survived', y='Fare', hue='Pclass')

train_len = len(train)
# combine two dataframes
df = pd.concat([train, test], axis=0)
df = df.reset_index(drop=True)
df.head()

df.tail()

## find the null values
df.isnull().sum()

# drop or delete the column
df = df.drop(columns=['Cabin'], axis=1)
df['Age'].mean()

# fill missing values using mean of the numerical column
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'].mode()[0]

# fill missing values using mode of the categorical column
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

sns.distplot(df['Fare'])

df['Fare'] = np.log(df['Fare']+1)
sns.distplot(df['Fare'])

corr = df.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm')

df.head()

## drop unnecessary columns
df = df.drop(columns=['Name', 'Ticket'], axis=1)
df.head()

from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()

train = df.iloc[:train_len, :]
test = df.iloc[train_len:, :]
train.head()

test.head()

# input split
X = train.drop(columns=['PassengerId', 'Survived'], axis=1)
y = train['Survived']

X.head()

from sklearn.model_selection import train_test_split, cross_val_score
# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))
    
    score = cross_val_score(model, X, y, cv=5)
    print('CV Score:', np.mean(score))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model)

from xgboost import XGBClassifier
model = XGBClassifier()
classify(model)

from lightgbm import LGBMClassifier
model = LGBMClassifier()
classify(model)

from catboost import CatBoostClassifier
model = CatBoostClassifier(verbose=0)
classify(model)

model = LGBMClassifier()
model.fit(X, y)

test.head()

# input split for test data
X_test = test.drop(columns=['PassengerId', 'Survived'], axis=1)
X_test.head()

pred = model.predict(X_test)
pred

#test Submission
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
sub.head()

sub.info()

sub['Survived'] = pred
sub['Survived'] = sub['Survived'].astype('int')
sub.info()

sub.head()

sub.to_csv('submission.csv', index=False)  #to keep the file clean from wierd characters


