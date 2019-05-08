import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
%matplotlib inline
#read the dataset
df =pd.read_csv('HR.csv')
df.head(3)
#data exploration and visulization'
left = df[df.left==1]#it will retirn us those employees that left the HR company
retained = df[df.left==0]#it return us those employees whose continue their work in HR company
#calculate the average number foe all column
df.groupby('left').mean()

# From above table we can draw following conclusions,

# **Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
# **Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
# **Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm
pd.crosstab(df.salary,df.left).plot(kind='bar')#crosstab is called the crosstabulation used in pandas
plt.show()

# Above bar chart shows employees with high salaries are likely to not leave the company
pd.crosstab(df.Department,df.left).plot(kind='bar')
# From above chart there seem to be some impact of department on employee retention but it is not major hence we will ignore department in our analysis
# From the data analysis so far we can conclude that we will use following variables as dependant variables in our model
# **Satisfaction Level**
# **Average Monthly Hours**
# **Promotion Last 5 Years**
# **Salary**
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
#from above all the column are in number except the salary, so we conveert it into a number,it is ordianl from ,,
#by using the dummy variable we convert it into the number
dummies = pd.get_dummies(subdf.salary)

#merged these two datasets
merged = pd.concat([subdf,dummies],axis =1)
#drop the oringinal salary column and one dummy column
final = merged.drop(['salary','medium'],axis =1)
#get the x and y for training the model
x = final
y =df.left
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
log = LogisticRegression()
log.fit(x_train,y_train)
log.predict(x_test)
log.score(x_train,y_train)*100
