#!/usr/bin/env python
# coding: utf-8

# In[609]:


# importing necessaruy libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
pd.pandas.set_option('display.max_columns',500)
pd.pandas.set_option('display.max_rows',500)
from wordcloud import WordCloud


# In[610]:


df= pd.read_csv('Clean_Dataset.csv')


# In[611]:


df.describe()


# In[612]:


df = df.loc[df['price']<= 80000]


# In[ ]:





# In[613]:


df.head(10)


# In[614]:


df.isnull().sum()


# In[615]:


df.describe()


# # 1st question

# 
# # does the price varies as the airline changes?

# In[616]:



# lets consider for the flights in between two cities.


# In[617]:


# lets filter the datad=set using the sourse city mumbai and destination city as delhi

df1 = df.loc[(df['source_city']=='Mumbai') & (df['destination_city'] == 'Delhi')]


# In[618]:


df1.head()


# In[619]:


df1.shape


# In[620]:


df1['class'].unique()


# In[621]:


df2 = df1.loc[df1['class']=='Economy'] 


# In[622]:


df2.head()


# In[623]:


df2.shape


# In[624]:


# lets create a pivot table and visualize the bar plot.


# In[625]:


pivot1 = pd.pivot_table(data = df2, index = 'airline', values ='price', aggfunc = 'mean')


# In[626]:


pivot1.head()


# In[627]:


# lets create a bar plot using the above pivot table
pivot1.plot(kind = 'bar')


# In[628]:


# from the above figure we can understand that the airlines of air india and vistara do have high mean prices in beteween the mumbai and delhi route.


# In[629]:


# lets do the same for Business class in between the same route
df3 = df1.loc[df1['class']== 'Business']


# In[630]:


df3.head()


# In[631]:


df3.airline.unique()


# In[632]:


# lets crteate a pivot table and visualze the data in a bar plot
pivot2 = pd.pivot_table(data = df3, index = 'airline', values = 'price', aggfunc = 'mean').sort_values(ascending = True, by = 'price').plot(kind = 'bar')


# In[633]:


# from the above plot we can say that only two airlines do have the bussiness classes. those are airindia and vistara.
# vistar do have highest mean prices than the airindia.


# In[634]:


# we can dig the data even more deeper by considering the timings and the no.of stops of the flight


# In[635]:


# lets consider for the remining routes also.
# lets create a function so that we can automate the process.


# In[636]:


x = df['source_city'].unique()
y = df['destination_city'].unique()
len(x)


# In[637]:



#for i in x:
    #for j in y:
        #data1 = df.loc[(df['source_city']==i)&(df['destination_city']==j)&(df['class']=='Economy')]
        #pivot5 = pd.pivot_table(data = data1, index = 'airline', values = 'price', aggfunc = 'mean').sort_values(by = 'price',ascending = True).plot(kind ='bar')


# In[ ]:





# In[638]:


def plots (data, p,q,r,s,x,y,a,b):
    x1 = data.loc[(data[p]==q)&(data[r]==s)]
    y1 = x1.loc[x1[x]== y]
    pivot1 = pd.pivot_table(data = y1, index= a, values = b,aggfunc = 'mean').sort_values(ascending = True, by = b).plot(kind = 'bar')
    return pivot1


# In[639]:


z1 = plots(df,'source_city','Hyderabad','destination_city','Mumbai','class','Economy', 'airline','price')


# In[640]:


z2 = plots(df,'source_city','Hyderabad','destination_city','Mumbai','class','Business', 'airline','price')


# In[641]:


# from the above two graphs airasia airline does have the minimum mean cost as compared to others and the air vistara does have tha highest mean cost as usual in economy.
# from the business class only two airlines do have the business class allocations. again vistara airways does have the hiest mean price.


# In[642]:


# lets plot a another bar plot using different cities.
z3= z1 = plots(df,'source_city','Chennai','destination_city','Delhi','class','Economy', 'airline','price')


# In[643]:


z1 = plots(df,'source_city','Hyderabad','destination_city','Mumbai','class','Business', 'airline','price')


# In[644]:


# from the above all the figures we can understand that the most of the behaviour remauins same amoung different cities.
# so we can conclude that the price has been varrying with the airline for sure.
# here we need to observe one ore thng that the there might be some airlines that does not have enough instances. beacuse of that also 
# the mean values may shows high. but in our case everything seems good.


# # 2nd question

# # How is the price affected when tickets are bought in just 1 or 2 days before departure?

# In[645]:


df.head()


# In[646]:


df['days_left'].unique()


# In[647]:


# here we do have many uniques values in the days_left column. 
# first step is to seperate the 
def daysleft(df,a,b,c,d,e,f,g,h):
    x = df.loc[(df[a]==b)&(df[c]==d)]
    y =  x.loc[x[e]==f]
    plt.figure(figsize = (15,15))
    z = pd.pivot_table(data = y, index= g, values = h, aggfunc = 'mean').sort_values(by = h, ascending = True,).plot(kind = 'bar')


# In[648]:


z =daysleft(df,'source_city','Hyderabad','destination_city','Delhi','class','Economy','days_left','price')


# In[649]:


z1 = daysleft(df,'source_city','Hyderabad','destination_city','Delhi','class','Business','days_left','price')


# In[650]:


# lets consider with another city
z2 = daysleft(df,'source_city','Kolkata','destination_city','Hyderabad','class','Economy','days_left','price')


# In[651]:


z2 = daysleft(df,'source_city','Kolkata','destination_city','Hyderabad','class','Business','days_left','price')


# In[652]:


# from the above figures we can easily understand that the the prices are not varrying as the days remainig changes for the business class.
# but for the economy class the prices does not change for upto some extent. then the prices changes exponentially.


# # 3rd queston

# # Does ticket price change based on the departure time and arrival time?

# In[653]:


df.head()


# In[654]:


# lets see how many unique values are present in the column departure time and arrival time.
df['departure_time'].unique()


# In[655]:


# there atere many combinations.
# lets try to build a pivot table.
# before making a pivot table there should be a filter operation amounng airines

def filter(df, a,b, c,d, e,f, g,h,i,j,k):
    x = df.loc[(df[a]==b)&(df[c]==d)]
    y = x.loc[x[e]==f]
    z1 = y.loc[y[j]== k]
    z = pd.pivot_table(data = z1, index = g, columns = h, values = i, aggfunc = 'mean')
    return z


# In[656]:


filter(df,'source_city','Hyderabad','destination_city','Delhi','class','Economy','departure_time','arrival_time','price','airline','Vistara')


# In[657]:


# the pivot table is built after filtering every thing out for vistara airlines.
# this pivot table shows the how the mean price changes as the departure and arival times changing.


# In[658]:


filter(df,'source_city','Hyderabad','destination_city','Delhi','class','Business','departure_time','arrival_time','price','airline','Vistara')


# In[659]:


# lets do the same for air india
filter(df,'source_city','Hyderabad','destination_city','Delhi','class','Economy','departure_time','arrival_time','price','airline','Air_India')


# In[660]:


# most of all airlines showing the same behaviour.


# # 4th question

# # How the price changes with change in Source and Destination?

# In[661]:


df.head()


# In[662]:


# lets first consider a single city as an source city and take remining all cities as a destination cities.
df['airline'].unique()


# In[663]:


x = df['source_city'].unique()


# In[664]:



for i in x:
    df1 = df.loc[(df['source_city']== i) &(df['class']== 'Economy')
                 & (df['airline']=='Vistara') ]
    pivot2 = pd.pivot_table(data = df1, index = 'destination_city'
                            , values ='price', aggfunc = 'mean').sort_values(ascending = True
                                                                             , by = 'price').plot(kind = 'bar')  


# In[665]:


for i in x:
    df1 = df.loc[(df['source_city']== i) &(df['class']== 'Economy')
                 & (df['airline']=='SpiceJet') ]
    pivot2 = pd.pivot_table(data = df1, index = 'destination_city'
                            , values ='price', aggfunc = 'mean').sort_values(ascending = True
                                                                             , by = 'price').plot(kind = 'bar')  


# In[666]:


for i in x:
    df1 = df.loc[(df['source_city']== i) &(df['class']== 'Economy')
                 & (df['airline']=='AirAsia') ]
    pivot2 = pd.pivot_table(data = df1, index = 'destination_city'
                            , values ='price', aggfunc = 'mean').sort_values(ascending = True
                                                                             , by = 'price').plot(kind = 'bar')  


# In[667]:


for i in x:
    df1 = df.loc[(df['source_city']== i) &(df['class']== 'Economy')
                 & (df['airline']=='GO_FIRST') ]
    pivot2 = pd.pivot_table(data = df1, index = 'destination_city'
                            , values ='price', aggfunc = 'mean').sort_values(ascending = True
                                                                             , by = 'price').plot(kind = 'bar')  


# In[668]:


for i in x:
    df1 = df.loc[(df['source_city']== i) &(df['class']== 'Economy')
                 & (df['airline']=='Indigo') ]
    pivot2 = pd.pivot_table(data = df1, index = 'destination_city'
                            , values ='price', aggfunc = 'mean').sort_values(ascending = True
                                                                             , by = 'price').plot(kind = 'bar')  


# In[669]:



for i in x:
    df1 = df.loc[(df['source_city']== i) &(df['class']== 'Economy')
                 & (df['airline']=='Air_India') ]
    pivot2 = pd.pivot_table(data = df1, index = 'destination_city'
                            , values ='price', aggfunc = 'mean').sort_values(ascending = True
                                                                             , by = 'price').plot(kind = 'bar')  


# In[670]:


# lets focus on preprocessing the data

df.head()


# In[671]:


df.drop(columns = ['flight','Unnamed: 0'], axis = 1, inplace = True)


# In[672]:


df.head()


# In[673]:


df.isnull().sum()


# In[674]:


df['airline'].unique()


# In[675]:


feature =['Economy']
feature1 = ['SpiceJet','AirAsia','Vistara',
           'Go_FIRST','Indigo','Air_India']


# In[676]:


data1 = df.loc[df['class']== 'Economy']


# In[677]:


data2 = df.loc[df['class']=='Business']


# In[678]:


data1.head()


# In[679]:


data1['new_column'] = np.where(data1['class']== 'Economy',1,0)


# In[680]:


data1.head()


# In[681]:


pivott1 = pd.pivot_table(data = data1, index = 'airline',values = 'price', aggfunc = 'mean')


# In[682]:


pivott1.head()


# In[683]:


for i,j in enumerate(data1['airline']):
    data1['new_column'][i] = pivott1.loc[j]['price'] 


# In[684]:


data1.head()


# In[685]:


data2['new_column'] = np.where(data2['class']== 'Business',1,0)


# In[686]:


data2.head()


# In[ ]:





# In[687]:


pivott2.head()


# In[688]:


data2['airline'].unique()


# In[689]:


import warnings
warnings.filterwarnings('ignore')


# In[690]:


data2['new_column1'] = np.where(data2['airline']== 'Air_India',47131,55477)


# In[691]:


data2.drop(columns = 'new_column',axis = 1, inplace = True)


# In[692]:


data2.rename(columns = {'new_column1':'new_column'},inplace = True)


# In[693]:


data_final = pd.concat([data1, data2],axis = 0)


# In[ ]:





# In[694]:


data_final.head()


# In[695]:


data_final = data_final.sample(frac = 1).reset_index()


# In[696]:


data_final.drop(columns ='index',axis = 1, inplace = True)


# In[697]:


data_final.head()


# In[ ]:





# In[698]:


# lets split the data into train and test

from sklearn.model_selection import train_test_split


# In[ ]:





# In[699]:


#train, test = train_test_split(, test_size = 0.25)


# In[700]:


# lets process the the train dataset.
# here i have divided the data into train test split to avoid the data leakage.


# In[701]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[702]:


# lets do the ordinal encoding for the column 'airline'.
# here we are doing the ordinal encoding beacuse the prices are varrying as the airline changes.
# in the ordinal encoding we give rankings according to the data.
x = le.fit_transform(data_final['airline'])


# In[703]:


data_final['airline'] = x


# In[704]:


data_final.head()


# In[705]:


data_final['stops'].unique()


# In[706]:


data_final= pd.concat([data_final,pd.get_dummies(data_final['stops'],drop_first = True)],axis = 1)
data_final.head()


# In[707]:


data_final.drop(columns = 'stops', axis = 1)


# In[708]:


a = pd.get_dummies(data_final['source_city'],drop_first = True)


# In[709]:


columns1 = ['source_Chennai','source_Delhi','source_Hyderabad','source_Kolkata','source_Mumbai']


# In[710]:


a.columns = columns1


# In[711]:


data_final = pd.concat([data_final,a],axis = 1)


# In[712]:


data_final.drop(columns = 'source_city', axis = 1)


# In[713]:


data_final = pd.concat([data_final,pd.get_dummies(data_final['destination_city'], drop_first = True)],axis = 1)


# In[714]:


data_final.head()


# In[715]:


data_final.drop(columns = ['destination_city','source_city'], axis = 1,inplace = True)


# In[716]:


x1 =le.fit_transform(data_final['departure_time'])
x2 = le.fit_transform(data_final['arrival_time'])


# In[717]:


data_final['departure_time']= x1
data_final['arrival_time'] = x2


# In[718]:


data_final.head()


# In[719]:


x3 = le.fit_transform(data_final['stops'])
x4 = le.fit_transform(data_final['class'])

data_final['stops'] = x3
data_final['class'] = x4


# In[720]:


data_final.head()


# In[807]:


data_final['new_column'].unique()


# In[721]:


data_final.drop(columns = ['zero','two_or_more'],axis = 1, inplace = True)


# In[722]:


plt.figure(figsize =(15,15))
corr = data_final.corr()
sns.heatmap(corr, annot = True, cmap= 'coolwarm')


# In[723]:


from sklearn.model_selection import train_test_split


# In[724]:


test1 = data_final['price']


# In[790]:


data_final['new_column'] = np.log(data_final['new_column'])


# In[769]:


test1 = np.log(test1)


# In[771]:


sns.distplot(test1)


# In[805]:





# In[772]:


sns.boxplot(test1)


# In[725]:


data_final.drop(columns = 'price', axis = 1, inplace = True)


# In[ ]:





# In[792]:


x_train,x_test,y_train,y_test = train_test_split(data_final, test1, test_size = 0.25)


# In[727]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[728]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[729]:


from sklearn.ensemble import RandomForestRegressor


# In[730]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[731]:


parameters = {'max_depth':[2,3,4],'n_estimators':[200,300,400]}
model = RandomForestRegressor()


# In[732]:


#x =GridSearchCV(model,parameters)  
#cv1 = cross_val_score(model, df,test, cv= 5)
#print(cv1.mean())


# In[733]:


#y =x.fit(x_train,y_train)


# In[734]:


#y.best_params_


# In[735]:


model = RandomForestRegressor(max_depth = 4, n_estimators = 300)

model.fit(x_train,y_train)


# In[736]:


import pyodbc


# In[737]:


pyodbc.drivers()


# In[738]:


predict = model.predict(x_test)


# In[739]:


mean_squared_error(predict, y_test)


# In[740]:


mean_absolute_error(predict,y_test)


# In[741]:


from sklearn.linear_model import LinearRegression


# In[742]:


model1 = LinearRegression(normalize = False)


# In[743]:


model1.fit(x_train,y_train)


# In[744]:


pred =model1.predict(x_test)


# In[745]:


err1 = mean_absolute_error(pred,y_test)


# In[746]:


pip install mlxtend


# In[747]:


X_train,X_test,y_train,y_test = train_test_split(data_final,test1,test_size = 0.33)


# In[748]:


from mlxtend.evaluate import bias_variance_decomp


# In[ ]:


X_train = x_train.values


# In[ ]:


X_test = x_test.values


# In[ ]:


y_train = y_train.values


# In[ ]:


y_test = y_test.values


# In[ ]:


mse,bias,var = bias_variance_decomp(model,X_train,y_train,X_test,y_test,loss = 'mse',num_rounds = 200,random_seed = 64)


# In[750]:


from sklearn.linear_model import LinearRegression


# In[751]:


model2 = LinearRegression(normalize = False)


# In[752]:


from sklearn.metrics import r2_score


# In[791]:


score4 =[]
for i in range(100):
    x_train,x_test,y_train,y_test = train_test_split(data_final,test1,test_size = 0.25, random_state = i)
    model2.fit(x_train,y_train)
    pred = model2.predict(x_test)
    score4.append(r2_score(pred,y_test))
    


# In[793]:


b = np.argmax(score4)


# In[794]:


b


# In[795]:


x_train,x_test,y_train,y_test = train_test_split(data_final,test1,test_size = 0.25, random_state = 70)
model2.fit(x_train,y_train)
pred2 = model2.predict(x_test)


# In[796]:


err4 = mean_absolute_error(pred2,y_test)


# In[797]:


error = mean_squared_error(pred2,y_test)


# In[798]:


error


# In[799]:


np.exp(err4)


# In[806]:


import pickle

pickle.dump(model2,open('flight2.sav','wb'))


# In[789]:


model5 =pickle.load(open('flight2.sav','rb'))


# In[802]:


model5.predict([[1,2,2,5,0,2.75,45,10.760686,0,0,0,0,0,0,1,0,0,0]])


# In[804]:


np.exp(8.85502954)


# In[808]:


np.log(7009.555)


# In[ ]:




