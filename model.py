import pandas as pd
import matplotlib
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('Bengaluru_House_Data.csv')
df1=df.drop(columns='society')
df1.dropna(inplace=True)
df1['size']=df1['size'].replace(['1 RK','1 Bedroom'],'1 BHK')
df1['size']=df1['size'].replace('2 Bedroom','2 BHK')
df1['size']=df1['size'].replace('3 Bedroom','3 BHK')
df1['size']=df1['size'].replace('4 Bedroom','4 BHK')
df1['size']=df1['size'].replace('5 Bedroom','5 BHK')
df1['size']=df1['size'].replace('6 Bedroom','6 BHK')
df1['size']=df1['size'].replace('7 Bedroom','7 BHK')
df1['size']=df1['size'].replace('8 Bedroom','8 BHK')
df1['size']=df1['size'].replace('9 Bedroom','9 BHK')
df1['size']=df1['size'].replace('10 Bedroom','10 BHK')
df1['size']=df1['size'].replace('11 Bedroom','11 BHK')
df1['size']=df1['size'].replace('12 Bedroom','12 BHK')
df1['size']=df1['size'].replace('43 Bedroom','43 BHK')
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df1[~df1['total_sqft'].apply(is_float)].head(10)
def convert_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df2 = df1.copy()
df2['total_sqft'] = df2['total_sqft'].apply(convert_to_num)
df2['price_per_sqft'] = df2['price']*100000/df2['total_sqft']
df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
df2.location = df2.location.apply(lambda x: x.strip())
location_stats = df2.groupby('location')['location'].count().sort_values(ascending = False)
location_stats_lessthan_10 = location_stats[location_stats<=10]
df2.location = df2.location.apply(lambda x: 'other' if x in location_stats_lessthan_10 else x)
df3 = df2[~(df2.total_sqft/df2.bhk<300)]
def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)
    return df_out

df4 = remove_outliers(df3)
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df5 = remove_bhk_outliers(df4)
df6 = df5[df5.bath < df5.bhk+2]
df7 = df6.drop(['size','price_per_sqft'],axis='columns')
dummies = pd.get_dummies(df7.location)
df8 = pd.concat([df7,dummies.drop('other',axis='columns')],axis='columns')
df9 = df8.drop('location',axis='columns')
X = df9.drop(['price','availability','area_type'],axis='columns')
y = df9.price
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    if lr_clf.predict([x])[0] >0:
        return lr_clf.predict([x])[0]
    else:
        return lr_clf.predict([x])[0] * -1

# Saving model to disk
pickle.dump(lr_clf, open('model.pkl','wb'))

# Loading model to compare the results
lr_clf = pickle.load(open('model.pkl','rb'))
print(predict_price('1st Phase JP Nagar',1000, 2, 2))
