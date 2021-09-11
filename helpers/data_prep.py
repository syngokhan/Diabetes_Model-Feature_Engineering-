#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# If we do this with the not slice operation, the equality is definitely ensured...

def another_replace_with_thresholds(dataframe, col_name , q1=0.25 , q3 = 0.75):
    
    up_limit, low_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    
    dataframe[ (dataframe[col_name] < low_limit) ][col_name] = low_limit
    dataframe[ (dataframe[col_name] > up_limit)  ][col_name] = up_limit


# In[3]:


def titanic():
    
    dataframe = pd.read_csv("/Users/gokhanersoz/Desktop/VBO_Dataset/titanic.csv")
    
    return dataframe


# In[4]:


def outlier_thresholds(dataframe, col_name , q1 = 0.25 , q3 = 0.75):
    
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile = quantile3 - quantile1
    up_limit = quantile3 + 1.5*interquantile
    low_limit = quantile1 - 1.5*interquantile
    
    return up_limit,low_limit


# In[5]:


def replace_with_thresholds(dataframe, col_name , q1=0.25 , q3 = 0.75):
    up_limit, low_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[ (dataframe[col_name] < low_limit) , col_name ] = low_limit
    dataframe.loc[ (dataframe[col_name] > up_limit) , col_name] = up_limit


# In[6]:


def check_outliers(dataframe, col_name , q1 = 0.25, q3 = 0.75):
    
    up_limit,low_limit= outlier_thresholds(dataframe, col_name , q1 , q3)
    
    results =     dataframe[ (dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis = None)
    
    if results:
        
        result = "There are outliers".title()
        
        return result
    
    else:
        
        result = "There are not outliers".title()
        
        return result


# In[7]:


def grap_outliers(dataframe, col_name , q1 = 0.25, q3 = 0.75 , index = False):
    
    up_limit , low_limit = outlier_thresholds(dataframe, col_name , q1 , q3)
    
    results =     dataframe[ ((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)) ]
    
    results_shape = results.shape[0]
    
    if results_shape > 10:
        
        print(results.head())
    
    else:
        
        print(results)
    
    if index:
        
        outliers_index = dataframe[((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))].index
        outliers_index = outliers_index.tolist()
        
        return outliers_index


# In[8]:


def remove_outliers(dataframe, col_name , q1 = 0.25, q3 = 0.75):
    
    up_limit, low_limit = outlier_thresholds(dataframe, col_name , q1 , q3)
    
    df_without_outliers =     dataframe[ ~((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))]
    
    return df_without_outliers


# In[9]:


def missing_values_table(dataframe, na_name = False):
    
    # We are check True False
    
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] 
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    
    # for columns's name we need to add keys ...
    
    missing_df = pd.concat([n_miss, np.round(ratio, 3)], axis = 1 , keys = ["n_miss", "ratio"] )
    
    print(missing_df,end = "\n\n")
    
    if na_name:
        return na_columns
    


# In[10]:


def another_missing_values_table(dataframe, na_name = False):
    # We are check True False
    
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] 
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    n_miss = pd.DataFrame(n_miss, columns = ["N_Miss"])
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    ratio = pd.DataFrame(ratio, columns = ["Ratio"])
    
    # for columns's name we need to add keys ...
    
    missing_df = pd.concat([n_miss, np.round(ratio, 3)], axis = 1 )
    
    if na_name:
        return na_columns


# In[11]:


def missing_vs_target(dataframe, target , na_columns ):
    
    # df["Age_Flag"] = np.where(df["Age"].isnull() , 1, 0)
    # df[(df["Age"].isnull())].groupby(["Age_Flag"])["Survived"].count() / df[(df["Age"].isnull())] .shape[0]
    # We are already reducing the empty values to 2 variables by making 1 and 0, and accordingly according to the target variable
    # We continue to trade...
    
    """
    We need to use missing_values_table(dataframe, na_name = True)....
    
    1 if missing value 0 if not missing value....
    
    """
    
    
    temp_df = dataframe.copy()
    
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    
    na_flags = temp_df.loc[:,temp_df.columns.str.contains("_NA_")].columns
    
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN" : temp_df.groupby(col)[target].mean(),
                      "Count" : temp_df.groupby(col)[target].count()}), end = "\n\n" )
        
    #return temp_df


# In[12]:


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# In[13]:


def one_hot_encoder(dataframe , categorical_cols , drop_first=False):
    dataframe = pd.get_dummies(dataframe,columns = categorical_cols, drop_first=drop_first)
    return dataframe


# In[14]:


def rare_analyser(dataframe, target, cat_cols):
    
    #test = pd.DataFrame({"Age" : [1,1,1,3,3,3,4,4,4,5,5,5,5,5,6,7,7,7,7],
    #####                 "Survived" :   [1,0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0]})
    #test.groupby("Age")["Survived"].count()
    #test.groupby("Age")["Survived"].mean()
    
    
    """
    Need cat_cols
    
    """
    
    for col in cat_cols:
        print("FOR" , col.upper(), ":", len(dataframe[col].value_counts()),end = "\n\n")
        #print(col, ":", dataframe[col].nunique())
        data = pd.DataFrame({"COUNT" : dataframe[col].value_counts(), \
                            "RATIO" : dataframe[col].value_counts() / len(dataframe), \
                            "TARGET_MEAN" : dataframe.groupby(col)[target].mean()})
        
        data = data.sort_values(by = "TARGET_MEAN", ascending = False)
        
        print(data,end= "\n\n")


# In[15]:


def rare_encoder(dataframe, rare_perc):
    
    """
    
    Object ones are determined.
    
    """
    
    
    temp_df = dataframe.copy()
    
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtype == "object" and                     (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]
    
    for var in rare_columns:
        
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
        
    return temp_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




