#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def check_df(dataframe,num = 5):
    
    print(" shape ".upper().center(50, "#") , end = "\n\n")
    print(dataframe.shape, end = "\n\n")
    
    print(" types ".upper().center(50, "#") , end = "\n\n")
    print(dataframe.dtypes, end = "\n\n")
    
    print(" head ".upper().center(50, "#") , end = "\n\n")
    print(dataframe.head(num), end = "\n\n")
    
    print(" tail ".upper().center(50, "#") , end = "\n\n")
    print(dataframe.tail(num), end = "\n\n")
    
    print(" na ".upper().center(50, "#"), end = "\n\n")
    print(dataframe.isnull().sum(), end = "\n\n")
    
    print(" dimension ".upper().center(50, "#"), end = "\n\n")
    print(f"{dataframe.ndim} Dimension", end = "\n\n")
    
    print(" quantiles ".upper().center(50,"#"), end = "\n\n")
    print(dataframe.describe([0, 0.01 ,0.05 ,  0.50, 0.95, 0.99, 1]).T , end = "\n\n")


# In[3]:


def cat_summary(dataframe, cat_col , plot = False):
    print(pd.DataFrame({cat_col.upper() : dataframe[cat_col].value_counts() ,
                        "RATIO" : dataframe[cat_col].value_counts() / len(dataframe) * 100}),end = "\n\n")
    
    print("".center(50,"#"),end = "\n\n")
    
    if plot : 
        sns.countplot(x = dataframe[cat_col] , data = dataframe)
        plt.show()


# In[4]:


def num_summary(dataframe, numerical_col , plot = False):
    
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe.describe(quantiles).T, end = "\n\n")
    
    print("".center(50,"#"), end = "\n\n")
    
    if plot:
        sns.histplot(dataframe[numerical_col] , bins = 20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


# In[5]:


def grab_col_names(dataframe , cat_th = 10 , car_th = 20 ,details = False):
    
    
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 returned lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """
 
    # Cat_Cols ,Cat_But_Car
    
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "object"]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype != "object" and                    (dataframe[col].nunique() < cat_th)]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtype == "object" and                    (dataframe[col].nunique() > car_th)]
    
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    # Num_Cols
    
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "object"]
    
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    if details:
        
        print(f"Observations : {dataframe.shape[0]}")
        print(f"Variables : {dataframe.shape[1]}")
        print(f"Cat Cols : {len(cat_cols)}")
        print(f"Num Cols : {len(num_cols)}")
        print(f"Cat But Car : {len(cat_but_car)}")
        print(f"Num But Cat : {len(num_but_cat)}")
    
    return cat_cols , num_cols , cat_but_car


# In[6]:


def target_summary_with_cat(dataframe,target, categorical_col):
    
    print(pd.DataFrame({"TARGET_MEAN" : dataframe.groupby(categorical_col)[target].mean()}) , end = "\n\n")


# In[7]:


def target_summary_with_num(dataframe, target, numerical_col):
    
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end = "\n\n")


# In[8]:


def high_correlated_cols(dataframe, plot=False, corr_th = 0.9):
    
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where( np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool) )
    
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    
    if plot : 
        plt.figure(figsize = (15,15))
        sns.heatmap(corr, cmap = "viridis")
        plt.show()
        
    return drop_list


# In[ ]:




