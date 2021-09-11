#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[3]:


from helpers.eda import *
from helpers.data_prep import *


# In[4]:


pd.set_option("display.max_columns", None)
pd.set_option("display.float_format" , lambda x : "%.4f" %x)
pd.set_option("display.width", 200)


# In[5]:


def check_na_values(dataframe):
    
    na_values = dataframe.isnull().sum()
    na_values = pd.DataFrame(na_values)
    na_values.columns = ["NA_Values"]
    na_values = na_values[na_values["NA_Values"] > 0 ]
    return na_values


# In[6]:


path = "/Users/gokhanersoz/Desktop/VBO_Dataset/diabetes.csv"


# In[7]:


diabetes = pd.read_csv(path)


# In[8]:


df = diabetes.copy()
print("DataFrame Shape : {}".format(df.shape))


# In[9]:


check_na_values(df)


# In[10]:


df.head()


# In[11]:


desc = df.describe().T
desc[desc["min"] == 0]


# In[12]:


df_new = pd.read_csv(path,
                     na_values= {"Glucose" : 0 , "BloodPressure" : 0, "SkinThickness" : 0 , 
                                 "Insulin" : 0 , "BMI" : 0})

df = df_new.copy()


# In[13]:


df.isnull().sum()


# ## Analysis

# In[14]:


check_df(df)


# In[15]:


cat_cols, num_cols , cat_but_car = grab_col_names(df,details=True)


# In[16]:


print("Cat Cols :\n", cat_cols,end = "\n\n")
print("Num Cols :\n", num_cols,end = "\n\n")
print("Cat But Car :\n ", cat_but_car)


# In[17]:


for cat in cat_cols:
    cat_summary(df, cat ,plot = True)


# In[18]:


for num in num_cols:
    num_summary(df , num ,plot = True)


# In[19]:


for num in num_cols:
    target_summary_with_num(df, "Outcome", num)


# In[20]:


for cat in cat_cols:
    target_summary_with_cat(df , "Outcome" , cat)


# In[21]:


sns.pairplot(df)
plt.show()


# ## NA_Values

# In[22]:


na_values = missing_values_table(df , na_name=True)


# In[23]:


missing_vs_target(df ,"Outcome", na_values)


# In[24]:


median = df.groupby("Outcome").median().iloc[:,1:-2]
median


# In[25]:


df[["Outcome"]].value_counts()


# In[26]:


na_values


# In[27]:


def distplot_na_values(dataframe,na_values):
    
    i = 1 
    num = len(na_values)
    size = 15
    plt.figure(figsize = (10,15))
    
    for na_value in na_values:
        
        plt.subplot(num,1,i)
        sns.distplot(dataframe[na_value])
        
        plt.title(na_value , fontsize = size)
        plt.xlabel(na_value, fontsize = size)
        plt.ylabel("Density" , fontsize = size)
        i+=1
        plt.tight_layout()
        
        
        
    plt.show()


# In[28]:


distplot_na_values(df , na_values)


# In[29]:


for na_col in na_values:
    df.loc[ ((df["Outcome"] == 0) & (df[na_col].isnull())), na_col] = df.groupby("Outcome")[na_col].median()[0]
    df.loc[ ((df["Outcome"] == 1) & (df[na_col].isnull())), na_col] = df.groupby("Outcome")[na_col].median()[1]


# In[30]:


check_na_values(df)


# In[31]:


distplot_na_values(df , na_values)


# In[32]:


def boxplot(dataframe, num_cols):
    
    i=1
    num_ = (len(num_cols))
    plt.figure(figsize = (20,20))
    size =15
    
    for num in num_cols:
        
        plt.subplot(num_,1,i)
        sns.boxplot(dataframe[num])
        plt.xlabel(f"{num}", fontsize = size)
        plt.title(f"{num}" , fontsize = size)
        i+=1
        plt.tight_layout()
    plt.show()


# In[33]:


boxplot(df , num_cols)


# In[34]:


for col in num_cols:
    print("For {} Outliers : {}".format(col.upper(),check_outliers(df , col)))


# In[35]:


for col in num_cols:
    replace_with_thresholds(df ,col)


# In[36]:


boxplot(df,num_cols)


# In[37]:


for col in num_cols:
    print("For {} Outliers : {}".format(col.upper(),check_outliers(df , col)))


# In[38]:


check_df(df)


# ## Feature Engineering

# In[39]:


df[["BMI"]].describe().T


# In[40]:


bins = [0 ,18.4 ,24.9 ,29.9 ,100]
labels = ["Thin", "Normal", "OverWeight", "Obese"]
df["NEW_BMI"] = pd.cut(df["BMI"], bins = bins , labels = labels)

df[["NEW_BMI"]].describe().T


# In[41]:


# Converting BMI Values Categorically


def analyses(dataframe ,col ):
    
    print("".center(50,"#"), end = "\n\n")
    
    data = dataframe.groupby(col).agg({"Outcome" : ["mean","count"]})
    data.columns = ["Outcome_Mean" , "Outcome_Count"]
    print(data , end = "\n\n")
    
    print("".center(50,"#") , end = "\n\n")
    
    print(pd.DataFrame({"Ratio" : dataframe[col].value_counts() / len(dataframe) ,
                        "Value_Counts" : dataframe[col].value_counts()}) , end = "\n\n")
    
    print("".center(50,"#"), end = "\n\n")
    
    dataframe[col].value_counts().plot.pie( autopct = "%1.0f%%", shadow = True , figsize = (7,7))


# In[42]:


analyses(df , "NEW_BMI")


# In[43]:


from scipy.stats import shapiro, kruskal, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest

def shapiro_(df, cols_name , target):
    
    for col in cols_name :
    
        p_value = shapiro(df.loc[ df[target] == col , "Outcome"])[1]
        print(f"For {col} PValues : {round(p_value,4)}")
    
        if p_value > 0.05:
        
            print(f"For {col.upper()} , H0 : is the Normal Distribution...", end = "\n\n")
            
        else:
        
            print(f"For {col.upper()} , H1: Not Normal Distribution..." , end = "\n\n")


# In[44]:


new_bmi = [col for col in df["NEW_BMI"].unique() if col not in "Thin"]

shapiro_(df , new_bmi, "NEW_BMI")


# In[45]:


p_value = kruskal(df.loc[df["NEW_BMI"] == "Normal", "Outcome"],
                  df.loc[df["NEW_BMI"] == "OverWeight", "Outcome"],
                  df.loc[df["NEW_BMI"] == "Obese", "Outcome"])[1]

p_value = round(p_value, 4)

one = "H0: M1 = M2 (There is no statistically significant difference between the two group means.)"
    
two = "H1: M1 != M2 (There is a statistically significant difference between the means of the two groups.)"

if p_value > 0.05:
    
    print(f"For P_Value : {p_value}\n\n{one}")
    
    
else:
    
    print(f"For P_Value : {p_value}\n\n{two}")


# ## 

# In[46]:


# Converting Glucose Values Categorically

df[["Glucose"]].describe().T


# In[47]:


bins = [0 , 140, 200, 300]
labels = ["NotDiabetes" , "PeriDiabetes", "Diabetes"]

df["NEW_GLUCOSE"] = pd.cut(df["Glucose"] , bins = bins , labels = labels)

df[["NEW_GLUCOSE"]].describe().T


# In[48]:


analyses(df , "NEW_GLUCOSE")


# In[49]:


new_glucose = [col for col in df["NEW_GLUCOSE"].unique() if col not in "Diabetes"]

shapiro_(df , new_glucose , "NEW_GLUCOSE") 


# In[50]:


p_value = mannwhitneyu(df.loc[ df["NEW_GLUCOSE"] == "PeriDiabetes", "Outcome"],
                       df.loc[ df["NEW_GLUCOSE"] == "NotDiabetes", "Outcome"])[1]

p_value = round(p_value, 4)

one = "H0: M1 = M2 (There is no statistically significant difference between the two group means.)"
    
two = "H1: M1 != M2 (There is a statistically significant difference between the means of the two groups.)"

if p_value > 0.05:
    
    print(f"For P_Value : {p_value}\n\n{one}")
    
else:
    
    print(f"For P_Value : {p_value}\n\n{two}")


# ## 

# In[51]:


# Converting Insulin Values Categorically

df[["Insulin"]].describe().T


# In[52]:


def Insulin(dataframe):
    
    insulin = dataframe["Insulin"]
    
    if 16 <= insulin <=166:
        
        return "Normal"
    
    else:
        
        return "AbNormal"


# In[53]:


df["NEW_INSULIN"] = df.apply(Insulin, axis = 1)

df[["NEW_INSULIN"]].describe().T


# In[54]:


analyses(df , "NEW_INSULIN")


# In[55]:


new_insulin = [col for col in df["NEW_INSULIN"].unique()]

shapiro_(df, new_insulin, "NEW_INSULIN")


# In[56]:


p_value = mannwhitneyu(df.loc[ df["NEW_INSULIN"] == "Normal", "Outcome"],
                       df.loc[ df["NEW_INSULIN"] == "AbNormal", "Outcome"])[1]

p_value = round(p_value, 4)

one = "H0: M1 = M2 (There is no statistically significant difference between the two group means.)"
    
two = "H1: M1 != M2 (There is a statistically significant difference between the means of the two groups.)"

if p_value > 0.05:
    
    print(f"For P_Value : {p_value}\n\n{one}")
    
else:
    
    print(f"For P_Value : {p_value}\n\n{two}")


# ## 

# In[57]:


# Converting age variable to categorical variable

df[["Age"]].describe().T


# In[58]:


df.loc[ (df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[ df["Age"] >= 50, "NEW_AGE_CAT"] = "senior"


# In[59]:


analyses(df, "NEW_AGE_CAT")


# In[60]:


new_age_cat = [col for col in df["NEW_AGE_CAT"].unique()]

shapiro_(df, new_age_cat , "NEW_AGE_CAT")


# In[61]:


p_value = mannwhitneyu(df.loc[ df["NEW_AGE_CAT"] == "senior", "Outcome"],
                       df.loc[ df["NEW_AGE_CAT"] == "mature", "Outcome"])[1]

p_value = round(p_value, 4)

one = "H0: M1 = M2 (There is no statistically significant difference between the two group means.)"
    
two = "H1: M1 != M2 (There is a statistically significant difference between the means of the two groups.)"

if p_value > 0.05:
    
    print(f"For P_Value : {p_value}\n\n{one}")
    
else:
    
    print(f"For P_Value : {p_value}\n\n{two}")


# ## 

# In[62]:


##
df.loc[ (df["BMI"] < 18.4) & ((df["Age"] >=21) & (df["Age"] < 50)) , "NEW_BMI_AGE"] = "ThinMature"
df.loc[ (df["BMI"] < 18.4) & (df["Age"] >= 50), "NEW_BMI_AGE"] = "ThinSenior"


##
df.loc[ ((df["BMI"] >=18.4) & (df["BMI"] < 24.9)) & ((df["Age"] >=21) & (df["Age"] < 50)),       "NEW_BMI_AGE"] = "NormalMature"

df.loc[ ((df["BMI"] >= 18.4) & (df["BMI"] < 24.9)) & (df["Age"] >= 50), "NEW_BMI_AGE"] = "NormalSenior"

##
df.loc[ ((df["BMI"] >=24.9) & (df["BMI"] < 29.9)) & ((df["Age"] >=21) & (df["Age"] < 50)),       "NEW_BMI_AGE"] = "OverWeightMature"

df.loc[ ((df["BMI"] >= 24.9) & (df["BMI"] < 29.9)) & (df["Age"] >= 50), "NEW_BMI_AGE"] = "OverWeightSenior"

##
df.loc[ ((df["BMI"] >=29.9) & (df["BMI"] < 100)) & ((df["Age"] >=21) & (df["Age"] < 50)),       "NEW_BMI_AGE"] = "ObeseMature"

df.loc[ ((df["BMI"] >= 29.9) & (df["BMI"] < 100)) & (df["Age"] >= 50), "NEW_BMI_AGE"] = "ObeseSenior"


# In[63]:


analyses(df, "NEW_BMI_AGE")


# In[64]:


new_bmi_age = [col for col in df["NEW_BMI_AGE"].unique() if col not in "ThinMature"]

shapiro_(df ,new_bmi_age, "NEW_BMI_AGE")


# In[65]:


pvalue = kruskal(df.loc[df["NEW_BMI_AGE"] == "ObeseSenior" , "Outcome"],
                 df.loc[df["NEW_BMI_AGE"] == "OverWeightmature" , "Outcome"],
                 df.loc[df["NEW_BMI_AGE"] == "NormalMature" , "Outcome"],
                 df.loc[df["NEW_BMI_AGE"] == "ObeseMature" , "Outcome"],
                 df.loc[df["NEW_BMI_AGE"] == "OverWeightSenior" , "Outcome"],
                 df.loc[df["NEW_BMI_AGE"] == "NormalSenior" , "Outcome"])[1]

p_value = round(p_value, 4)

one = "H0: M1 = M2 (There is no statistically significant difference between the two group means.)"
    
two = "H1: M1 != M2 (There is a statistically significant difference between the means of the two groups.)"

if p_value > 0.05:
    
    print(f"For P_Value : {p_value}\n\n{one}")
    
else:
    
    print(f"For P_Value : {p_value}\n\n{two}")


# ## 

# In[66]:


###
df.loc[(df["Glucose"] < 140) & ((df["Age"] >=21) & (df["Age"] <50)) ,"NEW_AGE_GLUCOSE"] = "NotDiabetesMature"
df.loc[(df["Glucose"] < 140) & (df["Age"] >=50) , "NEW_AGE_GLUCOSE"] = "NotDiabetesSenior"

####
df.loc[((df["Glucose"] >= 140) & (df["Glucose"] < 200)) & ((df["Age"] >=21) & (df["Age"] <50)),        "NEW_AGE_GLUCOSE"] = "PeriDiabetesMature"

df.loc[((df["Glucose"] >= 140) & (df["Glucose"] < 200)) & (df["Age"] >= 50) ,        "NEW_AGE_GLUCOSE"] = "PeriDiabetesSenior"

####
df.loc[((df["Glucose"] >= 200) & (df["Glucose"] < 300)) & ((df["Age"] >=21) & (df["Age"] <50))        , "NEW_AGE_GLUCOSE"] = "DiabetesMature"

df.loc[((df["Glucose"] >=200) & (df["Glucose"] < 300)) & (df["Age"] >= 50),       "NEW_AGE_GLUCOSE"] = "DiabetesSenior"


# In[67]:


df["NEW_AGE_GLUCOSE"].unique()


# In[68]:


analyses(df, "NEW_AGE_GLUCOSE")


# In[69]:


new_age_glucose = [col for col in df["NEW_AGE_GLUCOSE"].unique()]

shapiro_(df,new_age_glucose, "NEW_AGE_GLUCOSE")


# In[70]:


p_value = kruskal(df.loc[df["NEW_AGE_GLUCOSE"] == "PeriDiabetesSenior", "Outcome" ],
                  df.loc[df["NEW_AGE_GLUCOSE"] == "NotDiabetesMature", "Outcome" ],
                  df.loc[df["NEW_AGE_GLUCOSE"] == "PeriDiabetesMature", "Outcome" ],
                  df.loc[df["NEW_AGE_GLUCOSE"] == "NotDiabetesSenior", "Outcome" ])[1]

p_value = round(p_value,4)

p_value = round(p_value, 4)

one = "H0: M1 = M2 (There is no statistically significant difference between the two group means.)"
    
two = "H1: M1 != M2 (There is a statistically significant difference between the means of the two groups.)"

if p_value > 0.05:
    
    print(f"For P_Value : {p_value}\n\n{one}")
    
else:
    
    print(f"For P_Value : {p_value}\n\n{two}")


# In[71]:


df.head()


# ## Label_Encoding and One-Hot Enconding İşlemleri

# In[72]:


def nunique_dtypes(dataframe):
    
    data = pd.DataFrame()
    data["Name"] = [col for col in dataframe.columns ]
    data["Nunique"] = [dataframe[col].nunique() for col in dataframe.columns]
    data["Dtypes"] = [dataframe[col].dtype for col in dataframe.columns]
    return data

nunique_dtypes(df)


# In[73]:


cat_cols , num_cols, cat_but_car = grab_col_names(df, details=True)


# In[74]:


print("Cat Cols :\n", cat_cols,end = "\n\n")
print("Num Cols :\n", num_cols,end = "\n\n")
print("Cat But Car :\n ", cat_but_car)


# In[75]:


new_cat_cols = [col for col in cat_cols if col not in "Outcome"]
new_cat_cols


# In[76]:


for col in new_cat_cols:
    label_encoder(df , col)


# In[77]:


nunique_dtypes(df)


# In[78]:


df = one_hot_encoder(df , new_cat_cols,drop_first=True)
df.head()


# In[79]:


cat_cols , num_cols, cat_but_car = grab_col_names(df,details=True)


# In[80]:


print("Cat Cols :\n", cat_cols,end = "\n\n")
print("Num Cols :\n", num_cols,end = "\n\n")
print("Cat But Car :\n ", cat_but_car)


# In[81]:


for col in num_cols:
    print(f"For {col.upper()} Outliers : {check_outliers(df ,col)}",end = "\n\n")


# In[82]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


# In[83]:


X = df.drop("Outcome" , axis = 1)
y = df["Outcome"]


# ## Model 
# 
# ## Success Evaluation (Validation) with Holdout Method

# In[84]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.metrics import r2_score,f1_score,precision_score,recall_score,accuracy_score,                            confusion_matrix,classification_report,roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split


# In[85]:


def score_test(model , X , y , roc_plot = True , matrix = True):
    
    X_train, X_test, y_train, y_test= train_test_split(X ,y, test_size=0.2,random_state = 42)
    
    model.fit(X_train, y_train)
    
    y_pred_test= model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:,1]
    
    print("".center(50,"#"))
    print(f"{type(model).__name__}".center(50," "))
    print("".center(50,"#"), end = "\n\n")
    print("classification report".upper().center(50," "),end = "\n\n")
    print(classification_report(y_test, y_pred_test))
    
    print(" test scores ".upper().center(50,"#"),end = "\n\n")
    print(" accuracy score :".upper(), accuracy_score(y_test,y_pred_test),end = "\n\n")
    print(" precison score :".upper() , precision_score(y_test,y_pred_test),end = "\n\n")
    print(" recall score :".upper(), recall_score(y_test,y_pred_test),end = "\n\n")
    print(" f1 score :".upper(), f1_score(y_test,y_pred_test),end = "\n\n")
    print(" r2 score :".upper(), r2_score(y_test,y_pred_test),end = "\n\n")
    print(" roc auc score :".upper() , roc_auc_score(y_test , y_proba_test),end = "\n\n")
    print("".center(50,"#"))
    
    if roc_plot:
        
        roc_score =roc_auc_score(y_test , y_proba_test)
        
        fpr, tpr, threshols = roc_curve(y_test, y_proba_test)
        
        plt.figure(figsize = (10,7))
        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1], "--r")
        plt.xlim([-0.05,1.0])
        plt.ylim([0.0,1.1])
        size = 15
        plt.title("AUC (Area : %.3f)" % roc_score , fontsize = size)
        plt.xlabel("False Positive Rate" , fontsize = size)
        plt.ylabel("True Positive Rate" , fontsize = size)
        plt.show()
        
    if matrix :
        
        fig , axes = plt.subplots(figsize = (7,5))
        
        cm = confusion_matrix(y_test ,y_pred_test)
        ax = sns.heatmap(cm, annot=True , annot_kws={"size" : 23} , fmt = ".3g" , ax = axes,
                         cmap = "rainbow", linewidths=3, linecolor="white",cbar = False, center = 0)
        
        plt.xlabel(" Predict Label ", fontsize = size)
        plt.ylabel(" True Label ", fontsize = size)
        plt.title(f" Confusion Matrix For {type(model).__name__.upper()}", fontsize = size)
        plt.show()


# In[86]:


score_test(LogisticRegression() , X, y)


# ## Success Evaluation with CV and Hyperparameter Optimization with GridSearchCV

# In[87]:


logistic_param_grid ={'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                 'C' : np.logspace(-4, 4, 5),
                 'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
                 'max_iter' : [100, 1000, 2500, 5000]
                 }

models = [("LR",LogisticRegression(),logistic_param_grid)]


# In[88]:


from sklearn.model_selection import cross_validate

def base_model(models , X, y, cv = 5):
    
    data = pd.DataFrame()
    index = 0
    
    for name, model, params in models:
        
            results = cross_validate(estimator=model,
                                     X= X ,
                                     y= y,
                                     cv= cv,
                                     scoring = ["accuracy","roc_auc"],
                                     n_jobs=-1,
                                     verbose=0)
    
            data.loc[index,"NAME"] = name
            data.loc[index,"ROC_AUC"] = results["test_roc_auc"].mean()
            data.loc[index,"ACCURACY"] = results["test_accuracy"].mean()
            data.loc[index,"FIT_TIME"] = results["fit_time"].mean()
            data.loc[index,"SCORE_TIME"] = results["score_time"].mean()
            
            index+=1
    
    data = data.set_index("NAME")
    return data


# In[89]:


base_model(models,X, y, cv = 5)


# In[90]:


def hyperparamters_optimizer(models,X,y,cv = 5):
    
    models_dict = {}
    data = pd.DataFrame()
    index = 0
    
    for name, model, params in models:
        
        cv_results = cross_validate(estimator=model,
                                    X = X,
                                    y = y,
                                    cv = cv,
                                    scoring=["accuracy","roc_auc"],
                                    n_jobs=-1,
                                    verbose = 0)
        
        before_roc_auc = cv_results["test_roc_auc"].mean()
        before_accuracy = cv_results["test_accuracy"].mean()
        
        print("".center(50,"#"),end = "\n\n")
        print(f"NAME : {name.upper()}",end= "\n\n")
        print(f"Before GridSearch\n\nRoc Auc : {before_roc_auc}\nAccuracy : {before_accuracy}",end="\n\n")
        #print("".center(50,"#"),end = "\n\n") 
        
        
        best_grid = GridSearchCV(estimator=model,
                                 param_grid=params,
                                 cv = cv , 
                                 n_jobs=-1,
                                 verbose=0,
                                 scoring="roc_auc").fit(X,y)
        
        print(f"Best Params : {best_grid.best_params_}")
        
        final_model = model.set_params(**best_grid.best_params_)
        
        final_cv_results = cross_validate(estimator=final_model,
                                          X = X,
                                          y = y,
                                          cv = cv,
                                          scoring=["accuracy","roc_auc"],
                                          n_jobs=-1,
                                          verbose=0)
        
        final_roc_auc = final_cv_results["test_roc_auc"].mean()
        final_accuracy = final_cv_results["test_accuracy"].mean()
        
        print(f"\nAfter GridSearch\n\nRoc Auc : {final_roc_auc}\nAccuracy : {final_accuracy}",end="\n\n")
        print("".center(50,"#"),end="\n\n")
        
        data.loc[index , "NAME"] = name
        data.loc[index , "Before ROC_AUC"] = before_roc_auc
        data.loc[index , "After ROC_AUC"] = final_roc_auc
        data.loc[index, "Before ACCURACY"] = before_accuracy
        data.loc[index, "After ACCURACY"] = final_accuracy
        index+=1
        
        models_dict[name] = final_model
    
    data = data.set_index("NAME")
    
    return data , models_dict
        


# In[91]:


data , models_dict = hyperparamters_optimizer(models,X,y, cv = 10)


# In[92]:


data


# In[93]:


models_dict["LR"].get_params()


# In[94]:


def roc_auc_graph(model, X ,y):
        
        model.fit(X,y)
        
        y_proba = model.predict_proba(X)[:,1]
        roc_score =roc_auc_score(y , y_proba)
        
        fpr, tpr, threshols = roc_curve(y, y_proba)
        
        plt.figure(figsize = (10,7))
        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1], "--r")
        plt.xlim([-0.05,1.0])
        plt.ylim([0.0,1.1])
        size = 15
        plt.title("AUC (Area : %.3f)" % roc_score , fontsize = size)
        plt.xlabel("False Positive Rate" , fontsize = size)
        plt.ylabel("True Positive Rate" , fontsize = size)
        plt.show()
        
def confusion_matrix_graph(model, X ,y):
        
        y_pred = model.predict(X)
        
        fig , axes = plt.subplots(figsize = (7,5))
        
        size = 15
        cm = confusion_matrix(y ,y_pred)
        ax = sns.heatmap(cm, annot=True , annot_kws={"size" : 23} , fmt = ".3g" , ax = axes,
                         cmap = "rainbow", linewidths=3, linecolor="white",cbar = False, center = 0)
        
        plt.xlabel(" Predict Label ", fontsize = size)
        plt.ylabel(" True Label ", fontsize = size)
        plt.title(f" Confusion Matrix For {type(model).__name__.upper()}", fontsize = size)
        plt.show()


# In[95]:


roc_auc_graph(models_dict["LR"], X ,y)


# In[96]:


confusion_matrix_graph(models_dict["LR"], X ,y)


# In[97]:


# save final_model
import pickle

for name,model,params in models:
    
    pd.to_pickle(models_dict[name], open(name+"_diabetes.pkl","wb"))


# In[98]:


pd.to_pickle(df, open("diabetes.pkl","wb"))


# In[ ]:




