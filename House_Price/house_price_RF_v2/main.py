
#########################################################################################
"""
date:
    2017-08-31

desc:
    for kaggle 'house price predict'. It's a regression problem.

random forest tutorial:
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
plot tutorial:
    https://bespokeblog.wordpress.com/2011/07/07/basic-data-plotting-with-matplotlib-part-2-lines-points-formatting/
how tuning:
    https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/


"""
#########################################################################################


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



########################################################################
#             DATA
########################################################################



# **************************** read test data ****************************************
df_train_data = pd.read_csv('C:\\Users\\fuxt2\\Documents\\code\\python\\house_price\\data\\train.csv')


# **************************************** pick features ****************************************
all_feature_indices = [72,9,71,5,74,69,39,75,45,64,42,13,6,73,14,55,29,36,35,31,65,16,40,10,28,25,68,50,78,7,52,63,79,33,15,1,48,8,70,32,77,24,41,60,21,22,23,2,51,11,18,58,47,0,76,67,44,4,30,3,12,59,66,54,57,34,37,38,56,20,43,26,62,53,19,49,27,46,17,61]
df_X = df_train_data.iloc[:,all_feature_indices].copy()


# **************************************** encode to number ****************************************
for i in df_X:
    df_X[i].replace(np.NaN, 0, inplace=True)
    if df_X[i].dtypes != np.float64:
        df_X[i] = df_X[i].astype(str)  # conver to string
        encoder = LabelEncoder()
        encoder.fit(df_X[i])
        df_X[i] = encoder.transform(df_X[i])


# **************************************** standardizing ****************************************
f_min_max = lambda x: (x-np.min(x)) / (np.max(x) - np.min(x))
df_X = df_X.apply(f_min_max)


# **************************************** slice train & test data ****************************************
percent_test = 0
max_ncol = len(df_train_data.columns)
max_nrow = df_train_data.__len__() + 1
mid_nrow = round(max_nrow*(1-percent_test))

df_X_train = df_X[:mid_nrow].copy()
df_X_test = df_X[mid_nrow:].copy()

fl_y_train = df_train_data.iloc[:mid_nrow, max_ncol - 1:max_ncol]
fl_y_test = df_train_data.iloc[mid_nrow:, max_ncol - 1:max_ncol]


####################################
#             PREDICT
####################################



# **************************************** modelling ****************************************
n_features = 55
n_feature_index = len(all_feature_indices)
pick_feature_indices = all_feature_indices[- n_features: n_feature_index]
df_X_train = df_X_train.iloc[:, pick_feature_indices].copy()

RF_regression_model = RandomForestRegressor(max_depth=16,  # bigger, more precise
                                            random_state=0,
                                            n_estimators=160,  # bigger, more precise
                                            n_jobs=-1
                                            )
RF_regression_model.fit(X = df_X_train,y = fl_y_train.values.ravel())


# **************************************** test ****************************************
df_test_data = pd.read_csv('C:\\Users\\fuxt2\\Documents\\code\\python\\house_price\\data\\test.csv')


# **************************************** pick features ****************************************
df_X = df_test_data.iloc[:,pick_feature_indices].copy()


# **************************************** encode to number ****************************************
for i in df_X:
    df_X[i].replace(np.NaN, 0, inplace=True)
    if df_X[i].dtypes != np.float64:
        df_X[i] = df_X[i].astype(str)  # conver to string
        encoder = LabelEncoder()
        encoder.fit(df_X[i])
        df_X[i] = encoder.transform(df_X[i])


# **************************************** standardizing ****************************************
f_min_max = lambda x: (x-np.min(x)) / (np.max(x) - np.min(x))
df_X = df_X.apply(f_min_max)


# **************************************** predict ****************************************
fl_predict_y = RF_regression_model.predict(df_X)


# **************************************** output ****************************************
df_output = df_test_data[['Id']].copy()
df_output["SalePrice"] = fl_predict_y
df_output.to_csv("my_predict.csv", index=False, encoding='utf-8')




