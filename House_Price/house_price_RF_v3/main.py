
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
data clean:
    http://fmwww.bc.edu/repec/bocode/t/transint.html


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
pick_feature_indices = ['KitchenAbvGr', 'BsmtFinType2', 'LotConfig', 'Electrical', 'FullBath',
                       'HouseStyle', 'Heating', 'BsmtExposure', 'BsmtFinSF2', 'MSZoning',
                       'CentralAir', 'OverallCond', 'MasVnrType', 'Fireplaces', 'BsmtFullBath',
                       'TotRmsAbvGrd', '2ndFlrSF', 'Functional', 'OverallQual', 'BldgType']
df_X = df_train_data[pick_feature_indices].copy()


# **************************************** encode to number ****************************************
for i in df_X:
    if df_X[i].dtypes != np.float64:
        df_X[i] = df_X[i].astype(str)  # conver to string
        encoder = LabelEncoder()
        encoder.fit(df_X[i])
        df_X[i] = encoder.transform(df_X[i])
    df_X[i].replace(np.NaN, np.mean(df_X[i]), inplace=True)  # replace NaN with mean


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


# **************************************** reduce skewness ****************************************
# by log
fl_y_train_log = np.log(fl_y_train)



####################################
#             PREDICT
####################################



# **************************************** modelling ****************************************

RF_regression_model = RandomForestRegressor(max_depth=16,  # bigger, more precise
                                            random_state=0,
                                            n_estimators=160,  # bigger, more precise
                                            n_jobs=-1
                                            )
RF_regression_model.fit(X = df_X_train,
                        y = fl_y_train_log.values.ravel()
                        )


# **************************************** test ****************************************
df_test_data = pd.read_csv('C:\\Users\\fuxt2\\Documents\\code\\python\\house_price\\data\\test.csv')


# **************************************** pick features ****************************************
pick_feature_indices = ['KitchenAbvGr', 'BsmtFinType2', 'LotConfig', 'Electrical', 'FullBath',
                       'HouseStyle', 'Heating', 'BsmtExposure', 'BsmtFinSF2', 'MSZoning',
                       'CentralAir', 'OverallCond', 'MasVnrType', 'Fireplaces', 'BsmtFullBath',
                       'TotRmsAbvGrd', '2ndFlrSF', 'Functional', 'OverallQual', 'BldgType']
df_X = df_test_data[pick_feature_indices].copy()

# **************************************** encode to number ****************************************
for i in df_X:
    if df_X[i].dtypes != np.float64:
        df_X[i] = df_X[i].astype(str)  # conver to string
        encoder = LabelEncoder()
        encoder.fit(df_X[i])
        df_X[i] = encoder.transform(df_X[i])
    df_X[i].replace(np.NaN, np.mean(df_X[i]), inplace=True)  # replace NaN with mean


# **************************************** standardizing ****************************************
f_min_max = lambda x: (x-np.min(x)) / (np.max(x) - np.min(x))
df_X = df_X.apply(f_min_max)


# **************************************** predict ****************************************
fl_predict_y = RF_regression_model.predict(df_X)


# **************************************** output ****************************************
df_output = df_test_data[['Id']].copy()
df_output["SalePrice"] = fl_predict_y
df_output.to_csv("my_predict.csv", index=False, encoding='utf-8')




