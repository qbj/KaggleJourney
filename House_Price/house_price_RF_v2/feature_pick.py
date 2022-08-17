import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder



# **************************** read test data ****************************************
df_train_data = pd.read_csv('C:\\Users\\fuxt2\\Documents\\code\\python\\house_price\\data\\train.csv')


# **************************************** train & test ****************************************
max_ncol = len(df_train_data.columns)
max_nrow = df_train_data.__len__() + 1
percent_test = 0.3
mid_nrow = round(max_nrow*(1-percent_test))

fl_y_train = df_train_data.iloc[:mid_nrow, max_ncol - 1:max_ncol]
fl_y_test = df_train_data.iloc[mid_nrow:, max_ncol - 1:max_ncol]

df_X = df_train_data.iloc[:,:-1].copy() #all features
df_X_train = df_X[:mid_nrow].copy()
df_X_test = df_X[mid_nrow:].copy()


# **************************************** encode data to numbers ****************************************
for i in df_X_train:
    df_X_train[i].replace(np.NaN, 0, inplace=True)
    if df_X_train[i].dtypes != np.float64:
        df_X_train[i] = df_X_train[i].astype(str)  # conver to string
        encoder = LabelEncoder()
        encoder.fit(df_X_train[i])
        df_X_train[i] = encoder.transform(df_X_train[i])


# **************************************** standardizing ****************************************
f_min_max = lambda x: (x-np.min(x)) / (np.max(x) - np.min(x))
df_X_train = df_X_train.apply(f_min_max)


# **************************************** modelling ****************************************
RF_regression_model = RandomForestRegressor(max_depth=16,  # bigger, more precise
                                            random_state=0,
                                            n_estimators=160,  # bigger, more precise
                                            # min_samples_leaf = i, # bigger, less noise
                                            n_jobs=-1
                                            )
RF_regression_model.fit(X = df_X_train,y = fl_y_train.values.ravel())


# **************************************** feature importance ****************************************

feature_importance = RF_regression_model.feature_importances_
top_n_features = 80
indices = np.argsort(feature_importance)[- top_n_features :]
for i in indices:
    print(feature_importance[i])
# print(indices)
df_picked_feature = df_train_data.iloc[:,indices].copy()
# print(df_picked_feature.head())

print(','.join(map(str,indices)))
# [54 46 38 57 59 41 49 19 27 17 61 80]