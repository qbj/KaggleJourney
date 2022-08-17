import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sb


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
drop_null_list = []
for i in df_X_train:
    if np.sum(df_X_train[i].isnull()) / len(df_X_train[i]) > 0.05:    # NULL% > 5%
        drop_null_list.append(i)
    df_X_train[i].replace(np.NaN, 0, inplace=True)
    if df_X_train[i].dtypes != np.float64:
        df_X_train[i] = df_X_train[i].astype(str)  # conver to string
        encoder = LabelEncoder()
        encoder.fit(df_X_train[i])
        df_X_train[i] = encoder.transform(df_X_train[i])

#remove null
for i in drop_null_list:
    df_X_train = df_X_train.drop(i, axis = 1)



# **************************************** standardizing ****************************************
f_min_max = lambda x: (x-np.min(x)) / (np.max(x) - np.min(x))
df_X_train = df_X_train.apply(f_min_max)


# **************************************** reduce skewness ****************************************
# by log
fl_y_train_log = np.log(fl_y_train)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sb.distplot(fl_y_train, bins=50)
plt.title('Original Data')
plt.xlabel('Sale Price')

plt.subplot(1, 2, 2)
sb.distplot(fl_y_train_log, bins=50)
plt.title('Natural Log of Data')
plt.xlabel('Natural Log of Sale Price')

plt.tight_layout()
# plt.show()


# **************************************** modelling ****************************************
RF_regression_model = RandomForestRegressor(max_depth=16,  # bigger, more precise
                                            random_state=0,
                                            n_estimators=160,  # bigger, more precise
                                            min_samples_leaf=100, # bigger, less noise
                                            n_jobs=-1
                                            )
RF_regression_model.fit(X = df_X_train,y = fl_y_train_log.values.ravel())


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