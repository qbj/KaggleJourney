

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sb



########################################################################
#             DATA
########################################################################



# **************************** read test data ****************************************
df_train_data = pd.read_csv('C:\\Users\\fuxt2\\Documents\\code\\python\\house_price\\data\\train.csv')


# **************************************** pick features ****************************************
all_feature_indices = [31,33,67,35,37,38,39,40,42,43,45,46,48,50,51,53,57,58,60,61,62,63,64,65,66,30,29,34,27,1,2,3,4,5,6,7,8,9,10,11,12,28,14,13,68,16,23,26,22,19,21,20,52,59,49,36,32,24,18,41,25,55,44,47,56,54,17,15]
df_X = df_train_data.iloc[:,all_feature_indices].copy()



# **************************************** encode to number ****************************************
drop_null_list = []
for i in df_X:
    if np.sum(df_X[i].isnull()) / len(df_X[i]) > 0.20:    # NULL% > 5%
        drop_null_list.append(i)
    if df_X[i].dtypes != np.float64:
        df_X[i] = df_X[i].astype(str)  # conver to string
        encoder = LabelEncoder()
        encoder.fit(df_X[i])
        df_X[i] = encoder.transform(df_X[i])
    df_X[i].replace(np.NaN, np.mean(df_X[i]), inplace=True)  # replace NaN with mean

# remove null
for i in drop_null_list:
    df_X = df_X.drop(i, axis = 1)


# **************************************** standardizing ****************************************
f_min_max = lambda x: (x-np.min(x)) / (np.max(x) - np.min(x))
df_X = df_X.apply(f_min_max)


# **************************************** slice train & test data ****************************************
percent_test = 0.3
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

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# sb.distplot(fl_y_train, bins=50)
# plt.title('Original Data')
# plt.xlabel('Sale Price')
#
# plt.subplot(1, 2, 2)
# sb.distplot(fl_y_train_log, bins=50)
# plt.title('Natural Log of Data')
# plt.xlabel('Natural Log of Sale Price')

# plt.tight_layout()
# plt.show()



#######################################################################
#            EVALUATION
#######################################################################



# **************************************** metric ****************************************
fl_true_y = fl_y_test.values.ravel()
RMSE = lambda x,y : np.sqrt(np.mean((x-y)**2)) / np.mean(y)     #RMSE



# **************************************** error & tree depth ****************************************
parameters = []
errors = []
max_ncol = len(df_X.columns)
min_para = 1
max_para = max_ncol
df_X_train_org = df_X_train.copy()
df_X_test_org = df_X_test.copy()


for i in range(40, 41):
    df_X_train = df_X_train_org.iloc[:, max_ncol-i:max_ncol].copy()
    df_X_test = df_X_test_org.iloc[:, max_ncol-i:max_ncol].copy()

    #para
    RF_regression_model = RandomForestRegressor(max_depth = 16,  # bigger, more precise
                                                random_state=0,
                                                n_estimators = 160,  # bigger, more precise
                                                min_samples_leaf = 100, # bigger, less noise
                                                n_jobs = -1
                                                )
    # fitting
    RF_regression_model.fit(X=df_X_train, y=fl_y_train_log.values.ravel())
    fl_predict_y = RF_regression_model.predict(df_X_test)
    error = RMSE(fl_predict_y, fl_true_y)
    parameters.append(i)
    errors.append(error)

print(errors)
# # plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(parameters, errors)
# for xy in zip(parameters, errors):
#     ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
# plt.grid()
# plt.show()



####################################
#             PREDICT
####################################



# **************************************** test ****************************************
df_test_data = pd.read_csv('C:\\Users\\fuxt2\\Documents\\code\\python\\house_price\\data\\test.csv')


# **************************************** pick features ****************************************
pick_feature_indices = df_X_train.columns
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




