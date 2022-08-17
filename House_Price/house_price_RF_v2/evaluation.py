

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
all_feature_indices = [71,5,72,9,14,39,75,69,64,45,74,73,35,48,29,6,78,55,36,63,40,28,16,68,42,31,25,50,10,1,33,32,21,24,7,13,8,77,65,22,52,15,79,76,47,58,23,60,11,18,2,41,70,4,56,44,54,51,30,67,3,12,0,34,38,59,43,57,37,66,26,20,62,53,49,19,46,27,17,61]
all_feature_indices = [72,9,71,5,74,69,39,75,45,64,42,13,6,73,14,55,29,36,35,31,65,16,40,10,28,25,68,50,78,7,52,63,79,33,15,1,48,8,70,32,77,24,41,60,21,22,23,2,51,11,18,58,47,0,76,67,44,4,30,3,12,59,66,54,57,34,37,38,56,20,43,26,62,53,19,49,27,46,17,61]
df_X = df_train_data.iloc[:,all_feature_indices].copy()



# **************************************** encode to number ****************************************
for i in df_X:
    #df_X[i].replace(np.NaN, 0, inplace=True) #replace NaN with 0
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
percent_test = 0.3
max_ncol = len(df_train_data.columns)
max_nrow = df_train_data.__len__() + 1
mid_nrow = round(max_nrow*(1-percent_test))

df_X_train = df_X[:mid_nrow].copy()
df_X_test = df_X[mid_nrow:].copy()

fl_y_train = df_train_data.iloc[:mid_nrow, max_ncol - 1:max_ncol]
fl_y_test = df_train_data.iloc[mid_nrow:, max_ncol - 1:max_ncol]



#######################################################################
#            EVALUATION
#######################################################################



# **************************************** metric ****************************************
fl_true_y = fl_y_test.values.ravel()
RMSE = lambda x,y : np.sqrt(np.mean((x-y)**2)) / np.mean(y)     #RMSE



# **************************************** error & tree depth ****************************************
parameters = []
errors = []
min_para = 2
max_para = 79
n_feature_index = len(all_feature_indices)
df_X_train_org = df_X_train.copy()
df_X_test_org = df_X_test.copy()

for i in range(min_para, max_para,2):
    pick_feature_indices = all_feature_indices[- i : n_feature_index]
    df_X_train = df_X_train_org.iloc[:, pick_feature_indices].copy()
    df_X_test = df_X_test_org.iloc[:, pick_feature_indices].copy()

    #para
    RF_regression_model = RandomForestRegressor(max_depth = 16,  # bigger, more precise
                                                random_state=0,
                                                n_estimators = 160,  # bigger, more precise
                                                # min_samples_leaf = i, # bigger, less noise
                                                n_jobs = -1
                                                )
    # fitting
    RF_regression_model.fit(X=df_X_train, y=fl_y_train.values.ravel())
    fl_predict_y = RF_regression_model.predict(df_X_test)
    error = RMSE(fl_predict_y, fl_true_y)
    parameters.append(i)
    errors.append(error)


# plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(parameters, errors)
for xy in zip(parameters, errors):
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
plt.grid()
plt.show()

# conclusion:
# depth = 10
