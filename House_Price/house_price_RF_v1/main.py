
#########################################################################################
"""
date: 2017-08-31

desc: for kaggle 'house price predict'. It's a regression problem.

random forest tutorial: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

"""
#########################################################################################


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder



####################################
#             TRAIN STAGE
####################################



# **************************** read test data ****************************************
df_train_data = pd.read_csv('C:\\Users\\fuxt2\\Documents\\code\\python\\house_price\\data\\train.csv')


# **************************************** pick features ****************************************
max_ncol = len(df_train_data.columns)
max_nrow = df_train_data.__len__() + 1
percent_test = 0.3
mid_nrow = round(max_nrow*(1-percent_test))

fl_y_train = df_train_data.iloc[:mid_nrow, max_ncol - 1:max_ncol]
fl_y_test = df_train_data.iloc[mid_nrow:, max_ncol - 1:max_ncol]
df_X = df_train_data[['MoSold', 'GarageArea', 'KitchenQual', 'LotArea', 'Neighborhood']].copy()
df_X_train = df_X[:mid_nrow].copy()
df_X_test = df_X[mid_nrow:]
#print(df_X_train)
#print(df_X_test)



# **************************************** encoding ****************************************
# df_X['OldKitchenQual'] = df_X['KitchenQual']
df_X_train['KitchenQual'] = df_X_train['KitchenQual'].astype(str)  # conver to string
encoder = LabelEncoder()
encoder.fit(df_X_train['KitchenQual'])
df_X_train['KitchenQual'] = encoder.transform(df_X_train['KitchenQual'])

# df_X['OldNeighborhood'] = df_X['Neighborhood']
df_X_train['Neighborhood'] = df_X_train['Neighborhood'].astype(str)  # convert to string
encoder = LabelEncoder()
encoder.fit(df_X_train['Neighborhood'])
df_X_train['Neighborhood'] = encoder.transform(df_X_train['Neighborhood'])

# print(df_X_train.dtypes)
# print(df_X_train.head())


# **************************************** standardizing ****************************************
f_min_max = lambda x: (x-np.min(x)) / (np.max(x) - np.min(x))
df_X_train = df_X_train.apply(f_min_max)


# **************************************** modelling ****************************************
RF_regression_model = RandomForestRegressor(max_depth=3, random_state=0)
RF_regression_model.fit(X = df_X_train,y = fl_y_train.values.ravel())


# **************************************** feature importance ****************************************
#print(RF_regression_model.feature_importances_)


# **************************************** evaluation ****************************************



####################################
#             PREDICT STAGE
####################################



# **************************************** test ****************************************
df_test_data = pd.read_csv('C:\\Users\\fuxt2\\Documents\\code\\python\\house_price\\data\\test.csv')


# **************************************** pick features ****************************************
df_X = df_test_data[['MoSold', 'GarageArea', 'KitchenQual', 'LotArea', 'Neighborhood']].copy()


# **************************************** remove null ****************************************
np.isfinite(df_X["GarageArea"]).all()
fl_GarageArea_mean = np.mean(df_X["GarageArea"])

pos_row = 0
for i in df_X["GarageArea"]:
    pos_row = pos_row + 1
    if not(np.isfinite(i)):     #if not finite
        i = fl_GarageArea_mean
#        print(pos_row)

df_X["GarageArea"].replace(np.NaN,0,inplace = True)
# print(df_X.iloc[1115:1118,:])


# **************************************** encoding ****************************************
df_X['KitchenQual'] = df_X['KitchenQual'].astype(str)  # convert to string
encoder = LabelEncoder()
encoder.fit(df_X['KitchenQual'])
df_X['KitchenQual'] = encoder.transform(df_X['KitchenQual'])

df_X['Neighborhood'] = df_X['Neighborhood'].astype(str)  # convert to string
encoder = LabelEncoder()
encoder.fit(df_X['Neighborhood'])
df_X['Neighborhood'] = encoder.transform(df_X['Neighborhood'])


# **************************************** standardizing ****************************************
f_min_max = lambda x: (x-np.min(x)) / (np.max(x) - np.min(x))
df_X = df_X.apply(f_min_max)


# **************************************** predict ****************************************
fl_predict_y = RF_regression_model.predict(df_X)


# **************************************** output ****************************************
df_output = df_test_data[['Id']].copy()
df_output["SalePrice"] = fl_predict_y
df_output.to_csv("my_predict.csv", index=False, encoding='utf-8')
'''
f_out =  open("my_predict","w")
f_out.write(df_output.__str__())
f_out.close()
'''



