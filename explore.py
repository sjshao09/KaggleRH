import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


# Read Training Data Set
df = pd.read_csv('input/train.csv')
print "Original Training Data Shape: ", df.shape

# Print log(1+price_doc) values
log1y = np.log1p(df['price_doc'])
fig1 = plt.figure()
plt.hist(log1y, bins=200, color='b')
plt.xlabel('log(1+price_doc)')
plt.ylabel('Count')
plt.title('Distribution of log(1+price_doc)')
fig1.show()

# Top 40 features with most missing data
MissCount = df.isnull().sum().sort_values(ascending=False).head(40) / len(df) * 100
fig2 = plt.figure(figsize=(8, 12))
plt.barh(np.arange(len(MissCount)), MissCount)
plt.yticks(np.arange(len(MissCount))+0.5, MissCount.index, rotation='horizontal')
plt.title('Percentage of Missing Data')
plt.tight_layout()
fig2.show()


# Drop Error Row (id=10092, state=33, buildyear=20052009)
df = df[df.id !=10092]

## Numerical and Categorical data types
#df_dtype = df.dtypes
#display_nvar = len(alldata.columns)
#df_dtype_dict = alldata_dtype.to_dict()
print df.dtypes.value_counts()


# Object Columns
NonNumColName = ['timestamp', 'product_type', 'sub_area', 'culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 'ecology']
NonNumCol = df[NonNumColName]
#print NonNumCol.describe()

# Drop Non-Numerical Features and id
ColToDrop = NonNumColName + ['id']
df = df.drop(ColToDrop, axis=1)
print "Trimmed Training Data Shape: ", df.shape

# Feature Importance by Xgboost
low_y_cut  = 1*1e6
high_y_cut = 40*1e6

df = df.sample(frac=0.1)
df.fillna(df.median(axis=0), inplace=True)
y_is_within_cut = ((df['price_doc'] > low_y_cut) & (df['price_doc'] < high_y_cut))

train_X = df.loc[y_is_within_cut, df.columns[:-1]]
train_y = np.log1p(df.loc[y_is_within_cut, 'price_doc'].values.reshape(-1, 1))
print("Data for model: X={}, y={}".format(train_X.shape, train_y.shape))

model = xgb.XGBRegressor()
model.fit(train_X, train_y)
fig3 = plt.figure(figsize=(7,30))
ax3  = fig3.add_subplot(111)
xgb.plot_importance(model, ax=ax3)
plt.tight_layout()


# Display Figures
plt.show()


# Test
# Read Test Data
test_df = pd.read_csv('input/test.csv')
test_X  = test_df.drop(ColToDrop, axis=1)
test_y_predict = np.exp(model.predict(test_X))-1
submission = pd.DataFrame(index=test_df['id'], data={'price_doc':test_y_predict})
print submission.head()
submission.to_csv('submission.csv', header=True)

print "Finished"

