import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


# Read Training Data Set
df = pd.read_csv('input/train.csv')
print "Original Training Data Shape: ", df.shape

# Print log(1+price_doc) values
#y = df['price_doc']
df = df.assign(log1p_price_doc = lambda x: np.log1p(x.price_doc))
fig1 = plt.figure()
plt.hist(df['log1p_price_doc'], bins=200, color='b')
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

# Non-Numerical Features
NonNumColName = ['timestamp', 'product_type', 'sub_area', 'culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 'ecology']
NonNumCol = df[NonNumColName]
#print NonNumCol.describe()

# Drop Non-Numerical Features and price_doc, id
ColToDrop = NonNumColName + ['id'] + ['price_doc']
df = df.drop(ColToDrop, axis=1)
print "Trimmed Training Data Shape: ", df.shape

# Feature Importance by Xgboost
low_y_cut  = np.log1p(1*1e6)
high_y_cut = np.log1p(40*1e6)

df = df.sample(frac=0.1)
df.fillna(df.median(axis=0), inplace=True)
y_is_within_cut = ((df['log1p_price_doc'] > low_y_cut) & (df['log1p_price_doc'] < high_y_cut))

train_X = df.loc[y_is_within_cut, df.columns[:-1]]
train_y = df.loc[y_is_within_cut, 'log1p_price_doc'].values.reshape(-1, 1)
print("Data for model: X={}, y={}".format(train_X.shape, train_y.shape))

model = xgb.XGBRegressor()
model.fit(train_X, train_y)
fig3 = plt.figure(figsize=(7,30))
ax3  = fig3.add_subplot(111)
xgb.plot_importance(model, ax=ax3)
plt.tight_layout()


# Display Figures
plt.show()

