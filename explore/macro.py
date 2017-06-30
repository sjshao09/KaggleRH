import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Number of Lags to use
NUM_LAGS = 0

# Import Data
macro = pd.read_csv('input/macro.csv')
train = pd.read_csv('input/train.csv')

# Macro Data
MacroFeatures = ['timestamp', 'usdrub', 'oil_urals', 'mortgage_rate', 'cpi', 'ppi', 'rent_price_2room_eco',
'rent_price_1room_eco', 'balance_trade', 'balance_trade_growth', 'gdp_quart_growth', 'net_capital_export']
macro = macro[MacroFeatures]
macro["timestamp"] = pd.to_datetime(macro["timestamp"])
macro["year"]      = macro["timestamp"].dt.year
macro["month"]     = macro["timestamp"].dt.month
macro["yearmonth"] = 100*macro.year + macro.month
macro = macro.groupby("yearmonth").median()

# Create Lagged Macro Features
FeaturesToBeLagged = copy.deepcopy(MacroFeatures)
FeaturesToBeLagged.remove('timestamp')
for feature in FeaturesToBeLagged:
    for i in range(1, NUM_LAGS+1):
        LaggedFeatureName = feature + "_" + str(i)
        macro[LaggedFeatureName] = macro[feature].shift(i)

# Training Data
# Data Cleaning
train.loc[train.id==13549, 'life_sq'] = 74
train.loc[train.id==10092, 'build_year'] = 2007
train.loc[train.id==10092, 'state'] = 3
train.loc[train.id==13120, 'build_year'] = 1970
train.loc[train.id==25943, 'max_floor'] = 17
# Clean - Full Sq
train = train[(train.full_sq>1)|(train.life_sq>1)]
train.loc[(train.full_sq<10) & (train.life_sq>1), 'full_sq'] = train.life_sq
train = train[train.full_sq<400]
# Price_doc and price_doc/sq?
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["year"]      = train["timestamp"].dt.year
train["month"]     = train["timestamp"].dt.month
train["yearmonth"] = 100*train.year + train.month
train["price/sq"]  = train["price_doc"] / train["full_sq"]
prices = train[["yearmonth", "price_doc", "price/sq"]]
p = prices.groupby("yearmonth").median()
print p.head()

# Merge Training Data with Macro Data
df = macro.join(p)
df.fillna(df.median(axis=0), inplace=True)
df["price/cpi"]    = df["price_doc"] / df["cpi"]
df["price/usd"]    = df["price_doc"] / df["usdrub"]
df["price/sq/cpi"] = df["price/sq"]  / df["cpi"]

# Target
Target = "price/usd"
ColToDrop = ["price_doc", "price/sq", "price/cpi", "price/sq/cpi", "price/usd"]
# Prepare Data for Model
train_y = df.loc[201108:201412, Target] 
train_X = df.drop(ColToDrop, axis=1)
train_X = train_X.loc[201108:201412]

# Linear Regression - Training
model = LinearRegression()
model.fit(train_X.values, train_y.values)
print("Intercept of the model: {}".format(model.intercept_))
print("Coefficients of the model: {}".format(model.coef_))
TrainFig = plt.figure()
plt.plot(train_y.values, 'ko')
plt.plot(pd.Series(model.predict(train_X)).values, color='blue')
TrainFig.show()

# Linear Regression - Validation
valid_y = df.loc[201501:201506, Target]
valid_X = df.drop(ColToDrop, axis=1)
valid_X = valid_X.loc[201501:201506]
valid_y_predict = model.predict(valid_X)
ValidFig = plt.figure()
plt.plot(valid_y.values, 'ko')
plt.plot(pd.Series(valid_y_predict).values, color='blue')
ValidFig.show()
print "[INFO] MSE is", mean_squared_error(valid_y, valid_y_predict)


# Linear Regression - Prediction
test_X = df.drop(ColToDrop, axis=1)
test_X = test_X.loc[201507:201605]
test_y_predict = model.predict(test_X)
TestFig = plt.figure()
plt.plot(pd.Series(test_y_predict).values, color='blue')
TestFig.show()

plt.show()

