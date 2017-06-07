import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import geopandas as gpd
from pyproj import Proj, transform, Geod


# ----------------- Settings ----------------- #
EN_CROSSVALIDATION   = True
EN_IMPORTANCE        = True
DEFAULT_TRAIN_ROUNDS = 384

# ----------------- Read Data ----------------- #
df      = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
macro   = pd.read_csv('input/macro.csv')
shp     = gpd.read_file('input/moscow_adm.shp')

# ----------------- Data Cleaning ----------------- #
# Training Set
df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==13120, 'build_year'] = 1970
df.loc[df.id==25943, 'max_floor'] = 17
# Clean - Full Sq
df = df[(df.full_sq>1)|(df.life_sq>1)]
df.loc[(df.full_sq<10) & (df.life_sq>1), 'full_sq'] = df.life_sq
df = df[df.full_sq<400]
# Clean - Life Sq
df.loc[df.life_sq > df.full_sq*4, 'life_sq'] = df.life_sq/10
df.loc[df.life_sq > df.full_sq, 'life_sq'] = np.nan
df.loc[df.life_sq < 2, 'life_sq'] = np.nan
df.loc[df.life_sq < df.full_sq * 0.3, 'life_sq'] = np.nan
# Clean - Kitch Sq
df.loc[df.kitch_sq < 2, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.full_sq * 0.5, 'kitch_sq'] = np.nan
# Clean - Build Year
df.loc[df.build_year<1000, 'build_year'] = np.nan
df.loc[df.build_year>2050, 'build_year'] = np.nan
# Clean - Num Room
df.loc[df.num_room<1, 'num_room'] = np.nan
df.loc[(df.num_room>9) & (df.full_sq<100), 'num_room'] = np.nan
# Clean - Floor and Max Floor
df.loc[df.floor==0, 'floor'] = np.nan
df.loc[df.max_floor==0, 'max_floor'] = np.nan
df.loc[(df.max_floor==1) & (df.floor>1), 'max_floor'] = np.nan
df.loc[df.max_floor>50, 'max_floor'] = np.nan
df.loc[df.floor>df.max_floor, 'floor'] = np.nan

# Test Set
test_df.loc[test_df.id==30938,  'full_sq'] = 37.8
test_df.loc[test_df.id==35857,  'full_sq'] = 42.07
test_df.loc[test_df.id==35108,  'full_sq'] = 40.3
test_df.loc[test_df.id==33648,  'num_room'] = 1
# Clean - Full Sq
test_df.loc[(test_df.full_sq<10) & (test_df.life_sq>1), 'full_sq'] = test_df.life_sq
# Clean - Life Sq
test_df.loc[test_df.life_sq>test_df.full_sq*2, 'life_sq'] = test_df.life_sq/10
test_df.loc[test_df.life_sq > test_df.full_sq, 'life_sq'] = np.nan
test_df.loc[test_df.life_sq < 2, 'life_sq'] = np.nan
test_df.loc[test_df.life_sq < test_df.full_sq * 0.3, 'life_sq'] = np.nan
# Clean - Kitch Sq
test_df.loc[test_df.kitch_sq < 2, 'kitch_sq'] = np.nan
test_df.loc[test_df.kitch_sq > test_df.full_sq * 0.5, 'kitch_sq'] = np.nan
# Clean - Build Year
test_df.loc[test_df.build_year<1000, 'build_year'] = np.nan
test_df.loc[test_df.build_year>2050, 'build_year'] = np.nan
# Clean - Num Room
test_df.loc[test_df.num_room<1, 'num_room'] = np.nan
test_df.loc[(test_df.num_room>9) & (test_df.full_sq<100), 'num_room'] = np.nan
# Clean - Floor and Max Floor
test_df.loc[test_df.floor==0, 'floor'] = np.nan
test_df.loc[test_df.max_floor==0, 'max_floor'] = np.nan
test_df.loc[(test_df.max_floor==1) & (test_df.floor>1), 'max_floor'] = np.nan
test_df.loc[test_df.max_floor>50, 'max_floor'] = np.nan
test_df.loc[test_df.floor>test_df.max_floor, 'floor'] = np.nan

# ----------------- New Features ----------------- #
# Auxiliary Feature - price/sq
df['price/sq'] = df['price_doc'] / df['full_sq']
# New Feature - bad_floor
df['bad_floor'] = (df['floor']==1) | (df['floor']==df['max_floor'])
test_df['bad_floor'] = (test_df['floor']==1) | (test_df['floor']==test_df['max_floor'])
# New Feature - kitch_sq/full_sq
df["kitch_sq/full_sq"] = df["kitch_sq"] / df["full_sq"]
test_df["kitch_sq/full_sq"] = test_df["kitch_sq"] / test_df["full_sq"]
# log size
df["full_sq"] = np.log(df["full_sq"])
df["life_sq"] = np.log(df["life_sq"])
df["kitch_sq"] = np.log(df["kitch_sq"])
test_df["full_sq"] = np.log(test_df["full_sq"])
test_df["life_sq"] = np.log(test_df["life_sq"])
test_df["kitch_sq"] = np.log(test_df["kitch_sq"])


# ----------------- Macro Data ----------------- #
MacroFeatures = ['timestamp', 'usdrub', 'oil_urals', 'mortgage_rate', 'cpi', 'ppi', 'rent_price_2room_eco', 'micex',
'rent_price_1room_eco', 'balance_trade', 'balance_trade_growth', 'gdp_quart_growth', 'net_capital_export']
macro = macro[MacroFeatures]
df      = pd.merge(df, macro, on='timestamp', how='left')
test_df = pd.merge(test_df, macro, on='timestamp', how='left')

# Price in USD
df['price/usd'] = df['price_doc'] / df['usdrub'] * 0.9
Target = 'price/usd'

# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.title('Original Data Set')



# ----------------- Training Data ----------------- #
y_train = np.log1p(df[Target])
x_train = df.drop(["id", "timestamp", "price_doc", "price/usd", "price/sq"], axis=1)
# Encoding
for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
# Pack into DMatrix
dtrain  = xgb.DMatrix(x_train, y_train)


# ----------------- Test Data ----------------- #
x_test  = test_df.drop(["id", "timestamp", ], axis=1)
# Encoding        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
# Pack into DMatrix
dtest = xgb.DMatrix(x_test)


# ----------------- Parameters ----------------- #
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'nthread': 6,
    'seed': 0
}


# ----------------- Cross Validation ----------------- #
if EN_CROSSVALIDATION:
    print "[INFO] Cross Validation..."
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                   verbose_eval=10, show_stdv=True)
    DEFAULT_TRAIN_ROUNDS = len(cv_output)
    print "[INFO] Optimal Training Rounds =", DEFAULT_TRAIN_ROUNDS

# ----------------- Training ----------------- #
print "[INFO] Training for", DEFAULT_TRAIN_ROUNDS, "rounds..."
model = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAIN_ROUNDS, 
                       evals=[(dtrain, 'train')], verbose_eval=10)
y_hat = pd.Series(np.expm1(model.predict(dtest)), name='price/usd')
y_hat.to_frame()
test_df = test_df.join(y_hat)
test_df['price_doc'] = test_df['price/usd'] * test_df['usdrub']
y_predict = test_df['price_doc'].values
submission = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})
submission.to_csv('submission.csv', index=False)
print submission.head()
print "[INFO] Average Price =", test_df['price_doc'].mean()

# Plot Original, Training and Test Sets
ax4 = plt.subplot(312, sharex=ax1)
plt.hist(np.log1p(df['price_doc']), bins=200, color='b')
plt.title('Training Data Set')
plt.subplot(313, sharex=ax1)
plt.hist(np.log1p(test_df['price_doc']), bins=200, color='b')
plt.title('Test Data Set Prediction')
OrigTrainValidSetFig.show()

# Plot Feature Importance
if EN_IMPORTANCE:
    fig, ax = plt.subplots(1, 1, figsize=(8, 13))
    xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
    plt.tight_layout()

plt.show()

