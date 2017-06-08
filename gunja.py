import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime

# ----------------- Settings ----------------- #
EN_CROSSVALIDATION   = True
EN_IMPORTANCE        = True
DEFAULT_TRAIN_ROUNDS = 384

#load files
df      = pd.read_csv('input/train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv('input/test.csv', parse_dates=['timestamp'])
macro   = pd.read_csv('input/macro.csv', parse_dates=['timestamp'])


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
df.loc[df.life_sq < 5, 'life_sq'] = np.nan
df.loc[df.life_sq < df.full_sq * 0.3, 'life_sq'] = np.nan
df = df[df.life_sq<300]
# Clean - Kitch Sq
df.loc[df.kitch_sq < 2, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.full_sq * 0.5, 'kitch_sq'] = np.nan
df.loc[df.kitch_sq > df.life_sq, 'kitch_sq'] = np.nan

# Clean - Build Year
df.loc[df.build_year<1000, 'build_year'] = np.nan
df.loc[df.build_year>2050, 'build_year'] = np.nan
# Clean - Num Room
df.loc[df.num_room<1, 'num_room'] = np.nan
df.loc[(df.num_room>4) & (df.full_sq<60), 'num_room'] = np.nan
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
test_df.loc[test_df.life_sq < 5, 'life_sq'] = np.nan
test_df.loc[test_df.life_sq < test_df.full_sq * 0.3, 'life_sq'] = np.nan
# Clean - Kitch Sq
test_df.loc[test_df.kitch_sq < 2, 'kitch_sq'] = np.nan
test_df.loc[test_df.kitch_sq > test_df.full_sq * 0.5, 'kitch_sq'] = np.nan
test_df.loc[test_df.kitch_sq > test_df.life_sq, 'kitch_sq'] = np.nan
# Clean - Build Year
test_df.loc[test_df.build_year<1000, 'build_year'] = np.nan
test_df.loc[test_df.build_year>2050, 'build_year'] = np.nan
# Clean - Num Room
test_df.loc[test_df.num_room<1, 'num_room'] = np.nan
test_df.loc[(test_df.num_room>4) & (test_df.full_sq<60), 'num_room'] = np.nan
# Clean - Floor and Max Floor
test_df.loc[test_df.floor==0, 'floor'] = np.nan
test_df.loc[test_df.max_floor==0, 'max_floor'] = np.nan
test_df.loc[(test_df.max_floor==1) & (test_df.floor>1), 'max_floor'] = np.nan
test_df.loc[test_df.max_floor>50, 'max_floor'] = np.nan
test_df.loc[test_df.floor>test_df.max_floor, 'floor'] = np.nan


# ----------------- Outlier Removal ----------------- #
df = df[df.price_doc/df.full_sq <= 600000]
df = df[df.price_doc/df.full_sq >= 10000]


# ----------------- New Features ----------------- #
# month_year_cnt
month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df['month_year_cnt'] = month_year.map(month_year_cnt_map)
month_year = (test_df.timestamp.dt.month + test_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)
# week_year_cnt
week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df['week_year_cnt'] = week_year.map(week_year_cnt_map)
week_year = (test_df.timestamp.dt.weekofyear + test_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)
# month
df['month']      = df.timestamp.dt.month
test_df['month'] = test_df.timestamp.dt.month
# day of week
df['dow']        = df.timestamp.dt.dayofweek
test_df['dow']   = test_df.timestamp.dt.dayofweek
# floor/max_floor
df['floor/max_floor']      = df['floor'] / df['max_floor'].astype(float)
test_df['floor/max_floor'] = test_df['floor'] / test_df['max_floor'].astype(float)
# kitch_sq/full_sq
df["kitch_sq/full_sq"]      = df["kitch_sq"] / df["full_sq"].astype(float)
test_df["kitch_sq/full_sq"] = test_df["kitch_sq"] / test_df["full_sq"].astype(float)
# Avg Room Size
df['avg_room_size']      = df['life_sq'] / df['num_room'].astype(float)
test_df['avg_room_size'] = test_df['life_sq'] / test_df['num_room'].astype(float)
# Apartment Name 
df['apartment_name']      = df['sub_area'] + df['metro_km_avto'].astype(str)
test_df['apartment_name'] = test_df['sub_area'] + test_df['metro_km_avto'].astype(str)


# ----------------- Prepare Training and Test Data ----------------- #

# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.title('Original Data Set')


# ----------------- Prepare Training and Test Data ----------------- #
y_train = df["price_doc"]
x_train = df.drop(["id", "timestamp", "price_doc"], axis=1)
x_test  = test_df.drop(["id", "timestamp"], axis=1)
x_all   = pd.concat([x_train, x_test])

# Feature Encoding
for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values)) 
        x_all[c] = lbl.transform(list(x_all[c].values))

# Separate Training and Test Data
num_train = len(x_train)
x_train = x_all[:num_train]
x_test  = x_all[num_train:]
dtrain  = xgb.DMatrix(x_train, y_train)
dtest   = xgb.DMatrix(x_test)


# ----------------- Cross Validation ----------------- #

xgb_params = {
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'seed': 0
}

if EN_CROSSVALIDATION:
    print "[INFO] Cross Validation..."
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=10,
                   verbose_eval=20, show_stdv=True)
    DEFAULT_TRAIN_ROUNDS = len(cv_output)
    print "[INFO] Optimal Training Rounds =", DEFAULT_TRAIN_ROUNDS


# ----------------- Training ----------------- #
print "[INFO] Training for", DEFAULT_TRAIN_ROUNDS, "rounds..."
model      = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAIN_ROUNDS, 
                       evals=[(dtrain, 'train')], verbose_eval=10)
y_predict  = model.predict(dtest)
y_predict  = np.round(y_predict * 0.99)
gunja_output = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})
gunja_output.to_csv('submission.csv', index=False)
print gunja_output.head()
print "[INFO] Average Price =", gunja_output['price_doc'].mean()

# Plot Original, Training and Test Sets
ax4 = plt.subplot(312, sharex=ax1)
plt.hist(np.log1p(y_train), bins=200, color='b')
plt.title('Training Data Set')
plt.subplot(313, sharex=ax1)
plt.hist(np.log1p(y_predict), bins=200, color='b')
plt.title('Test Data Set Prediction')
OrigTrainValidSetFig.show()

# Plot Feature Importance
if EN_IMPORTANCE:
    fig, ax = plt.subplots(1, 1, figsize=(8, 13))
    xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
    plt.tight_layout()

plt.show()




