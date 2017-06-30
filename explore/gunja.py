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
DEFAULT_TRAIN_ROUNDS = 100

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
df.loc[(df.full_sq<10) & (df.life_sq>10), 'full_sq'] = df.life_sq
# Clean - Life Sq
df.loc[(df.life_sq>100) & (df.full_sq<100), 'life_sq'] = df.life_sq.astype(float)/10
df.loc[df.life_sq > df.full_sq, 'life_sq'] = np.nan
df.loc[df.life_sq < 5, 'life_sq'] = np.nan
df.loc[df.life_sq < df.full_sq * 0.3, 'life_sq'] = np.nan
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
test_df.loc[test_df.life_sq>test_df.full_sq*2, 'life_sq'] = test_df.life_sq.astype(float)/10
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


#print df.describe()


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



# ----------------- Price Adjustment ----------------- #

# Rates
rate_2016_q2 = 1
rate_2016_q1 = rate_2016_q2 / .99903
rate_2015_q4 = rate_2016_q1 / .9831
rate_2015_q3 = rate_2015_q4 / .9834
rate_2015_q2 = rate_2015_q3 / .9815
rate_2015_q1 = rate_2015_q2 / .9932
rate_2014_q4 = rate_2015_q1 / 1.0112
rate_2014_q3 = rate_2014_q4 / 1.0169
rate_2014_q2 = rate_2014_q3 / 1.0086
rate_2014_q1 = rate_2014_q2 / 1.0126
rate_2013_q4 = rate_2014_q1 / 0.9902
rate_2013_q3 = rate_2013_q4 / 1.0041
rate_2013_q2 = rate_2013_q3 / 1.0044
rate_2013_q1 = rate_2013_q2 / 1.0104
rate_2012_q4 = rate_2013_q1 / 0.9832
rate_2012_q3 = rate_2012_q4 / 1.0277
rate_2012_q2 = rate_2012_q3 / 1.0279
rate_2012_q1 = rate_2012_q2 / 1.0279
rate_2011_q4 = rate_2012_q1 / 1.076
rate_2011_q3 = rate_2011_q4 / 1.0236
rate_2011_q2 = rate_2011_q3 / 1
rate_2011_q1 = rate_2011_q2 / 1.011


# Test Price Adjustments
test_df['average_q_price'] = 1
# 2016 Q2
test_df_2016_q2_index = test_df.loc[test_df['timestamp'].dt.year == 2016].loc[test_df['timestamp'].dt.month >= 4].loc[test_df['timestamp'].dt.month <= 7].index
test_df.loc[test_df_2016_q2_index, 'average_q_price'] = rate_2016_q2
# 2016 Q1
test_df_2016_q1_index = test_df.loc[test_df['timestamp'].dt.year == 2016].loc[test_df['timestamp'].dt.month >= 1].loc[test_df['timestamp'].dt.month < 4].index
test_df.loc[test_df_2016_q1_index, 'average_q_price'] = rate_2016_q1
# 2015 Q4
test_df_2015_q4_index = test_df.loc[test_df['timestamp'].dt.year == 2015].loc[test_df['timestamp'].dt.month >= 10].loc[test_df['timestamp'].dt.month < 12].index
test_df.loc[test_df_2015_q4_index, 'average_q_price'] = rate_2015_q4
# 2015 Q3
test_df_2015_q3_index = test_df.loc[test_df['timestamp'].dt.year == 2015].loc[test_df['timestamp'].dt.month >= 7].loc[test_df['timestamp'].dt.month < 10].index
test_df.loc[test_df_2015_q3_index, 'average_q_price'] = rate_2015_q3


# Train Price Adjustments
df['average_q_price'] = 1
# 2015 Q4
df_2015_q4_index = df.loc[df['timestamp'].dt.year == 2015].loc[df['timestamp'].dt.month >= 10].loc[df['timestamp'].dt.month <= 12].index
df.loc[df_2015_q4_index, 'average_q_price'] = rate_2015_q4
# 2015 Q3
df_2015_q3_index = df.loc[df['timestamp'].dt.year == 2015].loc[df['timestamp'].dt.month >= 7].loc[df['timestamp'].dt.month < 10].index
df.loc[df_2015_q3_index, 'average_q_price'] = rate_2015_q3
# 2015 Q2
df_2015_q2_index = df.loc[df['timestamp'].dt.year == 2015].loc[df['timestamp'].dt.month >= 4].loc[df['timestamp'].dt.month < 7].index
df.loc[df_2015_q2_index, 'average_q_price'] = rate_2015_q2
# 2015 Q1
df_2015_q1_index = df.loc[df['timestamp'].dt.year == 2015].loc[df['timestamp'].dt.month >= 1].loc[df['timestamp'].dt.month < 4].index
df.loc[df_2015_q1_index, 'average_q_price'] = rate_2015_q1

# 2014 Q4
df_2014_q4_index = df.loc[df['timestamp'].dt.year == 2014].loc[df['timestamp'].dt.month >= 10].loc[df['timestamp'].dt.month <= 12].index
df.loc[df_2014_q4_index, 'average_q_price'] = rate_2014_q4
# 2014 Q3
df_2014_q3_index = df.loc[df['timestamp'].dt.year == 2014].loc[df['timestamp'].dt.month >= 7].loc[df['timestamp'].dt.month < 10].index
df.loc[df_2014_q3_index, 'average_q_price'] = rate_2014_q3
# 2014 Q2
df_2014_q2_index = df.loc[df['timestamp'].dt.year == 2014].loc[df['timestamp'].dt.month >= 4].loc[df['timestamp'].dt.month < 7].index
df.loc[df_2014_q2_index, 'average_q_price'] = rate_2014_q2
# 2014 Q1
df_2014_q1_index = df.loc[df['timestamp'].dt.year == 2014].loc[df['timestamp'].dt.month >= 1].loc[df['timestamp'].dt.month < 4].index
df.loc[df_2014_q1_index, 'average_q_price'] = rate_2014_q1

# 2013 Q4
df_2013_q4_index = df.loc[df['timestamp'].dt.year == 2013].loc[df['timestamp'].dt.month >= 10].loc[df['timestamp'].dt.month <= 12].index
df.loc[df_2013_q4_index, 'average_q_price'] = rate_2013_q4
# 2013 Q3
df_2013_q3_index = df.loc[df['timestamp'].dt.year == 2013].loc[df['timestamp'].dt.month >= 7].loc[df['timestamp'].dt.month < 10].index
df.loc[df_2013_q3_index, 'average_q_price'] = rate_2013_q3
# 2013 Q2
df_2013_q2_index = df.loc[df['timestamp'].dt.year == 2013].loc[df['timestamp'].dt.month >= 4].loc[df['timestamp'].dt.month < 7].index
df.loc[df_2013_q2_index, 'average_q_price'] = rate_2013_q2
# 2013 Q1
df_2013_q1_index = df.loc[df['timestamp'].dt.year == 2013].loc[df['timestamp'].dt.month >= 1].loc[df['timestamp'].dt.month < 4].index
df.loc[df_2013_q1_index, 'average_q_price'] = rate_2013_q1

# 2012 Q4
df_2012_q4_index = df.loc[df['timestamp'].dt.year == 2012].loc[df['timestamp'].dt.month >= 10].loc[df['timestamp'].dt.month <= 12].index
df.loc[df_2012_q4_index, 'average_q_price'] = rate_2012_q4
# 2012 Q3
df_2012_q3_index = df.loc[df['timestamp'].dt.year == 2012].loc[df['timestamp'].dt.month >= 7].loc[df['timestamp'].dt.month < 10].index
df.loc[df_2012_q3_index, 'average_q_price'] = rate_2012_q3
# 2012 Q2
df_2012_q2_index = df.loc[df['timestamp'].dt.year == 2012].loc[df['timestamp'].dt.month >= 4].loc[df['timestamp'].dt.month < 7].index
df.loc[df_2012_q2_index, 'average_q_price'] = rate_2012_q2
# 2012 Q1
df_2012_q1_index = df.loc[df['timestamp'].dt.year == 2012].loc[df['timestamp'].dt.month >= 1].loc[df['timestamp'].dt.month < 4].index
df.loc[df_2012_q1_index, 'average_q_price'] = rate_2012_q1

# 2011 Q4
df_2011_q4_index = df.loc[df['timestamp'].dt.year == 2011].loc[df['timestamp'].dt.month >= 10].loc[df['timestamp'].dt.month <= 12].index
df.loc[df_2011_q4_index, 'average_q_price'] = rate_2011_q4
# 2011 Q3
df_2011_q3_index = df.loc[df['timestamp'].dt.year == 2011].loc[df['timestamp'].dt.month >= 7].loc[df['timestamp'].dt.month < 10].index
df.loc[df_2011_q3_index, 'average_q_price'] = rate_2011_q3
# 2011 Q2
df_2011_q2_index = df.loc[df['timestamp'].dt.year == 2011].loc[df['timestamp'].dt.month >= 4].loc[df['timestamp'].dt.month < 7].index
df.loc[df_2011_q2_index, 'average_q_price'] = rate_2011_q2
# 2011 Q1
df_2011_q1_index = df.loc[df['timestamp'].dt.year == 2011].loc[df['timestamp'].dt.month >= 1].loc[df['timestamp'].dt.month < 4].index
df.loc[df_2011_q1_index, 'average_q_price'] = rate_2011_q1

# Price Adjustment
df['price_doc'] = df['price_doc'] * df['average_q_price']



# ----------------- Prepare Training and Test Data ----------------- #

# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.title('Original Data Set')


# ----------------- Prepare Training and Test Data ----------------- #
y_train = df["price_doc"]
x_train = df.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
x_test  = test_df.drop(["id", "timestamp", "average_q_price"], axis=1)
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
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.5,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'seed': 0
}

if EN_CROSSVALIDATION:
    print "[INFO] Cross Validation..."
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                   verbose_eval=20, show_stdv=True)
    DEFAULT_TRAIN_ROUNDS = len(cv_output)
    print "[INFO] Optimal Training Rounds =", DEFAULT_TRAIN_ROUNDS

'''
# ----------------- Training ----------------- #
print "[INFO] Training for", DEFAULT_TRAIN_ROUNDS, "rounds..."
model      = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAIN_ROUNDS, 
                       evals=[(dtrain, 'train')], verbose_eval=10)
y_predict  = model.predict(dtest)
gunja_output = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict, 'average_q_price': test_df['average_q_price']})
gunja_output['price_doc'] = gunja_output['price_doc'] * gunja_output['average_q_price']
gunja_output.drop('average_q_price', axis=1, inplace=True)
print gunja_output.head()
gunja_output.to_csv('gunja_output.csv', index=False)
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
'''



