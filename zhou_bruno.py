import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("input/macro.csv", parse_dates=['timestamp'])

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)
mult = 0.969
y_train = df_train['price_doc'].values  * mult + 10
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
# Next line just adds a lot of NA columns (becuase "join" only works on indexes)
# but somewhow it seems to affect the result
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



######## BEGIN 2ND SET OF BILL S CHANGES 
'''
## same ones as above 
df_all['area_per_room'] = df_all['life_sq'] / df_all['num_room'].astype(float)
df_all['livArea_ratio'] = df_all['life_sq'] / df_all['full_sq'].astype(float)
df_all['yrs_old'] = 2017 - df_all['build_year'].astype(float)
df_all['avgfloor_sq'] = df_all['life_sq']/df_all['max_floor'].astype(float) #living area per floor
df_all['pts_floor_ratio'] = df_all['public_transport_station_km']/df_all['max_floor'].astype(float) #apartments near public t?
#f_all['room_size'] = df_all['life_sq'] / df_all['num_room'].astype(float)
df_all['gender_ratio'] = df_all['male_f']/df_all['female_f'].astype(float)
df_all['kg_park_ratio'] = df_all['kindergarten_km']/df_all['park_km'].astype(float)
df_all['high_ed_extent'] = df_all['school_km'] / df_all['kindergarten_km']
df_all['pts_x_state'] = df_all['public_transport_station_km'] * df_all['state'].astype(float) #public trans * state of listing
df_all['lifesq_x_state'] = df_all['life_sq'] * df_all['state'].astype(float)
df_all['floor_x_state'] = df_all['floor'] * df_all['state'].astype(float)
'''
#########  END 2ND SET OF BILL S CHANGES


# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)


factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'seed': 0,
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)
'''
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)
print('best num_boost_rounds = ', len(cv_output))
num_boost_rounds = len(cv_output)
'''
num_boost_rounds = 420  # From Bruno's original CV, I think
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds, evals=[(dtrain, 'train')], verbose_eval=10)

'''
# ---------------------- Predict Training Set for Ensemble --------------------- #
# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("input/macro.csv", parse_dates=['timestamp'])

df_train.loc[df_train.life_sq>7000, 'life_sq'] = 74
mult = 0.969
y_train = df_train['price_doc'].values  * mult + 10
id_train = df_train['id']
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
# Next line just adds a lot of NA columns (becuase "join" only works on indexes)
# but somewhow it seems to affect the result
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)

factorize = lambda t: pd.factorize(t[1])[0]
df_obj = df_all.select_dtypes(include=['object'])
X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
X_train = X_all[:num_train]
X_test = X_all[num_train:]


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)
# Convert to numpy values
X_all = df_values.values
X_train = X_all[:num_train]
X_test = X_all[num_train:]
dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)

train_predict = model.predict(dtrain)
train_predict_df = pd.DataFrame({'id': id_train, 'price_doc': train_predict})
train_predict_df.to_csv('bruno_train.csv', index=False)
# ---------------------- Predict Training Set for Ensemble -------end---------- #
'''

y_pred = model.predict(dtest)

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
print "[INFO] Bruno Model Average Price =", df_sub['price_doc'].mean()
df_sub.to_csv('bruno_test.csv', index=False)
print df_sub.head()
