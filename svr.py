import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


# ----------------- Settings ----------------- #
EN_CROSSVALIDATION = False


# ----------------- Read Data ----------------- #
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
# price/sq
df['price/sq'] = df['price_doc'] / df['full_sq']
# New Feature - bad_floor
df['bad_floor'] = (df['floor']==1) | (df['floor']==df['max_floor'])
test_df['bad_floor'] = (test_df['floor']==1) | (test_df['floor']==test_df['max_floor'])
# log size
df["full_sq"] = np.log(df["full_sq"])
df["life_sq"] = np.log(df["life_sq"])
df["kitch_sq"] = np.log(df["kitch_sq"])
test_df["full_sq"] = np.log(test_df["full_sq"])
test_df["life_sq"] = np.log(test_df["life_sq"])
test_df["kitch_sq"] = np.log(test_df["kitch_sq"])


# ----------------- Handle Missing Data ----------------- #
# Fill by Mean
df.fillna(df.median(axis=0), inplace=True)
test_df['product_type'].fillna(test_df['product_type'].mode().iloc[0], inplace=True)
test_df.fillna(df.median(axis=0), inplace=True)


# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.title('Original Data Set')


# ----------------- Training Data ----------------- #
y_train = np.log1p(df['price_doc'])
x_train = df.drop(["id", "timestamp", "price_doc", "price/sq"], axis=1)
# Encoding
for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))


# ----------------- Test Data ----------------- #
x_test  = test_df.drop(["id", "timestamp"], axis=1)
# Encoding        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))

# ----------------- Standardlisation ----------------- #
scaler  = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)


# ----------------- Cross Validation ----------------- #
if EN_CROSSVALIDATION:
    print "[INFO] Cross Validation..."
    model = SVR(kernel='rbf', tol=0.001, cache_size=4096)
    param_dist = {'gamma': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
                  'C': [8, 16, 24, 32, 40, 48, 56, 64, 72, 80],
                  'epsilon': [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
                 }
    # Scoring: 'neg_mean_squared_error', 'neg_mean_squared_log_error'
    r = RandomizedSearchCV(model, param_distributions=param_dist, scoring='neg_mean_squared_error', 
                           cv=4, n_iter=128, iid=False, verbose=2, n_jobs=6)
    r.fit(x_train, y_train)
    print ('Best score: {}'.format(r.best_score_))
    print ('Best parameters: {}'.format(r.best_params_))
    top_parameters = pd.DataFrame(r.cv_results_)
    top_parameters = top_parameters[['mean_test_score', 'std_test_score', 'params']].sort_values(by='mean_test_score', ascending=False)
    print top_parameters.head(16)

'''
   mean_test_score  std_test_score   params
5        -0.279929        0.059918   {u'epsilon': 0.25, u'C': 32, u'gamma': 0.0002} => 0.31931, avg712
4        -0.280964        0.061110   {u'epsilon': 0.24, u'C': 32, u'gamma': 0.0002} => 0.31901, avg717
3        -0.281964        0.062060   {u'epsilon': 0.23, u'C': 32, u'gamma': 0.0002} => 0.31879, avg721
2        -0.283140        0.063244   {u'epsilon': 0.22, u'C': 32, u'gamma': 0.0002} => 0.31888, avg726
1        -0.284553        0.064547   {u'epsilon': 0.21, u'C': 32, u'gamma': 0.0002} => 0.31892, avg729
0        -0.285949        0.065621   {u'epsilon': 0.20, u'C': 32, u'gamma': 0.0002} => 0.31891, avg733
0        -0.287327        0.066543   {u'epsilon': 0.19, u'C': 32, u'gamma': 0.0002} => 0.31899, avg738  
'''

# ----------------- Training ----------------- #
print "[INFO] Training..."
model = SVR(kernel='rbf', epsilon=0.23, C=32, gamma=0.0002, tol=0.001, cache_size=16384, verbose=2)
model.fit(x_train, y_train)
print "[INFO] Predicting..."
y_predict  = np.expm1(model.predict(x_test))
submission = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})
submission.to_csv('svr_output.csv', index=False)
print submission.head()
print "[INFO] SVR Average Price =", submission['price_doc'].mean()

# Plot Original, Training and Test Sets
ax4 = plt.subplot(312, sharex=ax1)
plt.hist(y_train, bins=200, color='b')
#plt.setp(ax4.get_xticklabels(), visible=False)
plt.title('Training Data Set')
plt.subplot(313, sharex=ax1)
plt.hist(np.log1p(y_predict), bins=200, color='b')
plt.title('Test Data Set Prediction')
OrigTrainValidSetFig.show()

plt.show()

