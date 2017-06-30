import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import datetime


EN_CROSSVALIDATION   = True
EN_IMPORTANCE        = True
DEFAULT_TRAIN_ROUNDS = 200

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
df.loc[(df.full_sq<2) & (df.life_sq<2), 'full_sq'] = np.nan
df.loc[(df.full_sq<10) & (df.life_sq>1), 'full_sq'] = df.life_sq
# Clean - Life Sq
df.loc[df.life_sq > df.full_sq*4, 'life_sq'] = df.life_sq/10
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



# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.title('Original Data Set')



# -------------------- Read Models -------------------- #
gunja_train = pd.read_csv('ensemble/gunja_train.csv')
louis_train = pd.read_csv('ensemble/louis_train.csv')
bruno_train = pd.read_csv('ensemble/bruno_train.csv')
svr_train   = pd.read_csv('ensemble/svr_train.csv')
df_train    = df

# Preprocessing
louis_train.loc[louis_train.price_doc<0, 'price_doc'] = -louis_train['price_doc']
gunja_train['price_doc'] = np.log1p(gunja_train['price_doc'])
louis_train['price_doc'] = np.log1p(louis_train['price_doc'])
bruno_train['price_doc'] = np.log1p(bruno_train['price_doc'])
svr_train['price_doc']   = np.log1p(svr_train['price_doc'])
df_train['price_doc']    = np.log1p(df_train['price_doc'])

# Feature Selection of Training Data Set
ColToSelect = ['id', 'full_sq', 'life_sq', 'kitch_sq', 'build_year', 'state', 'floor', 'max_floor', 'num_room']
df_train = df_train[ColToSelect+["price_doc"]]

# Prepare Training Data
gunja_train.rename(columns = {'price_doc':'gunja'}, inplace=True)
louis_train.rename(columns = {'price_doc':'louis'}, inplace=True)
bruno_train.rename(columns = {'price_doc':'bruno'}, inplace=True)
svr_train.rename(columns = {'price_doc':'svr'}, inplace=True)
df_train = pd.merge(df_train, gunja_train, on='id', how='left')
df_train = pd.merge(df_train, louis_train, on='id', how='left')
df_train = pd.merge(df_train, bruno_train, on='id', how='left')
df_train = pd.merge(df_train, svr_train,   on='id', how='left')
train_X  = df_train.drop(['id', 'price_doc'], axis=1)
train_y  = df_train['price_doc']


# Encoding
for c in train_X.columns:
    if train_X[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_X[c].values)) 
        train_X[c] = lbl.transform(list(train_X[c].values))
# Pack into DMatrix
dtrain = xgb.DMatrix(train_X, train_y)


# ----------------- Parameters ----------------- #
xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 1,
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
model      = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAIN_ROUNDS, 
                       evals=[(dtrain, 'train')], verbose_eval=10)




# -------------------- Prediction -------------------- #
gunja_test = pd.read_csv('ensemble/gunja_test.csv')
louis_test = pd.read_csv('ensemble/louis_test.csv')
bruno_test = pd.read_csv('ensemble/bruno_test.csv')
svr_test   = pd.read_csv('ensemble/svr_test.csv')
df_test    = test_df

# Preprocessing
gunja_test['price_doc'] = np.log1p(gunja_test['price_doc'])
louis_test['price_doc'] = np.log1p(louis_test['price_doc'])
bruno_test['price_doc'] = np.log1p(bruno_test['price_doc'])
svr_test['price_doc']   = np.log1p(svr_test['price_doc'])

# Feature Selection of Test Data Set
df_test = df_test[ColToSelect]

# Merge with 4 basic models
gunja_test.rename(columns = {'price_doc':'gunja'}, inplace=True)
louis_test.rename(columns = {'price_doc':'louis'}, inplace=True)
bruno_test.rename(columns = {'price_doc':'bruno'}, inplace=True)
svr_test.rename(columns = {'price_doc':'svr'}, inplace=True)
df_test = pd.merge(df_test, gunja_test, on='id', how='left')
df_test = pd.merge(df_test, louis_test, on='id', how='left')
df_test = pd.merge(df_test, bruno_test, on='id', how='left')
df_test = pd.merge(df_test, svr_test,   on='id', how='left')


# ----------------- Test Data ----------------- #
x_test  = df_test.drop(["id"], axis=1)
# Encoding        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
# Pack into DMatrix
dtest = xgb.DMatrix(x_test)


# Ensemble Prediction
print "[INFO] Predicting..."
y_predict  = model.predict(dtest)
y_predict  = np.expm1(y_predict)
submission = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})
submission.to_csv('submission.csv', index=False)
print submission.head()
print "[INFO] Stacking Average Price =", submission['price_doc'].mean()


# Plot Original, Training and Test Sets
ax4 = plt.subplot(312, sharex=ax1)
plt.hist(train_y, bins=200, color='b')
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

