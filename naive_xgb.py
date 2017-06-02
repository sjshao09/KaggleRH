import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb

# ----------------- Settings ----------------- #
EN_CROSSVALIDATION   = True
EN_IMPORTANCE        = True
DEFAULT_TRAIN_ROUNDS = 384




# ----------------- Read Data ----------------- #
df      = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
macro   = pd.read_csv('input/macro.csv')

# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.title('Original Data Set')

'''
# ----------------- Data Cleaning Essential ----------------- #
# Training Set
df = df[(df.full_sq>1)|(df.life_sq>1)]
df = df[(df.full_sq<500)]
df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==25943, 'max_floor'] = 17
df.loc[df.build_year>2050, 'build_year'] = np.nan
df.loc[(df.full_sq<10) & (df.life_sq>1), 'full_sq'] = df.life_sq
df.loc[df.life_sq>df.full_sq*2, 'life_sq'] = df.life_sq/10
# Test Set
test_df.loc[test_df.id==30938,  'full_sq'] = 37.8
test_df.loc[test_df.id==35857,  'full_sq'] = 42.07
test_df.loc[test_df.id==35108,  'full_sq'] = 40.3
test_df.loc[test_df.life_sq>test_df.full_sq*2, 'life_sq'] = test_df.life_sq/10
'''

'''
# ----------------- Data Cleaning Enhanced ----------------- #
# Data Cleaning - Training Set
# full_sq
# life_sq
df.loc[(df.life_sq<1)|(df.life_sq>df.full_sq), 'life_sq'] = 1
# kitch_sq
df.loc[df.kitch_sq<2, 'kitch_sq'] = 1
# build_year
df.loc[(df.build_year<1000), 'build_year'] = np.nan
# num_room
df.loc[df.full_sq<30, 'num_room'] = 1
df.loc[df.num_room<1, 'num_room'] = np.nan
df.loc[(df.num_room>9) & (df.full_sq<100), 'num_room'] = np.nan
# floor
df.loc[df.floor==0, 'floor'] = np.nan
# max_floor
df.loc[(df.max_floor==0)|(df.max_floor<df.floor), 'max_floor'] = np.nan

# Data Cleaning - Test Set
# life_sq
test_df.loc[test_df.life_sq<1, 'life_sq']  = np.nan
# kitch_sq
test_df.loc[(test_df.kitch_sq>test_df.full_sq*0.7)|(test_df.kitch_sq<1) , 'kitch_sq'] = 1
# build_year
test_df.loc[(test_df.build_year<1000) | (test_df.build_year>2050), 'build_year'] = np.nan
# num_room
test_df.loc[test_df.id==33648,  'num_room'] = 1
# max_floor
test_df.loc[test_df.max_floor<test_df.floor, 'max_floor'] = np.nan
'''


# ----------------- Training Data ----------------- #
y_train = df["price_doc"] * 0.968 + 10
x_train = df.drop(["id", "timestamp", "price_doc"], axis=1)
# Encoding
for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
# Pack into DMatrix
dtrain  = xgb.DMatrix(x_train, y_train)


# ----------------- Test Data ----------------- #
x_test  = test_df.drop(["id", "timestamp"], axis=1)
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
model      = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAIN_ROUNDS, 
                       evals=[(dtrain, 'train')], verbose_eval=10)
y_predict  = model.predict(dtest)
submission = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})
submission.to_csv('submission.csv', index=False)
print submission.head()

# Plot Original, Training and Test Sets
ax4 = plt.subplot(312, sharex=ax1)
plt.hist(np.log1p(y_train), bins=200, color='b')
#plt.setp(ax4.get_xticklabels(), visible=False)
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

