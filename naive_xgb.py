import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb

# ----------------- Settings ----------------- #
EN_CROSSVALIDATION   = False
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
print "[INFO] Average Price =", submission['price_doc'].mean()

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

