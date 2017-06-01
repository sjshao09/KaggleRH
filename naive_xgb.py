import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb

# ----------------- Read Data ----------------- #
train = pd.read_csv('input/train.csv')
test  = pd.read_csv('input/test.csv')
macro = pd.read_csv('input/macro.csv')


# ----------------- Training Data ----------------- #
y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
# Encoding
for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))



# ----------------- Test Data ----------------- #
x_test  = test.drop(["id", "timestamp"], axis=1)
# Encoding        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))


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
print "[INFO] Cross Validation..."
dtrain    = xgb.DMatrix(x_train, y_train)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                   verbose_eval=10, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
OptTrainRounds = len(cv_output)
print "[INFO] Optimal Training Rounds =", OptTrainRounds

# ----------------- Training ----------------- #
print "[INFO] Training for", OptTrainRounds,"rounds..."
model   = xgb.train(xgb_params, dtrain, num_boost_round=OptTrainRounds, 
                    evals=[(dtrain, 'train')], verbose_eval=10)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

# ----------------- Prediction ----------------- #
y_predict  = model.predict(xgb.DMatrix(x_test))
submission = pd.DataFrame({'id': test.id, 'price_doc': y_predict})
submission.to_csv('submission.csv', index=False)
print submission.head()

plt.show()

