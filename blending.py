import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import datetime


# -------------------- Read Models -------------------- #
gunja_train = pd.read_csv('ensemble/gunja_train.csv')
louis_train = pd.read_csv('ensemble/louis_train.csv')
bruno_train = pd.read_csv('ensemble/bruno_train.csv')
svr_train   = pd.read_csv('ensemble/svr_train.csv')
df_train    = pd.read_csv('input/train.csv')

# Preprocessing
louis_train.loc[louis_train.price_doc<0, 'price_doc'] = -louis_train['price_doc']
gunja_train['price_doc'] = np.log1p(gunja_train['price_doc'])
louis_train['price_doc'] = np.log1p(louis_train['price_doc'])
bruno_train['price_doc'] = np.log1p(bruno_train['price_doc'])
svr_train['price_doc']   = np.log1p(svr_train['price_doc'])
df_train['price_doc']    = np.log1p(df_train['price_doc'])

# Feature Selection of Training Data Set
ColToSelect = ["id"]
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

#print train_X.head()

# Train Ensemble Model
'''
# ----------------- Cross Validation ----------------- #
EN_CROSSVALIDATION = False
if EN_CROSSVALIDATION:
    print "[INFO] Cross Validation..."
    model = ElasticNet(positive=True)
    param_dist = {#'alpha': [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
	          'alpha': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                  'l1_ratio': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
                 }
    r = RandomizedSearchCV(model, param_distributions=param_dist, scoring='neg_mean_squared_error', 
                           cv=4, n_iter=128, iid=False, verbose=1, n_jobs=6)
    r.fit(train_X, train_y)
    print ('Best score: {}'.format(r.best_score_))
    print ('Best parameters: {}'.format(r.best_params_))
    top_parameters = pd.DataFrame(r.cv_results_)
    top_parameters = top_parameters[['mean_test_score', 'std_test_score', 'params']].sort_values(by='mean_test_score', ascending=False)
    print top_parameters.head(16)


#model = ElasticNet(alpha=1e-6, l1_ratio=0.85, positive=True)
model = ElasticNet(alpha=0.0001, l1_ratio=0.85, positive=True)
model.fit(train_X, train_y)
print("Intercept of the model: {}".format(model.intercept_))
print("Coefficients of the model: {}".format(model.coef_))
'''

EN_CROSSVALIDATION = False
if EN_CROSSVALIDATION:
    NUM_ROUNDS = 50000
    RandNumbers = np.random.rand(NUM_ROUNDS, 3)
    RandNumbers.sort(axis=1)
    records = list()
    for i in range(0, NUM_ROUNDS):
        curRandNum = RandNumbers[i]
        w1 = curRandNum[0]
        w2 = curRandNum[1] - curRandNum[0]
        w3 = curRandNum[2] - curRandNum[1]
        w4 = 1 - curRandNum[2]
        y_predict = w1*train_X['louis'] + w2*train_X['bruno'] + w3*train_X['gunja'] + w4*train_X['svr']
        mse = mean_squared_error(train_y, y_predict)
        cur_record = [mse, w1, w2, w3, w4]
        records.append(cur_record)

    records_df = pd.DataFrame(records, columns=['mse', 'louis', 'bruno', 'gunja', 'svr'])
    records_df = records_df.nsmallest(16, 'mse')
    records_df.sort_values(by='mse', inplace=True)
    print records_df.head(16)


'''
            mse     louis     bruno     gunja       svr
23346  0.180893  0.016382  0.979585  0.002216  0.001817
4492   0.180905  0.001443  0.959506  0.035782  0.003269
5280   0.180938  0.009151  0.955015  0.032792  0.003041
16546  0.180967  0.000891  0.970465  0.020926  0.007718
7207   0.181027  0.016572  0.940679  0.038173  0.004577
42047  0.181031  0.007393  0.946985  0.038436  0.007186
7084   0.181055  0.014257  0.950384  0.027824  0.007536
2046   0.181079  0.032202  0.950518  0.011913  0.005367
2131   0.181123  0.033971  0.909522  0.053994  0.002513
37904  0.181124  0.001421  0.897453  0.094971  0.006155
28489  0.181128  0.006079  0.903741  0.083585  0.006596
40257  0.181135  0.025190  0.901505  0.069741  0.003564
11772  0.181142  0.019142  0.942593  0.028042  0.010223
45355  0.181145  0.045578  0.935963  0.013832  0.004627
26322  0.181166  0.005054  0.942723  0.038213  0.014010
27759  0.181169  0.008819  0.890934  0.093891  0.006356
33761  0.181177  0.005008  0.921176  0.062033  0.011783
25148  0.181211  0.040720  0.891842  0.063915  0.003523
20848  0.181216  0.038290  0.945253  0.005851  0.010606
29442  0.181221  0.026844  0.874382  0.095144  0.003630
5865   0.181234  0.051649  0.923558  0.018156  0.006636
'''

# -------------------- Prediction -------------------- #
gunja_test = pd.read_csv('ensemble/gunja_test.csv')
louis_test = pd.read_csv('ensemble/louis_test.csv')
bruno_test = pd.read_csv('ensemble/bruno_test.csv')
svr_test   = pd.read_csv('ensemble/svr_test.csv')
andy_test  = pd.read_csv('ensemble/andy_test.csv')
df_test    = pd.read_csv('input/test.csv')

# Preprocessing
gunja_test['price_doc'] = np.log1p(gunja_test['price_doc'])
louis_test['price_doc'] = np.log1p(louis_test['price_doc'])
bruno_test['price_doc'] = np.log1p(bruno_test['price_doc'])
andy_test['price_doc'] = np.log1p(andy_test['price_doc'])
svr_test['price_doc']   = np.log1p(svr_test['price_doc'])

# Feature Selection of Test Data Set
df_test = df_test[ColToSelect]

# Merge with 4 basic models
gunja_test.rename(columns = {'price_doc':'gunja'}, inplace=True)
louis_test.rename(columns = {'price_doc':'louis'}, inplace=True)
bruno_test.rename(columns = {'price_doc':'bruno'}, inplace=True)
andy_test.rename(columns = {'price_doc':'andy'}, inplace=True)
svr_test.rename(columns = {'price_doc':'svr'}, inplace=True)
df_test = pd.merge(df_test, gunja_test, on='id', how='left')
df_test = pd.merge(df_test, louis_test, on='id', how='left')
df_test = pd.merge(df_test, bruno_test, on='id', how='left')
df_test = pd.merge(df_test, andy_test, on='id', how='left')
df_test = pd.merge(df_test, svr_test,   on='id', how='left')

# Ensemble Prediction
print "[INFO] Predicting..."
test_X = df_test.drop(['id'], axis=1)
#y_predict = model.predict(test_X)
#y_predict = 0.2*df_test['louis'] + 0.2*df_test['bruno'] + 0.2*df_test['gunja'] + 0.2*df_test['svr'] + 0.2*df_test['andy']
y_predict = 0.25*df_test['louis'] + 0.25*df_test['bruno'] + 0.25*df_test['gunja'] + 0.25*df_test['svr']
y_predict = np.expm1(y_predict)
submission = pd.DataFrame({'id': df_test.id, 'price_doc': y_predict})
submission.to_csv('submission.csv', index=False)
print submission.head()
print "[INFO] Ensemble Average Price =", submission['price_doc'].mean()



# -------------------- PROBABILISTIC IMPROVEMENTS -------------------- #


# APPLY PROBABILISTIC IMPROVEMENTS
from scipy.stats import norm
# Parameters
prediction_stderr = 0.006  #  assumed standard error of predictions
                          #  (smaller values make output closer to input)
train_test_logmean_diff = 0.1  # assumed shift used to adjust frequencies for time trend
probthresh = 90  # minimum probability*frequency to use new price instead of just rounding
rounder = 2  # number of places left of decimal point to zero

preds = submission
train = pd.read_csv('input/train.csv')
test  = pd.read_csv('input/test.csv')

# Select investment sales from training set and generate frequency distribution
invest = train[train.product_type=="Investment"]
freqs = invest.price_doc.value_counts().sort_index()

# Select investment sales from test set predictions
test_invest_ids = test[test.product_type=="Investment"]["id"]
invest_preds = pd.DataFrame(test_invest_ids).merge(preds, on="id")

# Express X-axis of training set frequency distribution as logarithms, 
#    and save standard deviation to help adjust frequencies for time trend.
lnp = np.log(invest.price_doc)
stderr = lnp.std()
lfreqs = lnp.value_counts().sort_index()

# Adjust frequencies for time trend
lnp_diff = train_test_logmean_diff
lnp_mean = lnp.mean()
lnp_newmean = lnp_mean + lnp_diff

def norm_diff(value):
    return norm.pdf((value-lnp_diff)/stderr) / norm.pdf(value/stderr)

newfreqs = lfreqs * (pd.Series(lfreqs.index.values-lnp_newmean).apply(norm_diff).values)

# Logs of model-predicted prices
lnpred = np.log(invest_preds.price_doc)

# Create assumed probability distributions
stderr = prediction_stderr
mat =(np.array(newfreqs.index.values)[:,np.newaxis] - np.array(lnpred)[np.newaxis,:])/stderr
modelprobs = norm.pdf(mat)

# Multiply by frequency distribution.
freqprobs = pd.DataFrame( np.multiply( np.transpose(modelprobs), newfreqs.values ) )
freqprobs.index = invest_preds.price_doc.values
freqprobs.columns = freqs.index.values.tolist()

# Find mode for each case.
prices = freqprobs.idxmax(axis=1)

# Apply threshold to exclude low-confidence cases from recoding
priceprobs = freqprobs.max(axis=1)
mask = priceprobs<probthresh
prices[mask] = np.round(prices[mask].index,-rounder)

# Data frame with new predicitons
newpricedf = pd.DataFrame( {"id":test_invest_ids.values, "price_doc":prices} )

# Merge these new predictions (for just investment properties) back into the full prediction set.
newpreds = preds.merge(newpricedf, on="id", how="left", suffixes=("_old",""))
newpreds.loc[newpreds.price_doc.isnull(),"price_doc"] = newpreds.price_doc_old
newpreds.drop("price_doc_old",axis=1,inplace=True)
print newpreds.head()
newpreds.to_csv('blending_with_prob_adjust.csv', index=False)
print "[INFO] Adjusted Average Price =", newpreds['price_doc'].mean()

