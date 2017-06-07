import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV


# ----------------- Settings ----------------- #
EN_CROSSVALIDATION   = False
EN_IMPORTANCE        = True


# ----------------- Read Data ----------------- #
df      = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
macro   = pd.read_csv('input/macro.csv')


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

# ----------------- Fill by median ----------------- #
df.fillna(df.median(axis=0), inplace=True)
test_df['product_type'].fillna(test_df['product_type'].mode().iloc[0], inplace=True)
test_df.fillna(df.median(axis=0), inplace=True)

'''
# ----------------- Remove Extreme Data ----------------- #
RANDOM_SEED_SPLIT = 1
df_1m = df[ (df.price_doc<=1000000) & (df.product_type=="Investment") ]
df    = df.drop(df_1m.index)
df_1m = df_1m.sample(frac=0.1, replace=False, random_state=RANDOM_SEED_SPLIT)

df_2m = df[ (df.price_doc==2000000) & (df.product_type=="Investment") ]
df    = df.drop(df_2m.index)
df_2m = df_2m.sample(frac=0.33, replace=False, random_state=RANDOM_SEED_SPLIT)

df_3m = df[ (df.price_doc==3000000) & (df.product_type=="Investment") ]
df    = df.drop(df_3m.index)
df_3m = df_3m.sample(frac=0.5, replace=False, random_state=RANDOM_SEED_SPLIT)

df    = pd.concat([df, df_1m, df_2m, df_3m])
'''

# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.title('Original Data Set')



# ----------------- Training Data ----------------- #
y_train = df['price_doc']
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


# ----------------- Parameters ----------------- #


# ----------------- Cross Validation ----------------- #
if EN_CROSSVALIDATION:
    print "[INFO] Cross Validation..."
    etr = ExtraTreesRegressor(random_state=0, bootstrap=True, oob_score=True, n_jobs=1, verbose=0)
    param_dist = {"n_estimators": [256, 512, 768],
                  "max_depth": [6, 7, 8, 9, 10],
                  "max_features": [100, 200, 280],
                  "min_samples_split": [2, 4, 6],
                  "min_samples_leaf": [6, 8, 10],
                  "criterion": ["mse"]
                 }
    r = RandomizedSearchCV(etr, param_distributions=param_dist, cv=4, n_iter=256, 
                           iid=False, n_jobs=6, verbose=2)
    r.fit(x_train, y_train)
    print ('Best score: {}'.format(r.best_score_))
    print ('Best parameters: {}'.format(r.best_params_))


# ----------------- Training ----------------- #
print "[INFO] Training..."
model = ExtraTreesRegressor(random_state=0, bootstrap=True, oob_score=True, n_jobs=6,
                            verbose=1, max_depth=10, n_estimators=256, max_features=280, 
                            min_samples_split=6, min_samples_leaf=6)

model.fit(x_train, y_train)
print('OOB score: {:6f}'.format(model.oob_score_))
print "[INFO] Predicting..."
y_predict  = model.predict(x_test)
submission = pd.DataFrame({'id': test_df.id, 'price_doc': y_predict})
submission.to_csv('submission.csv', index=False)
print submission.head()

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
    importance_df = pd.DataFrame({'feature':list(x_train), 'fscore':model.feature_importances_})
    importance_df = importance_df.nlargest(50, 'fscore')
    importance_df.sort_values(by='fscore', inplace=True)
    importance_df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(8, 13))
    plt.title('Extra Trees Regressor Important Features Top 50')
    plt.xlabel('fscore')
    plt.tight_layout()

plt.show()

