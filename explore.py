import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Settings
EN_CROSSVALIDATION = False
EN_TRAINING        = True
EN_IMPORTANCE      = False
EN_PREDICTION      = True
EN_MARCODATA       = False
EN_DOWNSAMPLING    = True
NUM_TRAIN_ROUNDS   = 1000
RANDOM_SEED        = 1

# Read Training Data Set and Macro Data Set
df       = pd.read_csv('input/train.csv')
df_macro = pd.read_csv('input/macro.csv')
if EN_MARCODATA:
    df = pd.merge(df, df_macro, on='timestamp', how='left')

# Transform product_type into numbers: Investment=0, OwnerOccupier=1
ProdTypeEncoder = LabelEncoder()
ProdTypeEncoder.fit(df['product_type'])
df['product_type'] = ProdTypeEncoder.transform(df['product_type'])


# Drop Error Row (id=10092, state=33, buildyear=20052009)
df = df[df.id != 10092]

# Object Columns
ObjColName_Train = ['timestamp', 'sub_area', 'culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 'ecology']
ObjColName_Macro = ['child_on_acc_pre_school', 'modern_education_share', 'old_education_build_share']
ObjCol = df[ObjColName_Train]
#print ObjCol.describe()
#print ObjCol.dtypes.value_counts()
# Drop Non-Numerical Features and id
if EN_MARCODATA:
    ColToDrop = ObjColName_Train + ObjColName_Macro + ['id']
else:
    ColToDrop = ObjColName_Train + ['id']
df = df.drop(ColToDrop, axis=1)
# Fill Missing Values
#df.fillna(df.median(axis=0), inplace=True)


# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.title('Original Data Set')



# Down Sampling
if EN_DOWNSAMPLING:
    df_1m = df[ (df.price_doc<=1000000) & (df.product_type==0) ]
    df    = df.drop(df_1m.index)
    df_1m = df_1m.sample(frac=0.1, replace=False, random_state=RANDOM_SEED)

    df_2m = df[ (df.price_doc==2000000) & (df.product_type==0) ]
    df    = df.drop(df_2m.index)
    df_2m = df_2m.sample(frac=0.8, replace=False, random_state=RANDOM_SEED)

    df_3m = df[ (df.price_doc==3000000) & (df.product_type==0) ]
    df    = df.drop(df_3m.index)
    df_3m = df_3m.sample(frac=0.5, replace=False, random_state=RANDOM_SEED)

    df    = pd.concat([df, df_1m, df_2m, df_3m])




'''
print df_macro.describe()
MissCount = df_macro.isnull().sum().sort_values(ascending=False).head(40) / len(df_macro) * 100
fig2 = plt.figure(figsize=(8, 12))
plt.barh(np.arange(len(MissCount)), MissCount)
plt.yticks(np.arange(len(MissCount))+0.5, MissCount.index, rotation='horizontal')
plt.title('Percentage of Missing Data')
plt.tight_layout()
fig2.show()
'''



# Separate Training Set and Validation Set
df_valid = df.sample(frac=0.1, random_state=RANDOM_SEED)
df_train = df.drop(df_valid.index)
print "[INFO] Trimmed Original Data Set Shape:", df.shape
print "[INFO]         Training Data Set Shape:", df_train.shape
print "[INFO]       Validation Data Set Shape:", df_valid.shape


# Plot Original Set, Train Set and Validation Set
ax2 = plt.subplot(312, sharex=ax1)
plt.hist(np.log1p(df_train['price_doc'].values), bins=200, color='b')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.title('Training Data Set (90%)')
plt.subplot(313, sharex=ax1)
plt.hist(np.log1p(df_valid['price_doc'].values), bins=200, color='b')
plt.title('Validation Data Set (10%)')
OrigTrainValidSetFig.show()



'''
# Top 40 features with most missing data
MissCount = df.isnull().sum().sort_values(ascending=False).head(40) / len(df) * 100
fig2 = plt.figure(figsize=(8, 12))
plt.barh(np.arange(len(MissCount)), MissCount)
plt.yticks(np.arange(len(MissCount))+0.5, MissCount.index, rotation='horizontal')
plt.title('Percentage of Missing Data')
plt.tight_layout()
fig2.show()
'''



# Prepare Training Data, Fill Missing Values with median
low_y_cut  = 1e2
high_y_cut = 1e10
# Training Set
df_train.fillna(df_train.median(axis=0), inplace=True)
y_range_train = ((df_train['price_doc'] > low_y_cut) & (df_train['price_doc'] < high_y_cut))
train_X = df_train.loc[y_range_train, :]
train_X = train_X.drop('price_doc', axis=1)
train_y = np.log1p(df_train.loc[y_range_train, 'price_doc'].values.reshape(-1, 1))
dtrain = xgb.DMatrix(train_X, train_y)
# Validation Set
df_valid.fillna(df_valid.median(axis=0), inplace=True)
#valid_X = df_valid.iloc[:,:-1]
#valid_y = np.log1p(df_valid['price_doc'].values.reshape(-1, 1))
y_range_valid = ((df_valid['price_doc'] > low_y_cut) & (df_valid['price_doc'] < high_y_cut))
valid_X = df_valid.loc[y_range_valid, :]
valid_X = valid_X.drop('price_doc', axis=1)
valid_y = np.log1p(df_valid.loc[y_range_valid, 'price_doc'].values.reshape(-1, 1))
dvalid = xgb.DMatrix(valid_X, valid_y)


# xgboost cross validation - for parameter selection
xgb_params = {
    'learning_rate': 0.05,
    'max_depth': 4,
    'gamma': 0,
    'sub_sample': 0.7,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'seed': 0,
    'nthread': 6
}
if EN_CROSSVALIDATION:
    print "[INFO] Running Cross-Validation..."
    xgb.cv(xgb_params, dtrain, num_boost_round=NUM_TRAIN_ROUNDS, nfold=5, shuffle=True,
           metrics={'rmse'}, seed=0, early_stopping_rounds=20,
           callbacks=[xgb.callback.print_evaluation(show_stdv=True)])


# xgboost training
if EN_TRAINING:
    print "[INFO] Training..."
    model = xgb.train(xgb_params, dtrain, num_boost_round=NUM_TRAIN_ROUNDS,
                  early_stopping_rounds=20,
                  evals=[(dtrain, 'training'),(dvalid, 'validation')], verbose_eval=1,
                  callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
    train_y_hat = model.predict(dtrain)
    rmsle_train = np.sqrt(mean_squared_error(train_y, train_y_hat))
    valid_y_hat = model.predict(dvalid)
    rmsle_valid = np.sqrt(mean_squared_error(valid_y, valid_y_hat))

    print "[INFO] RMSLE   training set =", rmsle_train
    print "[INFO] RMSLE validation set =", rmsle_valid

    if EN_IMPORTANCE:
        # Plot Feature Importance
        fig3 = plt.figure(figsize=(7,30))
        xgb.plot_importance(model, ax=fig3.add_subplot(111))
        plt.tight_layout()

    if EN_PREDICTION:
        # Make Prediction
        print "[INFO] Making Prediction..."
        test_df  = pd.read_csv('input/test.csv')
        if EN_MARCODATA:
            test_df = pd.merge(test_df, df_macro, on='timestamp', how='left')
        test_df.fillna(test_df.median(axis=0), inplace=True)
        # Handle NA in product_type and apply encoding
        test_df['product_type'].fillna(test_df['product_type'].mode().iloc[0], inplace=True) 
        test_df['product_type'] = ProdTypeEncoder.transform(test_df['product_type'])
        # Drop Columns
        test_X = test_df.drop(ColToDrop, axis=1)
        test_y_predict = np.exp(model.predict(xgb.DMatrix(test_X)))-1
        submission = pd.DataFrame(index=test_df['id'], data={'price_doc':test_y_predict})
        print submission.head()
        submission.to_csv('submission.csv', header=True)
        # Plot Training, Validation and Test Sets
        TrainValidTestSetFig = plt.figure()
        ax4 = plt.subplot(311, sharex=ax1)
        plt.hist(train_y, bins=200, color='b')
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.title('Training Data Set')
        ax5 = plt.subplot(312, sharex=ax4)
        plt.hist(valid_y, bins=200, color='b')
        plt.setp(ax5.get_xticklabels(), visible=False)
        plt.title('Validation Data Set')
        plt.subplot(313, sharex=ax4)
        plt.hist(np.log1p(test_y_predict), bins=200, color='b')
        plt.title('Test Data Set Prediction')
        TrainValidTestSetFig.show()



# End of Script - display figures
#plt.show()
print "Finished"


