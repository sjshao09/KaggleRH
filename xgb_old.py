import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pyproj import Proj, transform, Geod

# Settings
EN_IMPUTATION      = True
EN_CROSSVALIDATION = True
EN_TRAINING        = True
EN_IMPORTANCE      = False
EN_PREDICTION      = True
EN_MARCODATA       = False
EN_TRAINALL        = False
NUM_TRAIN_ROUNDS   = 1000
RANDOM_SEED_SPLIT  = 1
RANDOM_SEED_TRAIN  = 0

# Read Data
df       = pd.read_csv('input/train.csv')
df_macro = pd.read_csv('input/macro.csv')
shp      = gpd.read_file('input/moscow_adm.shp')


# Select and Merge Marco Features
MacroCol = ['timestamp', 'oil_urals', 'gdp_quart_growth', 'cpi', 'usdrub', 'salary_growth', 'unemployment', 'mortgage_rate', 'deposits_rate', 'rent_price_3room_eco', 'rent_price_3room_bus']
df_macro = df_macro[MacroCol]
if EN_MARCODATA:
    df = pd.merge(df, df_macro, on='timestamp', how='left')


# Merge Location Data - Distance from Kremlin
shp['centroid'] = shp.centroid
shp['long'] = shp['centroid'].apply(lambda p: p.x)
shp['lat']  = shp['centroid'].apply(lambda p: p.y)
shp['km_long'] = 37.617664
shp['km_lat']  = 55.752121
angle1, angle2, dist1 = Geod(ellps='WGS84').inv(shp['long'].values, shp['lat'].values, shp['km_long'].values, shp['km_lat'].values)
shp['distance_from_kremlin'] = dist1/1000
df_location = pd.DataFrame({'sub_area':shp['RAION'], 'district':shp['OKRUGS'], 'distance_from_kremlin':shp['distance_from_kremlin'] })
df = pd.merge(df, df_location, on='sub_area', how='left')


# Data Cleaning
# full_sq
df = df[(df.full_sq>1)|(df.life_sq>1)]
df = df[(df.full_sq<350)]
df.loc[(df.full_sq<2) & (df.life_sq>1), 'full_sq'] = df.life_sq
# life_sq
df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.life_sq>df.full_sq*2, 'life_sq'] = df.life_sq/10
df.loc[(df.life_sq<1)|(df.life_sq>df.full_sq), 'life_sq'] = 1
# kitch_sq
df.loc[df.kitch_sq<2, 'kitch_sq'] = 1
# build_year
df.loc[(df.build_year<1000)|(df.build_year>2050), 'build_year'] = np.nan
df.loc[df.id==10092, 'build_year'] = 2007
# state
df.loc[df.id==10092, 'state'] = 3
# num_room
df.loc[df.full_sq<30, 'num_room'] = 1
df.loc[df.num_room<1, 'num_room'] = np.nan
df.loc[(df.num_room>9) & (df.full_sq<100), 'num_room'] = np.nan
# floor
df.loc[df.floor==0, 'floor'] = np.nan
# max_floor
df.loc[df.id==25943, 'max_floor'] = 17
df.loc[(df.max_floor==0)|(df.max_floor<df.floor), 'max_floor'] = np.nan

# Imputing Values
if EN_IMPUTATION:
    # Imputing Missing Values
    # life_sq
    df.loc[df.life_sq<2, 'life_sq'] = np.nan
    df['life_sq'].fillna(np.maximum(df['full_sq']*0.732-4.241,1), inplace=True)
    # kitch_sq
    df.loc[df.kitch_sq<2, 'kitch_sq'] = np.nan
    df['kitch_sq'].fillna(np.maximum(df['kitch_sq']*0.078+4.040,1), inplace=True)
    # Fill the rest of missing values with median
    df.fillna(df.median(axis=0), inplace=True)

# Plot Original Data Set
OrigTrainValidSetFig = plt.figure()
ax1 = plt.subplot(311)
plt.hist(np.log1p(df['price_doc'].values), bins=200, color='b')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.title('Original Data Set')

# Down Sampling
df    = df[df.timestamp<'2015-01-01']
df_1m = df[ (df.price_doc<=1000000) & (df.product_type=="Investment") ]
df    = df.drop(df_1m.index)
df_1m = df_1m.sample(frac=0.1, replace=False, random_state=RANDOM_SEED_SPLIT)
df_2m = df[ (df.price_doc==2000000) & (df.product_type=="Investment") ]
df    = df.drop(df_2m.index)
df_2m = df_2m.sample(frac=0.1, replace=False, random_state=RANDOM_SEED_SPLIT)
df_3m = df[ (df.price_doc==3000000) & (df.product_type=="Investment") ]
df    = df.drop(df_3m.index)
df_3m = df_3m.sample(frac=0.1, replace=False, random_state=RANDOM_SEED_SPLIT)
df    = pd.concat([df, df_1m, df_2m, df_3m])


# Object Columns
ObjColName_Train = ['timestamp', 'culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 'ecology']
ObjColName_Macro = ['child_on_acc_pre_school', 'modern_education_share', 'old_education_build_share']
ObjCol = df[ObjColName_Train]
# Drop Non-Numerical Features and id
if EN_MARCODATA:
    ColToDrop = ObjColName_Train + ObjColName_Macro + ['id']
else:
    ColToDrop = ObjColName_Train + ['id']
df = df.drop(ColToDrop, axis=1)


# object Column Encoding
# Transform product_type into numbers: Investment=0, OwnerOccupier=1
df['product_type'].fillna(df['product_type'].mode().iloc[0], inplace=True)
ProdTypeEncoder = LabelEncoder()
ProdTypeEncoder.fit(df['product_type'])
df['product_type'] = ProdTypeEncoder.transform(df['product_type'])
# Transform district into numbers
DistrictEncoder = LabelEncoder()
DistrictEncoder.fit(df['district'])
df['district'] = DistrictEncoder.transform(df['district'])
# Transform sub_area into numbers
SubAreaEncoder = LabelEncoder()
SubAreaEncoder.fit(df['sub_area'])
df['sub_area'] = SubAreaEncoder.transform(df['sub_area'])


# Prepare Training and Validation Sets
high_y_cut = 50000000
if EN_TRAINALL:
    # Training Set
    train_X = df[df['price_doc'] < high_y_cut]
    train_y = np.log1p(train_X['price_doc'].values.reshape(-1, 1))
    train_X = train_X.drop('price_doc', axis=1)
    dtrain  = xgb.DMatrix(train_X, train_y)
else:
    # Separate Training Set and Validation Set
    df_valid = df.sample(frac=0.1, random_state=RANDOM_SEED_SPLIT)
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

    # Training Set
    train_X = df_train[(df_train['price_doc'] < high_y_cut)]
    train_y = np.log1p(train_X['price_doc'].values.reshape(-1, 1))
    train_X = train_X.drop('price_doc', axis=1)
    dtrain  = xgb.DMatrix(train_X, train_y)
    # Validation Set
    valid_X = df_valid[(df_valid['price_doc'] < high_y_cut)]
    valid_y = np.log1p(valid_X['price_doc'].values.reshape(-1, 1))
    valid_X = valid_X.drop('price_doc', axis=1)
    dvalid  = xgb.DMatrix(valid_X, valid_y)


# xgboost cross validation - for parameter selection
xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'gamma': 0,
    'sub_sample': 0.7,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'seed': RANDOM_SEED_TRAIN,
    'nthread': 6
}
if EN_CROSSVALIDATION or (EN_TRAINING and EN_TRAINALL):
    print "[INFO] Running Cross-Validation..."
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=5, shuffle=True,
           metrics={'rmse'}, seed=RANDOM_SEED_SPLIT, early_stopping_rounds=10, verbose_eval=10)
    OptTrainRounds = len(cv_output)
    print "[INFO] Optimal Training Rounds =", OptTrainRounds

# xgboost training
if EN_TRAINING:
    if EN_TRAINALL:
        print "[INFO] Training for", OptTrainRounds,"rounds..."
        model = xgb.train(xgb_params, dtrain, num_boost_round=OptTrainRounds,
                early_stopping_rounds=10, evals=[(dtrain, 'train')], verbose_eval=10)        
    else:
        print "[INFO] Training..."
        model = xgb.train(xgb_params, dtrain, num_boost_round=NUM_TRAIN_ROUNDS,
                early_stopping_rounds=10, evals=[(dtrain, 'train'),(dvalid, 'validation')], verbose_eval=10)
        train_y_hat = model.predict(dtrain)
        rmsle_train = np.sqrt(mean_squared_error(train_y, train_y_hat))
        valid_y_hat = model.predict(dvalid)
        rmsle_valid = np.sqrt(mean_squared_error(valid_y, valid_y_hat))
        print "[INFO] RMSLE   training set =", rmsle_train
        print "[INFO] RMSLE validation set =", rmsle_valid

    if EN_PREDICTION:
        # Make Prediction
        print "[INFO] Making Prediction..."
        # Read Test Data
        test_df = pd.read_csv('input/test.csv')
        test_df = pd.merge(test_df, df_location, on='sub_area', how='left')

        # Merge MarcoData
        if EN_MARCODATA:
            test_df = pd.merge(test_df, df_macro, on='timestamp', how='left')
        # Data Cleaning
        # full_sq
        test_df.loc[test_df.id==30938,  'full_sq'] = 37.8
	test_df.loc[test_df.id==35857,  'full_sq'] = 42.07
        test_df.loc[test_df.id==35108,  'full_sq'] = 40.3
	# life_sq
        test_df.loc[test_df.life_sq>test_df.full_sq*2, 'life_sq'] = test_df.life_sq/10
        test_df.loc[test_df.life_sq<1, 'life_sq']  = np.nan
        # kitch_sq
        test_df.loc[(test_df.kitch_sq>test_df.full_sq*0.7)|(test_df.kitch_sq<1) , 'kitch_sq'] = 1
        # build_year
        test_df.loc[(test_df.build_year<1000) | (test_df.build_year>2050), 'build_year'] = np.nan
        # num_room
        test_df.loc[test_df.id==33648,  'num_room'] = 1
        # max_floor
        test_df.loc[test_df.max_floor<test_df.floor, 'max_floor'] = np.nan

        # object Column Encoding
        # product_type
        test_df['product_type'].fillna(test_df['product_type'].mode().iloc[0], inplace=True) 
        test_df['product_type'] = ProdTypeEncoder.transform(test_df['product_type'])
        # district
        test_df['district'] = DistrictEncoder.transform(test_df['district'])
        # sub_area
        test_df['sub_area'] = SubAreaEncoder.transform(test_df['sub_area'])

        if EN_IMPUTATION:
            # Imputing Missing Values
            # life_sq
            test_df.loc[test_df.life_sq<2, 'life_sq'] = np.nan
            test_df['life_sq'].fillna(np.maximum(test_df['full_sq']*0.732-4.241,1), inplace=True)
            # kitch_sq
            test_df.loc[test_df.kitch_sq<2, 'kitch_sq'] = np.nan
            test_df['kitch_sq'].fillna(np.maximum(test_df['kitch_sq']*0.078+4.040,1), inplace=True)
            # Fill the rest of missing values with median
            test_df.fillna(test_df.median(axis=0), inplace=True)

        # Drop Columns
        test_X = test_df.drop(ColToDrop, axis=1)
        test_y_predict = np.exp(model.predict(xgb.DMatrix(test_X)))-1
        submission = pd.DataFrame(index=test_df['id'], data={'price_doc':test_y_predict})
        print submission.head()
        submission.to_csv('submission.csv', header=True)

    if EN_TRAINALL:
        # Plot Original, Training and Test Sets
        ax4 = plt.subplot(312, sharex=ax1)
        plt.hist(train_y, bins=200, color='b')
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.title('Training Data Set')
        plt.subplot(313, sharex=ax1)
        plt.hist(np.log1p(test_y_predict), bins=200, color='b')
        plt.title('Test Data Set Prediction')
        OrigTrainValidSetFig.show()
    else:
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

    if EN_IMPORTANCE:
        # Plot Feature Importance
        ImportanceFig = plt.figure(figsize=(7,30))
        xgb.plot_importance(model, ax=ImportanceFig.add_subplot(111))
        plt.tight_layout()
        ImportanceFig.show()

# End of Script - display figures
plt.show()
print "Finished"


