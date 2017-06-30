import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import ggplot

# ----------------- Settings ----------------- #
EN_CROSSVALIDATION   = True
EN_IMPORTANCE        = True
DEFAULT_TRAIN_ROUNDS = 384

# ----------------- Read Data ----------------- #
df      = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
macro   = pd.read_csv('input/macro.csv')

# Select and Merge Marco Features
MacroCol = ['timestamp', 'oil_urals', 'gdp_quart_growth', 'cpi', 'usdrub', 'mortgage_rate', 'balance_trade']
df_macro = macro[MacroCol]
df_macro["oilrub"] = df_macro['oil_urals'] * df_macro["usdrub"]
df = pd.merge(df, df_macro, on='timestamp', how='left')


# ----------------- Clean Data ----------------- # 
# Training Set
df = df[(df.full_sq>1)|(df.life_sq>1)]
df.loc[df.id==18344, 'full_sq'] = 63
df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==25943, 'max_floor'] = 17
df.loc[df.build_year>2050, 'build_year'] = np.nan
df.loc[(df.full_sq<10) & (df.life_sq>1), 'full_sq'] = df.life_sq
df.loc[df.life_sq>df.full_sq*2, 'life_sq'] = df.life_sq/10


# ----------------- Visualisation ------------------ #
#
df['price/sq']        = df['price_doc'] / df['full_sq']
price_sq_invest       = df.loc[df.product_type=='Investment', 'price/sq']
price_sq_owner        = df.loc[df.product_type=='OwnerOccupier', 'price/sq']
print df.describe()

# Plot Histogram
OrigPriceSqHist = plt.figure()
Orig1 = plt.subplot(311)
sns.distplot(np.log(df['price/sq'].values), bins=200, kde=False, rug=True)
plt.title('Log Price Per Square Meter - Overall')
Orig2 = plt.subplot(312, sharex=Orig1)
sns.distplot(np.log(price_sq_invest.values), bins=200, kde=False, rug=True)
plt.title('Log Price Per Square Meter - Investment')
Orig3 = plt.subplot(313, sharex=Orig1)
sns.distplot(np.log(price_sq_owner.values), bins=200, kde=False, rug=True)
plt.title('Log Price Per Square Meter - Owner')
OrigPriceSqHist.show()



# 1m, 2m, 3m investment purchase
df_1m = df[ (df.price_doc<=1000000) & (df.product_type=="Investment") ]
df_2m = df[ (df.price_doc==2000000) & (df.product_type=="Investment") ]
df_3m = df[ (df.price_doc==3000000) & (df.product_type=="Investment") ]
df_123m = pd.concat([df_1m, df_2m, df_3m])
Invest123mPriceSqHist = plt.figure()
Invest123m1 = plt.subplot(411, sharex=Orig1)
sns.distplot(np.log(df_1m['price/sq'].values), bins=200, kde=False, rug=False)
plt.title('Log Price Per Square Meter - <=1m Investment')
Invest123m2 = plt.subplot(412, sharex=Invest123m1)
sns.distplot(np.log(df_2m['price/sq'].values), bins=200, kde=False, rug=False)
plt.title('Log Price Per Square Meter - 2m Investment')
Invest123m3 = plt.subplot(413, sharex=Invest123m1)
sns.distplot(np.log(df_3m['price/sq'].values), bins=200, kde=False, rug=False)
plt.title('Log Price Per Square Meter - 3m Investment')
Invest123m4 = plt.subplot(414, sharex=Invest123m1)
sns.distplot(np.log(df_123m['price/sq'].values), bins=200, kde=False, rug=False)
plt.title('Log Price Per Square Meter - 1m,2m,3m Investment')
Invest123mPriceSqHist.show()
print "[INFO] 123m Investment:", len(df_123m.index), "records"

# 1m, 2m, 3m Histogram


# ----------------- Remove Extreme Unit Price ----------------- #
df['price/sq'] = df['price_doc'] / df['full_sq']
df = df[ (df['price/sq']<np.exp(13)) & (df['price/sq']>np.exp(10))]
#df_invest_to_drop = df.loc[(df.product_type=='Investment') & (df['price/sq']<np.exp(10))]


# ----------------- Unit Price over Time ----------------- #
df['timestamp']       = pd.to_datetime(df["timestamp"])
df['year']            = df['timestamp'].dt.year
df['quarter']         = df['timestamp'].dt.quarter
df['month']           = df['timestamp'].dt.month
df_unitprice          = df.groupby(['year', 'quarter'])['price/sq'].median()
df_unitprice          = df_unitprice.to_frame()
df_unitprice.columns  = ['price/sq_year_quarter']

df_unitprice = df_unitprice.unstack(level=-1)
print df_unitprice.head()
#df_unitprice.plot(kind='bar', subplots=True)

# ----------------- Visualising Feature over time ---------------
def visualize_feature_over_time(df, feature):
    df['date_column'] = pd.to_datetime(df['timestamp'])
    df['mnth_yr'] = df['date_column'].apply(lambda x: x.strftime('%B-%Y'))

    df = df[[feature,"mnth_yr"]]
    df_vis = df.groupby('mnth_yr')[feature].mean()
    df_vis = df_vis.reset_index()
    df_vis['mnth_yr'] = pd.to_datetime(df_vis['mnth_yr'])
    df_vis.sort_values(by='mnth_yr')
    df_vis.plot(x='mnth_yr', y=feature)
    #    plt.figure()
    #    plt.show()



df["price/sq/cpi"] = df["price/sq"] / df["cpi"]
df["price/cpi"]    = df["price_doc"] / df["cpi"]
df["logprice/cpi"] = np.log(df["price_doc"]) / df["cpi"] 
visualize_feature_over_time(df, "price_doc")
visualize_feature_over_time(df, "price/sq")
visualize_feature_over_time(df, "cpi")
visualize_feature_over_time(df, "price/sq/cpi")
visualize_feature_over_time(df, "logprice/cpi")
visualize_feature_over_time(df, "price/cpi")

visualize_feature_over_time(df_macro, "usdrub")
visualize_feature_over_time(df_macro, "oilrub")
visualize_feature_over_time(df_macro, 'gdp_quart_growth')
visualize_feature_over_time(df_macro, 'balance_trade')


plt.show()
