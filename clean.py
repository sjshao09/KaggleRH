import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fancyimpute import KNN
from scipy import stats

# Settings: Number of Neighbours
K = 4

# Read the Original Training Data Set
df_train = pd.read_csv('input/train.csv')
df_test  = pd.read_csv('input/test.csv')

# Select the relevant columns
ColToSel = ['id', 'full_sq', 'life_sq', 'build_year', 'floor', 'max_floor', 'kitch_sq', 'num_room', 'state']
df = pd.concat([df_train.loc[:, ColToSel], df_test.loc[:, ColToSel]], ignore_index=True)
#df = df_train

# Remove some error rows
df = df[(df.full_sq>1)|(df.life_sq>1)]

# Subjectively correct some errors
df.loc[df.id==3530,  'full_sq'] = 53
df.loc[df.id==3530,  'life_sq'] = 26
df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==25943, 'max_floor'] = 17
df.loc[df.id==33648, 'num_room'] = 1


# Flag some errors to NA or modify by subjective rules
df.loc[(df.full_sq<2) & (df.life_sq>1), 'full_sq'] = df.life_sq
df.loc[(df.kitch_sq>df.full_sq*0.7)|(df.kitch_sq<2) , 'kitch_sq'] = np.nan
df.loc[(df.build_year<1000) | (df.build_year>2050), 'build_year'] = np.nan
df.loc[(df.num_room>9) & (df.full_sq<100), 'num_room'] = np.nan
df.loc[(df.full_sq<10) & (df.life_sq>10), 'full_sq'] = df.life_sq 
df.loc[df.life_sq<2, 'life_sq'] = np.nan
df.loc[df.life_sq>df.full_sq*2, 'life_sq'] = df.life_sq/10
df.loc[df.full_sq>310, 'full_sq'] = df.full_sq/10
df.loc[df.life_sq>df.full_sq, 'life_sq'] = np.nan
df.loc[(df.max_floor==0)|(df.max_floor>60), 'max_floor'] = np.nan
df.loc[(df.floor==0)|(df.floor>df.max_floor), 'floor'] = np.nan
df.loc[df.full_sq<30, 'num_room'] = 1


print "[INFO] Merged Data before K-NN Missing Data Imputation..."
print df.describe()
print df[df.id==560]
print df[df.id==3530]

# Plot full_sq vs life_sq before imputation
df_test = df_test[(df_test.full_sq>df_test.life_sq)]
FullSqLifeSqFig = plt.figure()
ax1 = plt.subplot(211)
plt.scatter(df['full_sq'].values, df['life_sq'].values, color='b')
plt.title('Training Data Set')
ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
plt.scatter(df_test['full_sq'].values, df_test['life_sq'].values, color='g')
plt.title('Test Data Set')
FullSqLifeSqFig.show()

# Apply linear Regression - full_sq
mask = ~np.isnan(df['full_sq'].values) & ~np.isnan(df['life_sq'].values)
slope, intercept, r_value, p_value, std_err = stats.linregress(df['full_sq'].values[mask], df['life_sq'].values[mask])
print "[INFO] life_sq = full_sq *",slope,"+",intercept

# Apply linear Regression - kitch_sq
mask = ~np.isnan(df['full_sq'].values) & ~np.isnan(df['kitch_sq'].values)
slope, intercept, r_value, p_value, std_err = stats.linregress(df['full_sq'].values[mask], df['kitch_sq'].values[mask])
print "[INFO] kitch_sq = full_sq *",slope,"+",intercept

# NumRooms
NumRoomsFig = plt.figure()
plt.subplot(111)
plt.scatter(df['full_sq'].values, df['num_room'].values, color='b')
plt.title('Number of Rooms')



'''
# Imputation
#df = df.loc[1:4000, :]
df_filled = df
# Imputation - num_room
df_NumRoom = df[['full_sq','num_room']]
df_filled_NumRoom_data = KNN(k=K).complete(df_NumRoom)
df_filled_NumRoom = pd.DataFrame(df_filled_NumRoom_data, columns=['full_sq', 'num_room'])
df_filled_NumRoom['num_room'] = df_filled_NumRoom[['num_room']].applymap(np.rint)
df_filled['num_room'] = df_filled_NumRoom['num_room']
# Imputation - life_sq
df_LifeSq = df[['full_sq','life_sq']]
df_filled_LifeSq_data = KNN(k=K).complete(df_LifeSq)
df_filled_LifeSq = pd.DataFrame(df_filled_LifeSq_data, columns=['full_sq','life_sq'])
df_filled['life_sq'] = df_filled_LifeSq['life_sq']
# Imputation - kitch_sq
df_KitchSq = df[['full_sq','kitch_sq']]
df_filled_KitchSq_data = KNN(k=K).complete(df_KitchSq)
df_filled_KitchSq = pd.DataFrame(df_filled_KitchSq_data, columns=['full_sq','kitch_sq'])
df_filled['kitch_sq'] = df_filled_KitchSq['kitch_sq']



# Replace Missing Values from Imputated Values
df_train.loc[df_train.id.isin(df_filled.id), ColToSel] = df_filled[ColToSel]
df_test.loc[df_test.id.isin(df_filled.id), ColToSel] = df_filled[ColToSel]


print df_filled[df_filled.id==560]
print df_filled[df_filled.id==3530]

NewTrainFileName = 'train_KNN' + str(K) + '.csv'
NewTestFileName  = 'test_KNN'  + str(K) + '.csv'
df_train.to_csv(NewTrainFileName, na_rep='NA', index=False)
df_test.to_csv(NewTestFileName, na_rep='NA', index=False)
'''

plt.show()
