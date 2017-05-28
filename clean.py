import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read the Original Training Data Set
df_orig = pd.read_csv('input/train.csv')

# Select the relevant columns
ColToSel = ['id', 'full_sq', 'life_sq', 'build_year', 'floor', 'max_floor', 'kitch_sq', 'num_room', 'price_doc']
df = df_orig.loc[:, ColToSel]

# Remove some error rows
df = df[(df.full_sq>1)|(df.life_sq>1)]

# Subjectively correct some errors
df.loc[df.id==3530,  'full_sq'] = 53
df.loc[df.id==3530,  'life_sq'] = 26
df.loc[df.id==13549, 'life_sq'] = 74
df.loc[df.id==10092, 'build_year'] = 2007
df.loc[df.id==10092, 'state'] = 3
df.loc[df.id==25943, 'max_floor'] = 17

# Flag some errors to NA or modify by subjective rules
df.loc[(df.full_sq==1) & (df.life_sq>1), 'full_sq'] = df.life_sq
df.loc[(df.kitch_sq>df.full_sq*0.7)|(df.kitch_sq<2) , 'kitch_sq'] = np.nan
df.loc[(df.build_year<1000) | (df.build_year>2050), 'build_year'] = np.nan
df.loc[(df.num_room>9) & (df.full_sq<100), 'num_room'] = np.nan
df.loc[(df.full_sq<10) & (df.life_sq>10), 'full_sq'] = df.life_sq 
df.loc[df.life_sq<2, 'life_sq'] = np.nan
df.loc[df.life_sq>df.full_sq*2, 'life_sq'] = df.life_sq/10
df.loc[df.full_sq>310, 'full_sq'] = df.full_sq/10
df.loc[df.life_sq>df.full_sq, 'life_sq'] = df.full_sq
df.loc[(df.max_floor==0)|(df.max_floor>60), 'max_floor'] = np.nan
df.loc[(df.floor==0)|(df.floor>df.max_floor), 'floor'] = np.nan


print df.shape
print df.describe()

print df[df.id==3530]
