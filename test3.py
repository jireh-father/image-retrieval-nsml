import pandas as pd
from sklearn import preprocessing
clustnum = 5
dfcount = 200
days = 60

df = pd.read_csv('E:\\resume\portfolio/sseoqkr7_sku_sale_data_0_27930_slice_season_29374.csv')
print(df)
dffinal = df.iloc[:days]
df_t = dffinal.T
print(df_t)
# data_nor = preprocessing.normalize(df_t, norm='l2')
# print(data_nor)