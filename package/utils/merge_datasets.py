import os

import pandas as pd 
import numpy as np 
from functools import reduce 

qep_data = pd.read_csv('D:/Research/TextClassification/MachineLearning/qep_nopreds_utf8.csv',
                        encoding='utf-8', 
                        quoting=1)
qep_testnums = qep_data.testnum
dataframes = []
for root, dirs, files in os.walk("D:/Research/TextClassification/MachineLearning"):
    for file in files:
        if file.endswith('All.csv'):
            print("Opening file", file)
            df = pd.read_csv(os.path.join(root, file),
                             encoding='latin-1',
                             usecols=(0,1,2))
            df_testnums = df.testnum
            diff = np.intersect1d(qep_testnums, df_testnums)
            if len(diff) > 0:
                print("Repeated testnums: {}".format(diff))
            dataframes.append(df)
            print("dataframe list length:", len(dataframes))
# final = pd.concat(dataframes, axis=1, ignore_index=True, sort=False)
final = reduce(lambda df1, df2: pd.merge(df1, df2, on='testnum', how='outer'), dataframes)
print(final.head())
print("final shape:", final.shape)
print("qep_data shape:", qep_data.shape)
print(qep_data.head())
end = qep_data.append(final, sort=False)
print("end shape after append:", end.shape)
cols = [x for x in end.columns if x.endswith('f')]
end.fillna(0, inplace=True)
end[cols] = end[cols].astype(int)
end.to_csv('D:/Research/TextClassification/MachineLearning/cat_full_data.csv', encoding='utf-8', quoting=1, index=False)