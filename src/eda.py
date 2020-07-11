import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import to_categorical


file_path = '/media/data/projects/crophisto/data.csv'
crop_codes = '/media/data/projects/crophisto/crop_codes.csv'

# load crop info
df_crops = pd.read_csv(crop_codes, sep=";")
# transform data to get a line per each
crop_list = df_crops["code"].to_numpy()
vocab = {val: idx for idx, val in enumerate(crop_list)}


# load dataframe
df = pd.read_csv(file_path)
cols = ['crop2011', 'crop2012', 'crop2013', 'crop2014', 'crop2016', 'crop2015', 'crop2017', 'crop2018', 'crop2019']
data = np.zeros(shape=[df.shape[0], 9], dtype=np.int8)

for index, row in df[cols].iterrows():
    lst = row.tolist()
    lst_conv = [vocab[x] for x in lst]
    data[index, :] = lst_conv

print(data.shape)
np.save(file_path.replace(".csv", ""), data)

