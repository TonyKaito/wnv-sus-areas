import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from tensorflow.keras import activations
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('cleaned_data.csv')
data.replace('',np.nan,regex = True)
data = data.dropna()
data = data.fillna(0)

plt.figure(figsize=(6,5))
plt.barh(data['SPECIES'], data['NUMBER OF MOSQUITOES'], color='#C32D4B')
plt.title('Number of mosquitoes in samples containing WNV', fontsize=15, y=1.05)
plt.yticks(fontsize=12)
plt.show()

plt.figure(figsize=(6,5))
plt.barh(data['WEEK'], data['NUMBER OF MOSQUITOES'], color='#00FFFF')
plt.title('Number of mosquitoes in samples containing WNV', fontsize=15, y=1.05)
plt.yticks(fontsize=12)
plt.show()

data['YEAR'] = ( data['YEAR'] - np.min(data['YEAR']) ) / ( np.max(data['YEAR']) - np.min(data['YEAR']) ) 
data['WEEK'] = ( data['WEEK'] - np.min(data['WEEK']) ) / ( np.max(data['WEEK']) - np.min(data['WEEK']) ) 
data['NUMBER OF MOSQUITOES'] = ( data['NUMBER OF MOSQUITOES'] - np.min(data['NUMBER OF MOSQUITOES']) ) / ( np.max(data['NUMBER OF MOSQUITOES']) - np.min(data['NUMBER OF MOSQUITOES']) ) 
ta = np.min(data['LATLONGCOMB'])
tb = np.max(data['LATLONGCOMB'])
data['LATLONGCOMB'] = ( data['LATLONGCOMB'] - np.min(data['LATLONGCOMB']) ) / ( np.max(data['LATLONGCOMB']) - np.min(data['LATLONGCOMB']) ) 
# data['LONGITUDE'] = ( data['LONGITUDE'] - np.min(data['LONGITUDE']) ) / ( np.max(data['LONGITUDE']) - np.min(data['LONGITUDE']) )
data['TAVG'] = ( data['TAVG'] - np.min(data['TAVG']) ) / ( np.max(data['TAVG']) - np.min(data['TAVG']) ) 


trap_types = {'T089': 0, 'T229': 1, 'T206': 2, 'T144': 3, 'T045': 4, 'T157': 5, 'T147': 6, 'T220': 7, 
              'T900': 8, 'T227': 9, 'T221': 10, 'T046': 11, 'T161': 12, 'T219': 13, 'T077': 14, 
              'T001': 15, 'T007': 16, 'T212': 17, 'T158': 18, 'T037': 19, 'T224': 20, 'T149': 21, 
              'T160': 22, 'T003': 23, 'T015': 24, 'T084': 25, 'T009': 26, 'T141': 27, 'T030': 28, 
              'T043': 29, 'T209': 30, 'T238': 31, 'T086': 32, 'T095': 33, 'T145': 34, 'T004': 35, 
              'T079': 36, 'T091': 37, 'T225': 38, 'T008': 39, 'T027': 40, 'T036': 41, 'T096': 42, 
              'T159': 43, 'T016': 44, 'T074': 45, 'T230': 46, 'T237': 47, 'T005': 48, 'T107': 49, 
              'T081': 50, 'T233': 51, 'T103': 52, 'T031': 53, 'T903': 54, 'T025': 55, 'T088': 56, 
              'T034': 57, 'T236': 58, 'T156': 59, 'T146': 60, 'T102': 61, 'T155': 62, 'T082': 63, 
              'T051': 64, 'T078': 65, 'T154': 66, 'T148': 67, 'T049': 68, 'T072': 69, 'T066': 70, 
              'T114': 71, 'T143': 72, 'T035': 73, 'T215': 74, 'T075': 75, 'T218': 76, 'T033': 77, 
              'T054': 78, 'T094': 79, 'T060': 80, 'T070': 81, 'T083': 82, 'T222': 83, 'T094B': 84, 
              'T012': 85, 'T099': 86, 'T039': 87, 'T085': 88, 'T097': 89, 'T152': 90, 'T054C': 91, 
              'T200': 92, 'T150': 93, 'T017': 94, 'T135': 95, 'T062': 96, 'T138': 97, 'T151': 98, 
              'T040': 99, 'T231': 100, 'T014': 101, 'T129': 102, 'T061': 103, 'T223': 104, 'T153': 105, 
              'T073': 106, 'T044': 107, 'T162': 108, 'T080': 109, 'T232': 110, 'T235': 111, 'T228': 112, 
              'T013': 113, 'T050': 114, 'T076': 115, 'T226': 116, 'T018': 117, 'T142': 118, 'T063': 119, 
              'T028': 120, 'T067': 121, 'T048': 122, 'T006': 123, 'T128': 124, 'T092': 125, 'T047': 126, 
              'T065': 127, 'T069': 128, 'T002': 129, 'T019': 130, 'T090': 131, 'T071': 132, 'T100': 133, 
              'T011': 134, 'T115': 135}
species_types = {'CULEX TARSALIS': 0, 'CULEX SALINARIUS': 1, 'CULEX PIPIENS': 2, 'CULEX RESTUANS': 3, 
                 'CULEX PIPIENS/RESTUANS': 4, 'CULEX TERRITANS': 5, 'CULEX ERRATICUS': 6}
# result_types = {'negative':0, 'positive':1}

data = data.replace({'TRAP':trap_types, 'SPECIES':species_types})
data['TRAP'] = ( data['TRAP'] - np.min(data['TRAP']) ) / ( np.max(data['TRAP']) - np.min(data['TRAP']) ) 
data['SPECIES'] = ( data['SPECIES'] - np.min(data['SPECIES']) ) / ( np.max(data['SPECIES']) - np.min(data['SPECIES']) ) 
# data['RESULT'] = ( data['RESULT'] - np.min(data['RESULT']) ) / ( np.max(data['RESULT']) - np.min(data['RESULT']) )

# data_pt = pd.DataFrame(data, columns=["SEASON YEAR", "WEEK", "TRAP_TYPE", "NUMBER OF MOSQUITOES", "SPECIES", "LATITUDE", "LONGITUDE", "RESULT"])
# print(len(data_pt['RESULT']), len(data_pt['SPECIES']))
# data_pt.plot(x="RESULT",y=["SEASON YEAR"],kind="scatter", figsize=(2, 2))
# data_pt.plot(x="RESULT",y=["WEEK"],kind="scatter", figsize=(2, 2))
# data_pt.plot(x="RESULT",y=["TRAP_TYPE"],kind="scatter", figsize=(2, 2))
# data_pt.plot(x="SEASON YEAR",y=["NUMBER OF MOSQUITOES"],kind="scatter")
# data_pt.plot(x="RESULT",y=["SPECIES"],kind="scatter", figsize=(2, 2))
# data_pt.plot(x="RESULT",y=["LATITUDE"],kind="scatter", figsize=(2, 2))
# data_pt.plot(x="RESULT",y=["LONGITUDE"],kind="scatter", figsize=(2, 2))

# data1 = data.loc[data['LATLONGCOMB'] == 1]
# data2 = data.loc[data['LATLONGCOMB'] == 0]
# data2 = data2.head(len(data1))
# frames = [data1, data2]
# data = pd.concat(frames)

# print("Correlation")
print("Year: ", np.corrcoef(data['YEAR'], data['LATLONGCOMB'])[0,1])
print("Week: ", np.corrcoef(data['WEEK'], data['LATLONGCOMB'])[0,1])
print("Species: ", np.corrcoef(data['SPECIES'], data['LATLONGCOMB'])[0,1])
print("Trap Type: ", np.corrcoef(data['TRAP'], data['LATLONGCOMB'])[0,1])
print("No. of Mosquitoes: ", np.corrcoef(data['NUMBER OF MOSQUITOES'], data['LATLONGCOMB'])[0,1])
print("Average Temperature: ", np.corrcoef(data['TAVG'], data['LATLONGCOMB'])[0,1])
# print("longitude: ", np.corrcoef(data['LONGITUDE'], data['RESULT'])[0,1])
print("WNV Present: ", np.corrcoef(data['WNVPRESENT'], data['LATLONGCOMB'])[0,1])

print(data.head())

msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]

x_train_data = train_data[["YEAR", "WEEK", "TRAP", "NUMBER OF MOSQUITOES", "SPECIES", "WNVPRESENT", "TAVG"]]
y_train_data = train_data["LATLONGCOMB"]
x_test_data = test_data[["YEAR", "WEEK", "TRAP", "NUMBER OF MOSQUITOES", "SPECIES", "WNVPRESENT", "TAVG"]]
y_test_data = test_data["LATLONGCOMB"]

# model creation
model = keras.Sequential()
model.add(Dense(1, kernel_initializer='uniform', input_shape=(7,)))
model.add(Activation(activations.relu))

opt = keras.optimizers.Adam(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=opt)
model.compile(loss='binary_crossentropy', optimizer=opt, run_eagerly=True)

x_train_data.head()

model.fit(x_train_data, y_train_data, epochs=5)
result = model.predict(x_test_data.head(1))
print(result[0][0], ta, tb)
value = result[0][0]
raw = ( value * (tb-ta) )
raw1 = int(raw/10000) * 10000
raw2 = raw % 100
raw = raw1 + raw2
location_raw = raw + ta
location_raw = float( int( location_raw * 10000 ) ) / float(10000)
latitude = float(int((location_raw/10000)) / float(100) )
longitude = float( ( location_raw % ( int(int(location_raw/10000) * 10000 ) ) ) / float(100) )
longitude = float(int(longitude * 100)) / 100
print(latitude, -longitude)
# k = 0
# c = 0
# for i, j in zip(y_test_data, result):
#   if j < 0.5:
#     if i == 0:
#       c += 1
#   else:
#     if i == 1:
#       c += 1
#   k += 1
# print("Sensitivity Score: ", c/k)