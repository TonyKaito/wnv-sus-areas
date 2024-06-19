import pandas as pd

df1 = pd.read_csv('cleaned_train.csv')

df2 = pd.read_csv('cleaned_weather.csv')

temperature_data = list()
for i in range(0, len(df2)):
	temperature_data.append([df2.at[i, 'Week'], df2.at[i, 'Tavg']])

for ii in range(len(temperature_data)):
	for jj in range(0, len(temperature_data)-ii-1):
		if temperature_data[jj][0] > temperature_data[jj+1][0]:
			temperature_data[jj+1][0], temperature_data[jj][0] = temperature_data[jj][0], temperature_data[jj+1][0]

# for i in range(len(temperature_data)):
# 	print(temperature_data[i])

# print((temperature_data))
newTmp = [0 for _ in range(0, 50)]
i = 0
while i < len(temperature_data):
	s = temperature_data[i][1]
	t = temperature_data[i][0]
	n = 1
	i += 1

	while i < len(temperature_data) and temperature_data[i][0] == t:
		s += temperature_data[i][1]
		i += 1
		n += 1
	avg = float(s) / float(n)
	newTmp[temperature_data[i-1][0]] = avg;

df1 = df1[['Year', 'Week', 'NumMosquitos', 'Trap', 'Species', 'WnvPresent', 'LATLONGCOMB']]

df1['Tavg'] = [ newTmp[df1.at[i, 'Week']] for i in range(len(df1))]
print(df1.head())
# print(set(df1['Trap']))
# print(set(df1['Species']))
# df1.to_csv('cleaned_data.csv')

traps = set(df1['Trap'])
dict_trap = {}
for ii, i in enumerate(traps):
	dict_trap[str(i)] = ii
	
print(dict_trap)

species = set(df1['Species'])
dict_sp = {}
for ii, i in enumerate(species):
	dict_sp[str(i)] = ii
	
print(dict_sp)