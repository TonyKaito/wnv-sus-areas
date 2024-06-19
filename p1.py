import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('location_data_IL.csv')
df.drop_duplicates()
df = df.append({'ADDRESS': 'unknown',
                'LATITUDE': 41.86,
                'LONGITUDE': -87.80,
                'WNVPRESENT': 2}, 
                ignore_index=True)

street_map = gpd.read_file('geo_export_ccb63d37-d208-47e5-9e12-05074598e2f3.shp')

crs = {'init':'epsg:3857'}
geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
geo_df = gpd.GeoDataFrame(df, crs=crs, geometry = geometry)

fig, ax = plt.subplots(figsize=(15,15))
street_map.plot(ax=ax, alpha=0.5, color='grey')

geo_df[geo_df['WNVPRESENT'] == 1].plot(ax=ax, markersize=20, color='red', marker='x', label='WNV Found', legend=True)
geo_df[geo_df['WNVPRESENT'] == 2].plot(ax=ax, markersize=20, color='blue', marker='x', label='WNV Prediction',legend=True)


plt.legend(prop={'size': 10})
plt.title('WNV tests in chicago', fontsize=15, fontweight='bold')
plt.xlim( -88.146669, -87.431252)
plt.ylim( 41.567330, 42.069856)

plt.show()