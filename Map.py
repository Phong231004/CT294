import geopandas as gpd
import folium
from folium.plugins import FastMarkerCluster
import pandas as pd
from sklearn.impute import SimpleImputer

url = "Crime_Data_from_2020_to_Present.csv"
df = pd.read_csv(url)
columns_to_drop = ['DR_NO', 'Date Rptd', 'DATE OCC', 'TIME OCC', 'AREA NAME', 'Rpt Dist No', 'Mocodes', 'LOCATION', 'Cross Street', 'Status Desc']
df.drop(columns=columns_to_drop, inplace=True)

# Handle missing numeric data
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
imputer_numeric = SimpleImputer(strategy='mean')
df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])

# Handle missing categorical data
categorical_columns = df.select_dtypes(include=['object']).columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])

df['LAT'] = pd.to_numeric(df['LAT'])
df['LON'] = pd.to_numeric(df['LON'])

# Create a GeoDataFrame from the DataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LON, df.LAT))

# Create a simple base map using Folium with a local tileset
m = folium.Map(location=[34.0522, -118.2437], zoom_start=10, tiles='openstreetmap')

# Use FastMarkerCluster for better performance
FastMarkerCluster(data=list(zip(gdf['LAT'], gdf['LON']))).add_to(m)

# Save the map as an HTML file
m.save('crime_hotspots_map.html')

# Display the map
m