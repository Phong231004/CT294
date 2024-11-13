# Import các thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np
import folium
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from joblib import dump
# Đường dẫn tới tập dữ liệu
url = "Crime_Data_from_2020_to_Present.csv"

# Đọc dữ liệu từ CSV
data = pd.read_csv(url)

# Kiểm tra xem các cột LAT và LON có trong dữ liệu hay không
if 'LAT' not in data.columns or 'LON' not in data.columns:
    raise KeyError("Cột 'LAT' hoặc 'LON' không tìm thấy trong dữ liệu.")

# Loại bỏ các hàng có giá trị NaN trong cột LAT và LON
data = data.dropna(subset=['LAT', 'LON'])

# Chuẩn bị dữ liệu cho KMeans
X = data[['LAT', 'LON']]

# Tìm số cụm tối ưu bằng phương pháp Elbow
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, marker='o')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('Inertia')
plt.title('Phương pháp Elbow để chọn số cụm tối ưu')
plt.grid()
plt.savefig('Elbow.png', dpi=600)
plt.show()

# Chọn số cụm tối ưu từ biểu đồ Elbow (k là 2)
optimal_clusters = 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

# Lưu mô hình KMeans vào tệp
dump(kmeans, 'kmeans_crime_model.pkl')
print("Mô hình KMeans đã được lưu thành công vào 'kmeans_crime_model.pkl'")

# Lấy tọa độ tâm của mỗi cụm
centroids = kmeans.cluster_centers_

# Hiển thị tọa độ tâm của mỗi cụm
print("Tọa độ tâm của mỗi cụm (LAT, LON):")
for i, center in enumerate(centroids):
    print(f"Cụm {i}: Latitude = {center[0]}, Longitude = {center[1]}")

# Tính toán các đặc trưng cho mỗi cụm
crime_characteristics = data.groupby('cluster').agg({
    'Crm Cd Desc': pd.Series.mode,
    'Premis Desc': pd.Series.mode,
    'DR_NO': 'count',
}).rename(columns={
    'DR_NO': 'Count', 
    'Crm Cd Desc': 'Most Common Offense', 
    'Premis Desc': 'Most Common Premise'
})

# Hiển thị các đặc trưng của mỗi cụm
print("Các đặc trưng của mỗi cụm:")
print(crime_characteristics)

# Trực quan hóa các cụm theo LAT và LON với màu sắc khác nhau cho từng cụm
plt.figure(figsize=(12, 6))
sns.scatterplot(x='LON', y='LAT', hue='cluster', data=data, palette='Set2', alpha=0.6)
plt.title('Phân nhóm tội phạm theo vị trí địa lý (LAT, LON)')
plt.xlabel('Kinh độ (LON)')
plt.ylabel('Vĩ độ (LAT)')
plt.legend(title='Cụm')
plt.savefig('Crime_clusters_geolocation.png', dpi=600)
plt.show()

# Vẽ và lưu hai biểu đồ phân tán riêng biệt cho từng cụm với màu khác nhau
colors = ['blue', 'red']  # Đặt màu cho từng cụm
for cluster in range(optimal_clusters):
    plt.figure(figsize=(8, 6))
    cluster_data = data[data['cluster'] == cluster]
    plt.scatter(cluster_data['LON'], cluster_data['LAT'], color=colors[cluster], alpha=0.6)
    plt.title(f'Biểu đồ phân tán cho Cụm {cluster}')
    plt.xlabel('Kinh độ (LON)')
    plt.ylabel('Vĩ độ (LAT)')
    plt.grid()
    # Lưu biểu đồ cho từng cụm
    plt.savefig(f'Cluster_{cluster}_scatter_plot.png', dpi=600)
    plt.show()
