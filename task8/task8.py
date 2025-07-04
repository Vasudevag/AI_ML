# Task 8: K-Means Clustering – Mall Customers

# 📦 Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 📥 Step 2: Load Dataset
df = pd.read_csv("/Users/vasu/Documents/python/Aiml/Mall_Customers.csv")
df.head()

# Step 3: Select Features (for clustering)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Scale the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📊 Step 5: Elbow Method to Find Optimal K
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid()
plt.show()

# ✅ Step 6: Train KMeans with optimal K (say K=5)
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 📈 Step 7: Visualize the Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', label='Centroids')
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segments (K=5)")
plt.legend()
plt.show()

# 🔍 Step 8: Evaluate with Silhouette Score
sil_score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", sil_score)
