# Amazon-Music-Clustering
Domain: Music Analytics / Unsupervised Machine Learning  
Project: Automatically group Amazon Music tracks by audio characteristics using clustering methods.


 Project Overview
Manually labeling millions of songs is impractical. This project applies K-Means clustering to Amazon Music audio features (danceability, energy, tempo, acousticness, etc.) so we can automatically discover genre/mood-like groups for:

-  Playlist curation  
-  Song discovery  
-  Artist analysis  
-  Market segmentation  


 Project Workflow
1. Data Exploration & Preprocessing
   - Load dataset single_genre_artists.csv
   - Drop non-numeric fields: track_id, track_name, artist_name
   - Handle missing values & duplicates
   - Normalize features with StandardScaler


2. Feature Selection
   - Use key audio features:  
     `danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms


3. K-Means Clustering
   - Use Elbow Method (Inertia/SSE) to find best k
   - Validate with Silhouette Score
   - Train final KMeans(n_clusters=k) model
   - Assign cluster labels to songs


4. Cluster Evaluation & Profiling
   - Evaluate clusters using:
     - Silhouette Score
     - Davies–Bouldin Index
   - Compute mean values of features per cluster
   - Example:  
     - Cluster 0 → High danceability, high energy → Party Tracks  
     - Cluster 1 → High acousticness, low energy → Chill/Acoustic  


5. Visualization
   - PCA (2D) scatter plot of clusters


 Evaluation Metrics
- Silhouette Score → cohesion & separation  
- Davies–Bouldin Index → cluster quality (lower = better)  
- Inertia (SSE) → used for Elbow method    

