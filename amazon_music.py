import streamlit as st
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)


class songs:
    def q(self):
        
        
        return {
            "0": "Light Melody Songs",
            "1": "Some Melody Songs",
            "2": "Melody + party songs",
            "3": "Pure Party Songs"
        }

# Load data
df = pd.read_csv(r'C:\Users\DEVA NANTHAN\Documents\amazon_music_project\cleaned_data.csv')
st.write("Data Loaded ✅")


songs = songs()
result = songs.q()
songs_select_label = st.selectbox("Select Mood", options=list(result.values()), key='select Moody songs')
songs_select = [k for k, v in result.items() if v == songs_select_label][0]

# Select cluster
# cluster_choice = st.selectbox("Choose a cluster:", sorted(df['clusters'].unique()))
songs_in_cluster = df[df['clusters'] == int(songs_select)]
st.dataframe(songs_in_cluster[['name_song','name_artists','clusters']].head(), hide_index=True)

# Features
features = ['danceability','energy','loudness','speechiness','acousticness','liveness','valence','tempo','duration_log']
x = df[features]
x_scaled = scaler.transform(x)
x_pca=pca.transform(x_scaled)
cluster_colors = {0: '#1f77b4',  # Blue
                  1: '#ff7f0e',  # Orange
                  2: '#2ca02c',  # Green
                  3: '#d62728',  # Red
                  }  

# Evaluate different k values
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=df['clusters'], ax=ax, palette=cluster_colors)
ax.set_title('2D PCA Scatter Plot of Clusters')
st.pyplot(fig)
