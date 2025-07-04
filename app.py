# ----------------------------------------------------
# ✅ Save this file as `app.py`
# ✅ Run with: streamlit run app.py
# ----------------------------------------------------

import os
import time
import io
import warnings
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
from keybert import KeyBERT
import umap
import plotly.express as px
import streamlit as st
from get_summary_file import GeminiTicketSummarizer, generate_cluster_name_gemini
import random

# Pick up to 10 summaries per cluster to get a decent name for the cluster
MAX_PER_CLUSTER = 5

# ----------------------------------------------------
# ✅ Suppress warnings
# ----------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "5"

# ----------------------------------------------------
# ✅ Streamlit page settings
# ----------------------------------------------------
st.set_page_config(page_title="Clustering UI", layout="wide")
st.title("📊 Text Clustering & Visualization App")

# ----------------------------------------------------
# ✅ Dynamic configs
# ----------------------------------------------------
SUMMARY_COLUMN_NAME = "Summary" 
NUM_TOP_CLUSTERS = 9
GEMINI_API_KEY = "AIzaSyCdXIHLdNJURJEzlt6-s8XlDWr6FOlAQTw" 

# ----------------------------------------------------
# ✅ File uploader
# ----------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    print("✅ Program started.")
    st.write("Thanks for uploading the excel file, We have started the process...")
    start_time = time.time()

    # ----------------------------------------------------
    # ✅ Load data
    # ----------------------------------------------------
    if 'df' not in st.session_state:
        df = pd.read_excel(uploaded_file)
        st.write("Started getting the meaningful summaries of each row...")
        with st.spinner("⚙️ Generating meaningful summaries..."):
            # ✅ Run summarizer
            summarizer = GeminiTicketSummarizer(
                api_key=GEMINI_API_KEY,
                batch_size=5,
                summary_column=SUMMARY_COLUMN_NAME,
                summary_separator='---',
                polite_pause=1.0
            )

            final_df = summarizer.summarize(df)

        df = final_df.dropna(subset=[SUMMARY_COLUMN_NAME]).reset_index(drop=True)
        st.write("Prepared the meaningful summaries...")
        st.session_state['df'] = df
        print(f"✅ Loaded {len(df)} rows.")
    else:
        df = st.session_state['df']
        print(f"✅ Using cached DataFrame with {len(df)} rows.")

    # ----------------------------------------------------
    # ✅ Load embedding model
    # ----------------------------------------------------
    @st.cache_resource
    def load_model():
        LOCAL_MODEL_DIR = "./local_models/all-MiniLM-L6-v2"
        HF_MODEL_NAME = "all-MiniLM-L6-v2"
        if os.path.exists(LOCAL_MODEL_DIR):
            print(f"✅ Using local model `{LOCAL_MODEL_DIR}`")
            return SentenceTransformer(LOCAL_MODEL_DIR)
        else:
            print(f"⬇️ Downloading `{HF_MODEL_NAME}`")
            model = SentenceTransformer(HF_MODEL_NAME)
            os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
            model.save(LOCAL_MODEL_DIR)
            return model

    model = load_model()
    # st.write("Loaded the Sentence Transformer model...")

    # ----------------------------------------------------
    # ✅ Embeddings
    # ----------------------------------------------------
    st.write("Started embeddings...")

    with st.spinner("⚙️ Generating embeddings..."):
        if 'embeddings' not in st.session_state:
            texts = df[SUMMARY_COLUMN_NAME].tolist()
            embeddings = model.encode(texts, show_progress_bar=True)
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)
            st.session_state['embeddings'] = scaled_embeddings
            print(f"✅ Embeddings ready. Shape: {scaled_embeddings.shape}")
        else:
            scaled_embeddings = st.session_state['embeddings']
            print("✅ Using cached embeddings.")

    # ----------------------------------------------------
    # ✅ Clustering
    # ----------------------------------------------------
    st.write("Started clustering...")
    with st.spinner("⚙️ Generating clusters..."):
        if 'clustered_df' not in st.session_state:
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1)
            hdbscan_labels = hdbscan_clusterer.fit_predict(scaled_embeddings)
            df["Cluster"] = hdbscan_labels
            print(f"✅ HDBSCAN clusters: {len(set(hdbscan_labels))}")

            # KMeans fallback for noise
            noise_mask = df["Cluster"] == -1
            noise_embeddings = scaled_embeddings[noise_mask]
            num_kmeans_clusters = 30
            if len(noise_embeddings) >= num_kmeans_clusters:
                # print("🔄 Running KMeans for noise...")
                kmeans = KMeans(n_clusters=num_kmeans_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(noise_embeddings)
                kmeans_labels_offset = kmeans_labels + df["Cluster"].max() + 1
                df.loc[noise_mask, "Cluster"] = kmeans_labels_offset
                # print(f"✅ KMeans added {num_kmeans_clusters} clusters for noise.")
            else:
                print("⚠️ Skipped KMeans. Not enough noise points.")

            # KeyBERT cluster names
            # kw_model = KeyBERT(model)
            # cluster_names = {}
            # for cluster_id in df["Cluster"].unique():
            #     texts = df[df["Cluster"] == cluster_id][SUMMARY_COLUMN_NAME]
            #     combined = " ".join(texts)
            #     keywords = kw_model.extract_keywords(combined, keyphrase_ngram_range=(1, 2),
            #                                          stop_words="english", top_n=3)
            #     keyword_labels = ", ".join([kw[0] for kw in keywords])
            #     cluster_names[cluster_id] = keyword_labels or "Uncategorized"

            print('Started getting meaninigful cluster names...')
            cluster_names = {}
            for cluster_id in df["Cluster"].unique():
                all_texts = df[df["Cluster"] == cluster_id][SUMMARY_COLUMN_NAME].tolist()
                if len(all_texts) > MAX_PER_CLUSTER:
                    sampled_texts = random.sample(all_texts, MAX_PER_CLUSTER)
                else:
                    sampled_texts = all_texts
                cluster_label = generate_cluster_name_gemini(sampled_texts, summarizer.model)
                cluster_names[cluster_id] = cluster_label

                
            df["Cluster Name"] = df["Cluster"].map(cluster_names)
            print(f"✅ Named {len(cluster_names)} clusters.")

            st.session_state['clustered_df'] = df.copy()
        else:
            df = st.session_state['clustered_df']
            print("✅ Using cached clustered DataFrame.")

    # ----------------------------------------------------
    # ✅ Download clustered Excel
    # ----------------------------------------------------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    st.download_button(
        label="📥 Download Clustered Excel",
        data=output,
        file_name="clustered_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ----------------------------------------------------
    # ✅ Cache UMAP embeddings
    # ----------------------------------------------------
    @st.cache_resource
    def run_umap_2d(data):
        return umap.UMAP(n_components=2, random_state=42).fit_transform(data)

    @st.cache_resource
    def run_umap_3d(data):
        return umap.UMAP(n_components=3, random_state=42).fit_transform(data)

    if 'umap_2d' not in st.session_state:
        st.session_state['umap_2d'] = run_umap_2d(scaled_embeddings)
    if 'umap_3d' not in st.session_state:
        st.session_state['umap_3d'] = run_umap_3d(scaled_embeddings)

    df["UMAP_X"], df["UMAP_Y"] = st.session_state['umap_2d'][:, 0], st.session_state['umap_2d'][:, 1]
    df["UMAP_3D_X"], df["UMAP_3D_Y"], df["UMAP_3D_Z"] = st.session_state['umap_3d'][:, 0], \
                                                         st.session_state['umap_3d'][:, 1], \
                                                         st.session_state['umap_3d'][:, 2]

    print("✅ UMAP embeddings ready.")

    # ----------------------------------------------------
    # ✅ Top clusters
    # ----------------------------------------------------
    top_clusters = df["Cluster"].value_counts().head(NUM_TOP_CLUSTERS).index.tolist()
    print(f"✅ Top clusters: {top_clusters}")
    df_plot = df[df["Cluster"].isin(top_clusters)].copy()
    
    # ----------------------------------------------------
    # ✅ 2D Plot + store image bytes
    # ----------------------------------------------------
    st.write("Generating the 2D graph...")
    if 'fig2d' not in st.session_state:
        # fig2d = px.scatter(
        #     df_plot,
        #     x="UMAP_X",
        #     y="UMAP_Y",
        #     color="Cluster Name",
        #     hover_data=[SUMMARY_COLUMN_NAME],
        #     title=f"Top {NUM_TOP_CLUSTERS} 2D UMAP Clusters"
        # )
        fig2d = px.scatter(
                df_plot,
                x="UMAP_X",
                y="UMAP_Y",
                color="Cluster Name",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                hover_data=[SUMMARY_COLUMN_NAME],
                title=f"Top {NUM_TOP_CLUSTERS} 2D UMAP Clusters",
                render_mode="svg",
            )

        fig2d.update_traces(marker=dict(line=dict(width=0)))

        st.session_state['fig2d'] = fig2d

        fig2d_bytes = io.BytesIO()
        fig2d.write_image(fig2d_bytes, format="png")
        fig2d_bytes.seek(0)
        st.session_state['fig2d_bytes'] = fig2d_bytes

    st.plotly_chart(st.session_state['fig2d'], use_container_width=True)
    st.download_button(
        label="📥 Download 2D UMAP Plot",
        data=st.session_state['fig2d_bytes'],
        file_name="umap_2d_plot.png",
        mime="image/png",
    )
    print("\n✅ New 2D plot ready.")

    # ----------------------------------------------------
    # ✅ 3D Plot + store HTML bytes
    # ----------------------------------------------------'
    st.write("Generating the 3D graph...")
    if 'fig3d' not in st.session_state:
        # fig3d = px.scatter_3d(
        #     df_plot,
        #     x="UMAP_3D_X",
        #     y="UMAP_3D_Y",
        #     z="UMAP_3D_Z",
        #     color="Cluster Name",
        #     hover_data=[SUMMARY_COLUMN_NAME],
        #     title=f"Top {NUM_TOP_CLUSTERS} 3D UMAP Clusters"
        # )
        fig3d = px.scatter_3d(
            df_plot,
            x="UMAP_3D_X",
            y="UMAP_3D_Y",
            z="UMAP_3D_Z",
            color="Cluster Name",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            hover_data=[SUMMARY_COLUMN_NAME],
            title=f"Top {NUM_TOP_CLUSTERS} 3D UMAP Clusters",
        )

        st.session_state['fig3d'] = fig3d

        fig3d_html = io.StringIO()
        fig3d.write_html(fig3d_html)
        st.session_state['fig3d_html'] = fig3d_html.getvalue()

    st.plotly_chart(st.session_state['fig3d'], use_container_width=True)
    st.download_button(
        label="📥 Download 3D UMAP Plot (HTML)",
        data=st.session_state['fig3d_html'],
        file_name="umap_3d_plot.html",
        mime="text/html",
    )
    print("✅ New 3D plot ready.")

    # ----------------------------------------------------
    # ✅ Done
    elapsed = time.time() - start_time
    st.success(f"✅ Done in {elapsed:.2f} seconds.")
    print(f"✅ All done in {elapsed:.2f} seconds.")

else:
    st.info("👆 Upload an Excel file to start.")
