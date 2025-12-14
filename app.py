import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ================= CONFIG =================
st.set_page_config(
    page_title="Rekomendasi Smartphone",
    layout="wide"
)

st.title("📱 Sistem Rekomendasi Smartphone")
st.write("Menggunakan Metode K-Means Clustering")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("smartphones.csv")

df = load_data()

st.subheader("📄 Dataset Smartphone")
st.dataframe(df.head())

# ================= PREPROCESSING =================
def preprocess_data(df):
    data = df[['price', 'rating', 'ram', 'battery', 'camera', 'display']].copy()

    data['price'] = (
        data['price']
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

    data['ram'] = data['ram'].str.extract(r'(\d+)\s*GB\s*RAM', expand=False)
    data['ram'] = pd.to_numeric(data['ram'], errors='coerce')
    data['battery'] = data['battery'].str.extract(r'(\d+)').astype(float)
    data['camera'] = data['camera'].str.extract(r'(\d+)').astype(float)
    data['display'] = data['display'].str.extract(r'(\d+\.?\d*)').astype(float)

    return data.dropna()

processed = preprocess_data(df)

df_model = df.loc[processed.index].copy()
df_model[['price', 'rating', 'ram', 'battery', 'camera', 'display']] = processed

# Konversi harga ke Rupiah
KURS_INR_TO_IDR = 190
df_model['price_idr'] = df_model['price'] * KURS_INR_TO_IDR

st.subheader("📊 Data Setelah Preprocessing")
st.dataframe(df_model.head())

st.subheader("🧪 Cek Nilai RAM (Debug)")
st.write(df_model['ram'].describe())


# ================= K-MEANS =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    df_model[['price_idr', 'ram']]
)

kmeans = KMeans(n_clusters=3, random_state=42)
df_model['Cluster'] = kmeans.fit_predict(X_scaled)

# ================= LABEL CLUSTER BERDASARKAN HARGA + RAM =================
cluster_summary = (
    df_model
    .groupby('Cluster')
    .agg({
        'price_idr': 'mean',
        'ram': 'mean'
    })
)

# Normalisasi manual untuk scoring
cluster_summary['price_norm'] = (
    cluster_summary['price_idr'] - cluster_summary['price_idr'].min()
) / (
    cluster_summary['price_idr'].max() - cluster_summary['price_idr'].min()
)

cluster_summary['ram_norm'] = (
    cluster_summary['ram'] - cluster_summary['ram'].min()
) / (
    cluster_summary['ram'].max() - cluster_summary['ram'].min()
)

# Skor gabungan
cluster_summary['score'] = (
    cluster_summary['price_norm'] +
    cluster_summary['ram_norm']
)

# Urutkan skor
cluster_summary = cluster_summary.sort_values('score')

cluster_label = {
    cluster_summary.index[0]: 'Low-end',
    cluster_summary.index[1]: 'Mid-end',
    cluster_summary.index[2]: 'High-end'
}

df_model['Kategori'] = df_model['Cluster'].map(cluster_label)

st.subheader("📊 Rata-rata Tiap Cluster (Debug)")
st.dataframe(cluster_summary)


# ================= INPUT USER =================
st.sidebar.subheader("🎯 Preferensi Pengguna")

budget = st.sidebar.number_input(
    "Budget Maksimal (Rp)",
    min_value=1_000_000,
    value=10_000_000,
    step=500_000
)

min_ram = st.sidebar.number_input("RAM Minimal (GB)", min_value=0)


st.sidebar.subheader("📂 Filter Tampilan Cluster")
pilih_kategori = st.sidebar.selectbox(
    "Tampilkan Kategori Smartphone",
    options=["Semua", "Low-end", "Mid-end", "High-end"]
)

# ================= REKOMENDASI (SPK) =================
df_rekomendasi = df_model[
    (df_model['price_idr'] <= budget) &
    (df_model['ram'] >= min_ram)
].copy()

df_rekomendasi = df_rekomendasi.sort_values('price_idr')

tampil_rekomendasi = df_rekomendasi[[
    'model', 'Kategori', 'price_idr',
    'ram', 'camera', 'battery', 'display'
]].rename(columns={
    'model': 'Model',
    'Kategori': 'Kategori',
    'price_idr': 'Harga (Rp)',
    'ram': 'RAM (GB)',
    'camera': 'Kamera (MP)',
    'battery': 'Baterai (mAh)',
    'display': 'Layar (inci)'
})

st.subheader("📌 Rekomendasi Smartphone (Berdasarkan Budget & RAM)")
st.dataframe(tampil_rekomendasi, use_container_width=True)

if df_rekomendasi.empty:
    st.warning("Tidak ada smartphone yang sesuai dengan kriteria.")

# ================= EKSPLORASI CLUSTER =================
st.subheader("📂 Eksplorasi Smartphone Berdasarkan Cluster")

df_cluster_view = df_model.copy()

if pilih_kategori != "Semua":
    df_cluster_view = df_cluster_view[
        df_cluster_view['Kategori'] == pilih_kategori
    ]

df_cluster_view = df_cluster_view.sort_values('price_idr')

tampil_cluster = df_cluster_view[[
    'model', 'Kategori', 'price_idr',
    'ram', 'camera', 'battery', 'display'
]].rename(columns={
    'model': 'Model',
    'Kategori': 'Kategori',
    'price_idr': 'Harga (Rp)',
    'ram': 'RAM (GB)',
    'camera': 'Kamera (MP)',
    'battery': 'Baterai (mAh)',
    'display': 'Layar (inci)'
})

st.dataframe(
    tampil_cluster,
    use_container_width=True
)

st.subheader("📊 Jumlah Data per Cluster")
st.write(df_model['Kategori'].value_counts())

st.subheader("📈 Statistik Tiap Cluster")

cluster_stats = (
    df_model
    .groupby('Kategori')
    .agg({
        'price_idr': ['min', 'mean', 'max'],
        'ram': ['min', 'mean', 'max']
    })
)

st.dataframe(cluster_stats)

