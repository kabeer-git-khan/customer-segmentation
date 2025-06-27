from src.data_preprocessing import load_data, preprocess_data
from src.model import find_optimal_k, build_kmeans

if __name__ == "__main__":
    df = load_data("data/raw/online_retail.xlsx")
    rfm, rfm_scaled = preprocess_data(df)
    find_optimal_k(rfm_scaled)  # Use the plot to choose K
    model, labels = build_kmeans(rfm_scaled, 4)
    rfm['Cluster'] = labels
    rfm.to_csv("data/processed/segmented_customers.csv", index=False)
    print("✅ Clustering complete! Segmented file saved.")
