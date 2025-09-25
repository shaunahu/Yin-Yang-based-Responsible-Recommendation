# read_matrix.py --- reusable class to load cluster_matrix and manifest
from read_matrix import ClusterReader

reader = ClusterReader(
    "../saved_clusters/cluster_matrix_K5_topk5.pt",
    "../saved_clusters/cluster_matrix_manifest_K5_topk5.json"
)

print("Item Cluster:", reader.get_cluster_of_item("N63834"))
print(reader.get_items_in_cluster(0)[:5])
print(reader.get_cluster_vector(0).shape)
