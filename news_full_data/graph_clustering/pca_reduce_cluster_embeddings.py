import os
import re
import json
import argparse
import numpy as np

def find_latest_manifest(saved_dir: str):
    """Pick the newest cluster_hybrid_manifest_*.json if --manifest not provided."""
    cands = [f for f in os.listdir(saved_dir)
             if f.startswith("cluster_hybrid_manifest_") and f.endswith(".json")]
    if not cands:
        raise FileNotFoundError(f"No manifest found in {saved_dir}.")
    cands = sorted(cands, key=lambda fn: os.path.getmtime(os.path.join(saved_dir, fn)))
    return os.path.join(saved_dir, cands[-1])

def load_manifest(path: str):
    with open(path, "r") as f:
        meta = json.load(f)
    tag = meta.get("tag")
    if not tag:
        m = re.search(r"cluster_hybrid_manifest_(.+)\.json$", os.path.basename(path))
        tag = m.group(1) if m else "unknown"
    return meta, tag

def run_pca_reduce(saved_dir: str, manifest_path: str, dim: int, whiten: bool, center_only: bool):
    from sklearn.decomposition import PCA

    # Where original clusters are
    src_dir = saved_dir
    # Force outputs into a subfolder "pca"
    out_dir = os.path.join(saved_dir, "pca")
    os.makedirs(out_dir, exist_ok=True)

    if not manifest_path:
        manifest_path = find_latest_manifest(src_dir)
    print(f"[INFO] Using manifest: {manifest_path}")

    meta, tag = load_manifest(manifest_path)
    clusters = meta.get("clusters", {})
    if not clusters:
        raise ValueError("No 'clusters' in manifest.")

    # Load each cluster hybrid embedding
    items, order = [], []
    for cid_str in sorted(clusters.keys(), key=lambda x: int(x)):
        info = clusters[cid_str]
        path = info.get("hybrid_path")
        if not path:
            raise ValueError(f"Cluster {cid_str} missing 'hybrid_path' in manifest.")
        if not os.path.isabs(path):
            path = os.path.join(src_dir, os.path.basename(path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Hybrid file not found: {path}")
        vec = np.load(path)
        items.append(vec.astype(np.float32))
        order.append(int(cid_str))
    X = np.vstack(items)
    print(f"[INFO] Loaded {len(order)} cluster hybrids, original dim={X.shape[1]}")

    # PCA
    pca = PCA(n_components=dim, whiten=whiten, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"[INFO] PCA reduced to dim={dim}. Explained variance sum={pca.explained_variance_ratio_.sum():.4f}")

    # Save reduced embeddings
    out_files = {}
    for cid, row in zip(order, X_pca):
        out_path = os.path.join(out_dir, f"cluster_{cid}_hybrid_{tag}_pca{dim}.npy")
        np.save(out_path, row.astype(np.float32))
        out_files[str(cid)] = out_path
    stacked_path = os.path.join(out_dir, f"all_clusters_hybrid_{tag}_pca{dim}.npy")
    np.save(stacked_path, X_pca.astype(np.float32))
    print(f"[INFO] Saved stacked PCA matrix -> {stacked_path}  shape={X_pca.shape}")

    # Save new manifest
    new_manifest = {
        "tag": tag,
        "source_manifest": os.path.basename(manifest_path),
        "original_dim": int(X.shape[1]),
        "pca_dim": int(dim),
        "whiten": bool(whiten),
        "center_only": bool(center_only),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "clusters": {
            cid: {
                "pca_path": out_files[cid],
                "hybrid_path": clusters[cid]["hybrid_path"],
                "size_nodes": clusters[cid].get("size_nodes"),
                "size_intra_edges": clusters[cid].get("size_intra_edges")
            }
            for cid in sorted(clusters.keys(), key=lambda x: int(x))
        },
        "stacked_pca_path": stacked_path
    }
    out_manifest = os.path.join(out_dir, f"cluster_hybrid_manifest_{tag}_pca{dim}.json")
    with open(out_manifest, "w") as f:
        json.dump(new_manifest, f, indent=2)
    print(f"[INFO] Wrote PCA manifest -> {out_manifest}")

def parse_args():
    ap = argparse.ArgumentParser(description="PCA reduction for saved cluster hybrid embeddings.")
    ap.add_argument("--saved_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_clusters")),
                    help="Directory containing cluster_hybrid_manifest_*.json and cluster_*_hybrid_*.npy")
    ap.add_argument("--manifest", default=None,
                    help="Path to cluster_hybrid_manifest_*.json. If omitted, uses the newest one in saved_dir.")
    ap.add_argument("--dim", type=int, default=4, help="Target PCA dimension (default: 4)")
    ap.add_argument("--whiten", action="store_true", help="Enable PCA whitening (may distort cosine)")
    ap.add_argument("--center_only", action="store_true", help="API symmetry only; PCA centers by default")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pca_reduce(
        saved_dir=args.saved_dir,
        manifest_path=args.manifest,
        dim=args.dim,
        whiten=args.whiten,
        center_only=args.center_only
    )
