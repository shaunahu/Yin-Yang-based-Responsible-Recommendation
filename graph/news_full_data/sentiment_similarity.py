# write_sentiment_similarity_feature.py
import os, json
import numpy as np
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

BASE_GRAPH_DIR = "news_full_dataset_graph_data"
FEAT_FILE = "edge_feats.npy"
META_FILE = "features_meta.json"
FEATURE_NAME = "sentiment_similarity"   # 元数据里登记的名字
DTYPE = np.float16
BATCH_ROWS = 2000

def ensure_meta_and_reserve_col(data_dir, feature_name, num_cols=4):
    meta_path = os.path.join(data_dir, META_FILE)
    feats_path = os.path.join(data_dir, FEAT_FILE)

    # 读节点数量，确定 m
    nodes = np.load(os.path.join(data_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes); m = n*(n-1)//2

    # 如果没有 edge_feats.npy 就创建并用 NaN 填充
    if not os.path.exists(feats_path):
        feat_mm = np.memmap(feats_path, dtype=DTYPE, mode="w+", shape=(m, num_cols))
        feat_mm[:] = np.nan
        feat_mm.flush()

    # 读取/新建 meta
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f: meta = json.load(f)
    else:
        meta = {"num_cols": num_cols, "dtype": str(DTYPE), "features": {}}

    # 分配列（已存在则复用）
    feats = meta["features"]
    if feature_name in feats:
        col = feats[feature_name]
    else:
        used = set(feats.values())
        col = next((c for c in range(meta["num_cols"]) if c not in used), None)
        if col is None:
            raise RuntimeError("edge_feats.npy 列已用满，请扩容或改为每个特征独立文件。")
        feats[feature_name] = col
        with open(meta_path, "w") as f: json.dump(meta, f, ensure_ascii=False, indent=2)
    return n, m, col, meta

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, BASE_GRAPH_DIR)

    # 读取 nodes 顺序
    nodes = np.load(os.path.join(data_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)

    # 计算每个节点的 title/abstract 情感
    items_csv = os.path.join(BASE_DIR, "data_news", "items_filtered.csv")
    df = pd.read_csv(items_csv, dtype=str)
    id2row = {iid:i for i, iid in enumerate(df["item_id"].tolist())}

    def norm_sent(txt: str) -> float:
        return (TextBlob(txt).sentiment.polarity + 1.0) / 2.0

    titles = (df["title"].fillna("")).tolist()
    abstracts = (df["abstract"].fillna("")).tolist()

    title_sent_all = np.zeros(len(df), dtype=np.float32)
    abstract_sent_all = np.zeros(len(df), dtype=np.float32)
    for r in tqdm(range(len(df)), desc="per-item sentiment (title+abstract)"):
        title_sent_all[r] = norm_sent(titles[r])
        abstract_sent_all[r] = norm_sent(abstracts[r])

    # 对齐到 nodes 顺序
    title_sent = np.array([title_sent_all[id2row[iid]] for iid in nodes], dtype=np.float32)  # (n,)
    abstract_sent = np.array([abstract_sent_all[id2row[iid]] for iid in nodes], dtype=np.float32)  # (n,)

    # 准备 memmap 与列号
    n, m, col, meta = ensure_meta_and_reserve_col(data_dir, FEATURE_NAME, num_cols=4)
    feat_mm = np.memmap(os.path.join(data_dir, FEAT_FILE), dtype=DTYPE, mode="r+", shape=(m, meta["num_cols"]))

    # 按上三角顺序写 sim = 1 - (|Δtitle| + |Δabstract|)/2
    cursor = 0
    with tqdm(total=n-1, desc=f"writing '{FEATURE_NAME}' -> col {col}") as pbar:
        for i_block_start in range(0, n-1, BATCH_ROWS):
            i_block_end = min(n-1, i_block_start + BATCH_ROWS)
            for i in range(i_block_start, i_block_end):
                cnt = n - i - 1
                if cnt <= 0: 
                    continue
                d_title = np.abs(title_sent[i] - title_sent[i+1:n])         # (cnt,)
                d_abs   = np.abs(abstract_sent[i] - abstract_sent[i+1:n])   # (cnt,)
                sim = 1.0 - (d_title + d_abs) / 2.0                          # (cnt,)
                feat_mm[cursor:cursor+cnt, col] = sim.astype(DTYPE)
                cursor += cnt
                pbar.update(1)

    feat_mm.flush()
    print("✅ done. sentiment_similarity 写入完成")

if __name__ == "__main__":
    main()
