// ...existing code...
# news_full_data

This folder contains utilities to filter the raw news dataset into a clean item list and behavior file, and to convert the filtered item list into graph representations (pickle or memmap).

## Purpose / workflow

1. Build a token map of valid item IDs (expected as `item_token_map.json` in the `data_news` folder).
2. Filter raw files (`items.tsv`, `new_behaviors.tsv`) to keep only valid item IDs.
   - Use the [`NewsDataFilter`](graph/news_full_data/filter_news_dataste_with_token_maps.py) helper.
3. Convert the filtered item CSV to a fully connected graph:
   - Small/regular: [`convert_item_dataset_to_graph.py`](graph/news_full_data/convert_item_dataset_to_graph.py)
   - Large-scale / memory-mapped: [`convert_item_dataset_to_graph_memmap.py`](graph/news_full_data/convert_item_dataset_to_graph_memmap.py)

## Key files

- [`filter_news_dataste_with_token_maps.py`](graph/news_full_data/filter_news_dataste_with_token_maps.py) — contains the [`NewsDataFilter`](graph/news_full_data/filter_news_dataste_with_token_maps.py) class that:
  - Loads `item_token_map.json`.
  - Produces `items_filtered.csv` and `new_behaviors_filtered.csv` under `data_news/`.
- [`convert_item_dataset_to_graph.py`](graph/news_full_data/convert_item_dataset_to_graph.py) — builds a NetworkX graph from `items_filtered.csv` and saves a pickle (`items_filtered_graph.pkl`).
- [`convert_item_dataset_to_graph_memmap.py`](graph/news_full_data/convert_item_dataset_to_graph_memmap.py) — creates memory-mapped arrays for large graphs (`edges_src_int32.npy`, `edges_dst_int32.npy`, `edge_feats.npy`). See the functions [`main`](graph/news_full_data/convert_item_dataset_to_graph_memmap.py) and [`compute_edge_features_batch`](graph/news_full_data/convert_item_dataset_to_graph_memmap.py).
- Input folder: `data_news/` (expected files: `item_token_map.json`, `items.tsv`, `new_behaviors.tsv`).
- Output folder(s): `news_full_dataset_graph_data/` (memmap / graph outputs).

## Quick usage

- Filter items and behaviors (runs both steps):
```sh
python [filter_news_dataste_with_token_maps.py](http://_vscodecontentref_/0)