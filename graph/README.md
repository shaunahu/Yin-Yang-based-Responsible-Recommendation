# News Graph Analysis

This project contains a collection of Python scripts to analyze news articles by representing them as a graph. The scripts cover various functionalities, including data conversion, similarity calculation, graph generation, and community detection.

## Project Structure

The project is organized into the following directories:

- `edge_data/`: Contains the raw data for edge attributes, such as topic similarity, semantic similarity, and sentiment similarity.
- `graph_clustering/`: Contains scripts for performing graph clustering.
- `graph_data/`: Contains the generated graph data in various formats.

## Scripts

### Data Conversion and Preprocessing

- **`convert_behavior_dataset_to_graph.py`**: Converts a user behavior dataset (impressions and history) into a graph format. It creates a fully connected graph of all items and saves it as a `.pkl` file.
- **`convert_item_dataset_to_graph.py`**: Converts an item dataset into a graph format.
- **`filter_news_subset_300.py`**: Filters a subset of news articles from a larger dataset.
- **`normalization.py`**: Normalizes the edge values in the graph.
- **`split_csv.py`**: Splits a CSV file into multiple smaller files.
- **`item_frequency_300_news_subset.py`**: Calculates the frequency of items in the news subset.

### Similarity Calculation

- **`semantic_similarity.py`**: Calculates the semantic similarity between news articles.
- **`sentiment_similarity.py`**: Calculates the sentiment similarity between news articles.

### Graph Generation and Manipulation

- **`generate_graph_with_edge_value.py`**: Augments a graph with edge attributes, such as semantic similarity, sentiment similarity, and topic similarity.
- **`load_graph.py`**: Loads a graph from a `.pkl` file.
- **`read_pkl_graph_data.py`**: Reads graph data from a `.pkl` file and prints some basic information about the graph.

### Graph Analysis

- **`community_detection_louvain_alg.py`**: Performs community detection on the graph using the Louvain algorithm.
- **`graph_clustering/`**: This directory contains scripts for performing graph clustering.
  - `main.py`: The main script for running the graph clustering algorithm.
  - `graph_clustering_adaptive_k.py`: Implements an adaptive k-means clustering algorithm for graphs.
  - `read_graph.py`: Reads a graph from a file.
  - `gclu`: A pre-compiled binary for graph clustering.

## Workflow

1. **Data Preparation**: Use the scripts in the root directory to preprocess and convert your data into the required format.
2. **Graph Generation**: Use `generate_graph_with_edge_value.py` to create a graph with edge attributes.
3. **Graph Analysis**: Use the scripts in `graph_clustering/` or `community_detection_louvain_alg.py` to analyze the graph.
