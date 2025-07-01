import networkx as nx
from collections import defaultdict
import math

class GraphClusterer:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def _prepare_edges(self, weights, verbose=False):
        """
        Prepare edge list with weighted similarity scores.
        Skips edges with NaN or inf values.
        """
        edges = []
        skipped = 0

        for u, v in self.graph.edges():
            edge_data = self.graph[u][v]
            score = sum(
                weights.get(feature, 0) * edge_data.get(feature, 0)
                for feature in weights
            )

            if not math.isfinite(score):
                skipped += 1
                continue

            edges.append([u, v, score])

        if verbose:
            print(f"✅ Prepared {len(edges)} edges (skipped {skipped} invalid)")
            print("🔍 Sample weighted edges:")
            for e in edges[:5]:
                print(e)

        return edges

    def _get_label_mapping(self, labels, node_list):
        """Create a mapping from node ID (str) to its cluster label (int)."""
        return dict(zip(node_list, labels))

    def cluster(self, num_clusters=5, repeats=5, scale="no", seed=42,
                costf="inv", graph_type="distance", feature_weights=None,
                verbose=True):
        """
        Cluster the graph using gclu with robust node mapping and debugging.
        Returns a dictionary with cluster stats and label mapping.
        """
        from gclu import gclu

        # Step 0: Default weights
        if feature_weights is None:
            feature_weights = {
                'semantic_similarity': 0.4,
                'topic_similarity': 0.3,
                'sentiment_similarity': 0.2,
                'frequent': 0.1
            }

        # Step 1: Prepare weighted edges
        edges = self._prepare_edges(feature_weights, verbose=verbose)

        # Step 2: Build node index and remap edges to integers
        node_set = set()
        for u, v, _ in edges:
            node_set.add(u)
            node_set.add(v)
        node_list = sorted(list(node_set))
        node_index = {node: idx for idx, node in enumerate(node_list)}

        if verbose:
            print(f"🧠 Found {len(node_list)} unique nodes")
            print(f"🔢 Requested clusters: {num_clusters}")

        if len(node_list) < num_clusters:
            raise ValueError(f"Too few unique nodes ({len(node_list)}) for {num_clusters} clusters.")

        indexed_edges = [[node_index[u], node_index[v], w] for u, v, w in edges]

        # Step 3: Call gclu on indexed edges
        try:
            labels = gclu(
                indexed_edges,
                graph_type=graph_type,
                num_clusters=num_clusters,
                repeats=repeats,
                scale=scale,
                seed=seed,
                costf=costf
            )
        except Exception as e:
            print("❌ GCLU crashed with error:", e)
            raise

        # Step 4: Map labels back to original node IDs
        label_map = self._get_label_mapping(labels, node_list)

        # Step 5: Build cluster info
        cluster_info = defaultdict(lambda: {'nodes': set(), 'edges': []})
        for u, v in self.graph.edges():
            cu = label_map.get(u)
            cv = label_map.get(v)
            if cu is not None and cu == cv:
                cluster_info[cu]['nodes'].update([u, v])
                cluster_info[cu]['edges'].append((u, v, self.graph[u][v]))

        if verbose:
            print(f"✅ Finished clustering into {num_clusters} clusters")
            for cid, info in cluster_info.items():
                print(f" - Cluster {cid}: {len(info['nodes'])} nodes, {len(info['edges'])} edges")

        return {
            'num_clusters': num_clusters,
            'label_map': label_map,
            'clusters': cluster_info
        }
