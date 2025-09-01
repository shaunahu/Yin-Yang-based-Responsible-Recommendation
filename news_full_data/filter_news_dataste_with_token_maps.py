import os
import json
import re
import pandas as pd
from typing import Iterable, Sequence, Optional

class NewsDataFilter:
    """
    Utilities to filter news datasets using a JSON map of valid item IDs.

    Files expected under <base_dir>/data_news/ by default:
      - item_token_map.json
      - items.tsv
      - new_behaviors.tsv
    """

    def __init__(self, base_dir: Optional[str] = None, data_subdir: str = "data_news"):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, data_subdir)
        self.item_token_map_path = os.path.join(self.data_dir, "item_token_map.json")
        self.valid_ids = self._load_valid_ids(self.item_token_map_path)
        self._splitter = re.compile(r"[,\s]+")

    # ---------- public API ----------

    def filter_items(
        self,
        items_tsv: str = "items.tsv",
        output_csv: str = "items_filtered.csv",
        keep_columns: Sequence[int] = (0, 1, 3, 4),
        new_names: Sequence[str] = ("item_id", "topic", "title", "abstract"),
    ) -> str:
        """
        Filter items.tsv to keep only rows whose item_id exists in item_token_map.json.
        Keep specific columns and rename them.

        Returns path to the written CSV.
        """
        items_path = self._abs(items_tsv)
        out_path = self._abs(output_csv)

        df = pd.read_csv(items_path, sep="\t", header=None, dtype=str)
        df = df[list(keep_columns)].copy()
        df.columns = list(new_names)

        df = df[df["item_id"].isin(self.valid_ids)]
        df.to_csv(out_path, index=False)
        return out_path

    def filter_behaviors(
        self,
        behaviors_tsv: str = "new_behaviors.tsv",
        output_csv: str = "new_behaviors_filtered.csv",
    ) -> str:
        """
        Filter the history and impression columns to keep only item IDs that
        exist in item_token_map.json. Drops rows where both become empty.

        Returns path to the written CSV.
        """
        beh_path = self._abs(behaviors_tsv)
        out_path = self._abs(output_csv)

        # First try with header; if missing, fallback to no header (userid, history, impression)
        df = pd.read_csv(beh_path, sep="\t", dtype=str)
        lower_cols = {c.lower(): c for c in df.columns}
        if "history" not in lower_cols or "impression" not in lower_cols:
            df = pd.read_csv(beh_path, sep="\t", header=None, dtype=str)
            if df.shape[1] < 3:
                raise ValueError("Expected at least 3 columns: userid, history, impression.")
            df = df.iloc[:, :3]
            df.columns = ["userid", "history", "impression"]
        else:
            df = df.rename(columns={lower_cols["history"]: "history",
                                    lower_cols["impression"]: "impression"})

        df["history"] = df["history"].fillna("").astype(str).apply(self._filter_history_tokens)
        df["impression"] = df["impression"].fillna("").astype(str).apply(self._filter_impression_tokens)

        # Drop rows where both are empty after filtering
        df = df[~((df["history"] == "") & (df["impression"] == ""))].copy()

        df.to_csv(out_path, index=False)
        return out_path

    def run_all(self) -> None:
        """Convenience runner for both filtering steps with default filenames."""
        items_out = self.filter_items()
        beh_out = self.filter_behaviors()
        print(f"✅ items saved -> {items_out}")
        print(f"✅ behaviors saved -> {beh_out}")

    # ---------- helpers ----------

    def _load_valid_ids(self, path: str) -> set:
        with open(path, "r") as f:
            data = json.load(f)
        # keys are the item IDs like "N10005"
        return set(data.keys())

    def _abs(self, relative_or_abs: str) -> str:
        return relative_or_abs if os.path.isabs(relative_or_abs) else os.path.join(self.data_dir, relative_or_abs)

    def _filter_history_tokens(self, s: str) -> str:
        """
        History usually looks like a sequence of item_ids:
          'N10005 N12345 N20000' or comma/space separated.
        Keep only those present in valid_ids.
        """
        if not s.strip():
            return ""
        items = [tok for tok in self._splitter.split(s.strip()) if tok]
        kept = [it for it in items if it in self.valid_ids]
        return " ".join(kept)

    def _filter_impression_tokens(self, s: str) -> str:
        """
        Impressions look like tokens 'N12345-1' / 'N12345-0'.
        Keep whole token if the ID before the first '-' is valid.
        """
        if not s.strip():
            return ""
        toks = [tok for tok in self._splitter.split(s.strip()) if tok]
        kept = []
        for tok in toks:
            parts = tok.split("-", 1)
            item_id = parts[0] if parts else tok
            if item_id in self.valid_ids:
                kept.append(tok)
        return " ".join(kept)


if __name__ == "__main__":
    # It will produce:
    #   data_news/items_filtered.csv
    #   data_news/new_behaviors_filtered.csv
    NewsDataFilter().run_all()
