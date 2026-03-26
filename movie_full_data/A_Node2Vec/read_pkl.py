#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os
from pprint import pprint


# ===== 修改这里 =====
PKL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "booksGraph",
    "recommendations_k10_book.pkl"
)


def inspect_object(obj, max_items=5):
    print("\n===== TYPE =====")
    print(type(obj))

    # dict
    if isinstance(obj, dict):
        print("\n===== DICT INFO =====")
        print("Length:", len(obj))

        print("\nSample keys:")
        keys = list(obj.keys())[:max_items]
        for k in keys:
            print("-", k)

        print("\nSample items:")
        for k in keys:
            print(f"\nKey: {k}")
            pprint(obj[k])

    # list
    elif isinstance(obj, list):
        print("\n===== LIST INFO =====")
        print("Length:", len(obj))

        print("\nSample elements:")
        for i in range(min(max_items, len(obj))):
            print(f"\nIndex {i}:")
            pprint(obj[i])

    # numpy array
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            print("\n===== NUMPY ARRAY =====")
            print("Shape:", obj.shape)
            print("Dtype:", obj.dtype)
            print("Sample:", obj[:5])
    except:
        pass

    else:
        print("\n===== VALUE =====")
        pprint(obj)


def main():
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"File not found: {PKL_PATH}")

    print(f"Loading: {PKL_PATH}")

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    inspect_object(data)


if __name__ == "__main__":
    main()