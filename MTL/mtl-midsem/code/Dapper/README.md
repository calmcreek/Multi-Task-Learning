# === FILE: README.md ===
"""
DAPPER MTL - Code templates

Files provided in this document (copy each section into its own file under your repo):

- code/Dapper/preprocess_dapper.py     # load raw DAPPER exports, create sliding-window features
- code/Dapper/features.py             # low-level feature extraction helpers
- code/Dapper/dataset.py              # PyTorch Dataset that reads features parquet
- code/Dapper/mtl_model.py            # Multi-modal, multi-task PyTorch model
- code/Dapper/train.py                # Training loop for MTL model
- code/Dapper/config.yaml             # basic config used by train.py
- code/Dapper/utils.py                # utility functions (logging, seed)

These are templates — adapt file paths, column names and label names to match the DAPPER files on Synapse.
"""
