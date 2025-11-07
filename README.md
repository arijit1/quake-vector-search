# 🔍 Quake-Inspired Adaptive Vector Search (Minimal Demo)

A small, self-contained **NumPy prototype** inspired by the paper  
**[“Quake: Adaptive Indexing for Vector Search” (arXiv:2506.03437)](https://arxiv.org/abs/2506.03437)** —  
showing how a vector search index can *adapt* to dynamic and skewed workloads.

---

## 🚀 Overview
This project demonstrates key ideas from **Quake**:
- 🧱 **Hierarchical IVF-style index** (coarse → base partitions)  
- 🎯 **Adaptive Partition Scanning (APS)** — dynamically chooses how many partitions (`nprobe`) to scan per query  
- 🔄 **Split / Merge maintenance** — restructures partitions as data and queries evolve  
- 🧩 Handles **online inserts/deletes** to simulate changing workloads  

Built entirely with **NumPy**, so it’s easy to run, inspect, and extend.

---

## ⚙️ Quickstart
```bash
git clone https://github.com/arijit1/quake-vector-search.git
cd QUAKE_working_demo
pip install -r requirements.txt
python run_demo.py
