# Contributing to Research-the-Researcher

Thank you for your interest in contributing to **Research-the-Researcher**! 🎓

This project is a 100% fully-local, high-performance Open Source Intelligence (OSINT) academic profiling engine. We welcome contributions from researchers, NLP/ML engineers, and open-source developers.

---

## 📋 Table of Contents
- [Code of Conduct & Ethics](#code-of-conduct--ethics)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Architecture & Key Modules](#architecture--key-modules)
- [Testing](#testing)
- [Pull Request Workflow](#pull-request-workflow)

---

## 🛡️ Code of Conduct & Ethics

**Research-the-Researcher** is strictly intended for public academic intelligence, literature reviews, researcher matchmaking, and institutional discovery. 
* All scraping is restricted to public academic registries (OpenAlex, Semantic Scholar, Google Scholar) and public professional profiles.
* Contributions that attempt to bypass authentication walls, scrape private data, or violate polite rate limits (`TokenBucket`) will be rejected.

---

## 🛠️ Development Setup

### 1. Prerequisites
* **Python 3.10+**
* **Ollama** installed with your model of choice (e.g. `ollama run qwen2.5:14b`).
* *(Optional)* **Docker Desktop** for local SearxNG search.

### 2. Clone & Install
```bash
git clone https://github.com/rk-roshan-kr/Research-the-Researcher.git
cd Research-the-Researcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python setup.py
```

### 3. Verify Setup
Run a test resolution via CLI:
```bash
python run.py --name "Demis Hassabis"
```

---

## 🏗️ Architecture & Key Modules

* `academic_pipeline/resolver.py`: Entity resolution and cross-database disambiguation.
* `academic_pipeline/dedup_engine.py`: 3-stage cascading deduplication (Bloom Filter → MinHash LSH → FAISS).
* `academic_pipeline/collaboration_graph.py`: Co-author graph generation using Max-Heap and Union-Find (DSU).
* `academic_pipeline/local_llm.py`: Async Ollama integration enforcing deterministic Pydantic schemas.
* `academic_pipeline/cache_manager.py`: Two-tier caching (In-memory LRU + SQLite Write-Ahead Logging).
* `web_app.py`: Flask backend serving the React dashboard and streaming real-time pipeline tasks.

---

## 🧪 Testing

We use `pytest` for unit and integration testing. Run tests with:
```bash
pytest tests/
```

Ensure new features include corresponding test cases in `tests/`.

---

## 🚀 Pull Request Workflow

1. **Fork the repo** and create your branch from `main`:
   ```bash
   git checkout -b feat/support-arxiv-search
   ```
2. **Commit your changes**:
   Follow semantic commit messages:
   * `feat: add support for XYZ source`
   * `fix: handle edge case in Bloom filter hash collision`
   * `perf: optimize FAISS batch embedding memory footprint`
3. **Push to your fork** and submit a Pull Request!

---

## 🌟 Maintainer

Project Lead: [**Roshan Kumar Gupta** (@rk-roshan-kr)](https://github.com/rk-roshan-kr)
