# Porool

**High-Performance Semantic Retrieval & Context Construction Engine**

`Porool.dll` – The intelligent orchestration layer for fast, production-grade RAG systems.

---

## Project Intention

Porool is a lightweight, high-performance **RAG orchestration engine** written in pure C99. 

Its main purpose is to act as the **brain** of a Retrieval-Augmented Generation pipeline — handling query-time intelligence without reinventing storage, tokenization, or embedding logic.

Instead of building yet another heavy Python-based RAG framework, Porool focuses on **speed, efficiency, and clean architecture** by delegating specialized tasks to two companion DLLs that you created:

- **Sorkuvai.dll** — Tokenization and embedding engine
- **Tharavu.dll** — Data engine with `.ovec` storage and mmap-based vector search

By keeping strict separation of concerns, Porool achieves excellent performance, minimal memory usage, and maximum maintainability.

## Core Philosophy

> **Porool is not a storage engine.**  
> **Porool is not an embedding model.**  
> **Porool is the intelligence layer** that connects language understanding with data retrieval.

It is designed with the mindset of a senior systems engineer who values:
- Sub-100ms query latency
- Zero-copy memory access
- Minimal heap allocations
- Clean, stable APIs
- Production-ready memory management

## Key Responsibilities

1. **Query Embedding** – Converts user query to embedding via Sorkuvai.dll
2. **Semantic Retrieval** – Retrieves top candidates using Tharavu.dll (mmap-backed ANN search)
3. **Intelligent Re-ranking** – Applies custom weighted scoring:
   - Cosine similarity
   - Chunk length optimization
   - Source priority
4. **Context Construction** – Builds clean, deduplicated, well-formatted context for the LLM with proper source attribution and strict length control
5. **End-to-End Orchestration** – Single-call `porool_query()` pipeline

## Architecture Highlights

- **Pure C99** – No external dependencies
- **Modular DLL design** – Easy integration with C/C++, Python (ctypes), or other languages
- **INI-based configuration**
- **Zero-copy reads** via Tharavu’s mmap
- **Thread-safe** after initialization
- **Strict memory ownership** and leak-free design
- **Extensible re-ranking** logic

## Project Structure

```bash
Porool/
├── porool.h              # Public API header
├── porool.c              # Core implementation
├── porool_config.ini     # Configuration file (INI format)
├── main.c                # Sample usage
├── Tharavu.dll           # Your data & vector search engine
├── Sorkuvai.dll          # Your tokenization & embedding engine
└── models/               # Embedding models