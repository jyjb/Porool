# Porool

**High-Performance Semantic Retrieval & Context Construction Engine**

Porool is a lightweight, pure C99 **RAG orchestration engine** that sits above two companion engines:

- **[Sorkuvai](../Sorkuvai)** — Tokenization, vocabulary, and Xavier-initialized embedding engine
- **[Tharavu](../Data%20Engine)** — Data engine managing `.odat` / `.ovec` / `.ovoc` files with mmap-backed vector search

Porool does not touch storage or embeddings directly — it delegates to Sorkuvai and Tharavu and adds the intelligence layer: chunking, composite re-ranking, and context construction.

---

## Architecture

```
User query
    │
    ▼
porool_cli / porool.dll
    ├── Sorkuvai.dll  →  tokenize + embed
    └── Tharavu.dll   →  ODAT (chunks) + OVEC (vectors) read/write
```

### Strict separation of concerns

| Layer | Responsibility |
|---|---|
| Tharavu | Persistent storage, mmap vector search, `.odat`/`.ovec` files |
| Sorkuvai | Tokenization, vocabulary, deterministic Xavier embeddings |
| Porool | Chunking, ingest pipeline, composite re-ranking, context output |

Porool never creates directories or manages file paths — that belongs to Tharavu (`tde_config_load` handles `data_dir` creation).

---

## Repository Layout

```
Porool/
├── scr/
│   ├── porool.c            Core DLL implementation
│   ├── porool_cli.c        Command-line tool (ingest / query / stats / peek)
│   └── porool_extract.c    Built-in text extraction (TXT, PDF, DOCX, XLSX)
├── include/
│   ├── porool.h            Public DLL API
│   ├── porool_extract.h    Text extraction API
│   ├── sorkuvai_dll.h      Sorkuvai API (from Sorkuvai repo)
│   └── tharavu_dll.h       Tharavu API (from Data Engine repo)
├── bin/                    Build output (DLL, EXE, runtime DLLs)
├── lib/                    Import library (libporool.a on Windows)
├── porool.ini              Runtime configuration
├── Makefile
└── README.md
```

---

## Building

Requires MinGW-w64 (Windows) or GCC (Linux/macOS). Tharavu and Sorkuvai must be built first.

```bash
# From the Porool directory
make
```

Produces:
- `bin/porool.dll` + `lib/libporool.a` — embeddable DLL
- `bin/porool.exe` — command-line tool
- Runtime DLLs copied to `bin/` automatically

### Clean

```bash
make clean
```

---

## Configuration — `porool.ini`

```ini
[porool]
vocabularies = general          ; which vocabulary group to use

[general]
english = general/english.ovoc  ; logical path to the vocab (Sorkuvai)

[scoring]
; Composite score = cosine*w1 + length_score*w2 + keyword_density*w3
w1 = 0.60
w2 = 0.20
w3 = 0.20
optimal_chunk_len    = 500.0    ; Gaussian peak for length score
length_penalty_sigma = 200.0    ; Gaussian width

[source_weights]
; Per-source priority multipliers [0.0, 2.0]
; my_docs = 1.5
```

Tharavu reads its own `tharavu.ini` for `data_dir` and other storage settings.

---

## Composite Re-ranking

Every retrieved chunk is scored:

```
score = cosine_similarity * w1
      + length_score      * w2
      + keyword_density   * w3
```

| Component | Description |
|---|---|
| `cosine_similarity` | Raw dot-product score from Tharavu vector search |
| `length_score` | Gaussian centred on `optimal_chunk_len`; rewards well-sized chunks |
| `keyword_density` | Fraction of distinct query words (>2 chars) found in the chunk |

To disable the length penalty set `w2 = 0.00` and redistribute the weight.

---

## Text Extraction (built-in, no external tools)

| Format | Notes |
|---|---|
| `.txt` | Direct read |
| `.pdf` | Inflate compressed streams, extract text runs |
| `.docx` | Unzip, parse `word/document.xml` |
| `.xlsx` | Unzip, parse `xl/sharedStrings.xml` + sheet XML |
| `.jpg` `.jpeg` `.png` | Requires `porool_register_ocr()` callback |

---

## Data Files

Tharavu resolves logical names to physical files under `data_dir`:

```
data_dir/
└── <db>/
    ├── <table>.odat    chunk text + source metadata
    └── <table>.ovec    embedding vectors (same logical name, different extension)
```

Porool uses `db.table` as the logical name for both `tde_open_odat()` and `tde_open_ovec()` — Tharavu distinguishes by file type automatically.

---

## License

See [LICENSE](LICENSE).
