# Porool CLI — Usage Guide

## Quick Start

```bash
# Ingest a document
porool ingest report.pdf --db mydb --table docs

# Query
porool query "What is the refund policy?" --db mydb --table docs

# Query across all tables in data_dir
porool query "deployment steps" --table all
```

---

## Commands

### `ingest`

Extract, chunk, embed, and store a document.

```
porool ingest <file> --db <db> --table <table> [--ini <tharavu.ini>]
```

| Argument | Required | Description |
|---|---|---|
| `<file>` | yes | Path to the file to ingest (`.txt`, `.pdf`, `.docx`, `.xlsx`) |
| `--db` | yes | Database name (maps to a subdirectory under `data_dir`) |
| `--table` | yes | Table name |
| `--ini` | no | Path to `tharavu.ini` (default: `tharavu.ini` in working directory) |

**What it does:**

1. Extracts text from the file (built-in, no external tools required)
2. Normalizes whitespace
3. Splits into paragraph-aware chunks (max ~500 chars each)
4. Initializes Tharavu + Sorkuvai
5. Appends new chunks to any existing table (re-embeds existing rows if the vector store is missing)
6. Saves the ODAT table and rebuilds the OVEC vector store

**Example:**

```bash
porool ingest docs/manual.pdf --db products --table manual
porool ingest docs/faq.txt    --db products --table faq
```

---

### `query`

Embed a query, search for the closest chunks, re-rank, and print JSON results.

```
porool query <text> --db <db> --table <table|all>
             [--topk N] [--max-chars N]
             [--ini <tharavu.ini>] [--data-dir <dir>]
```

| Argument | Required | Description |
|---|---|---|
| `<text>` | yes | Query string |
| `--db` | no | Database name. Omit or use `all` to search every table |
| `--table` | no | Table name, or `all` to search all tables in the DB |
| `--topk` | no | Maximum results to return (default: 5) |
| `--max-chars` | no | Maximum total characters of text in output (default: 2000) |
| `--ini` | no | Path to `tharavu.ini` |
| `--data-dir` | no | Root data directory (default: `./data`) |

**Modes:**

| `--db` | `--table` | Behaviour |
|---|---|---|
| `mydb` | `docs` | Single table |
| `mydb` | `all` | All tables in `mydb` |
| *(omitted)* | `all` | All tables in all databases under `data_dir` |
| `all` | `all` | Same as above |

**Output format (JSON):**

```json
{
  "query": "What is the refund policy?",
  "results": [
    {
      "rank": 1,
      "score": 0.742318,
      "source": "docs/manual.pdf",
      "text": "Refunds are processed within 5 business days..."
    }
  ]
}
```

`score` is the composite re-ranking score (see [Scoring](#scoring)).

**Examples:**

```bash
# Single table
porool query "installation requirements" --db products --table manual --topk 3

# All tables in one DB
porool query "pricing" --db products --table all

# All tables everywhere
porool query "error code 404"
```

---

### `stats`

Print chunk and vector counts for a table.

```
porool stats --db <db> --table <table> [--ini <tharavu.ini>]
```

**Example:**

```bash
porool stats --db products --table manual
```

```
Table:     products.manual
Chunks:    142
Vectors:   142
Dim:       64
```

---

### `peek`

Inspect the raw text and source of a specific chunk by row ID.

```
porool peek --db <db> --table <table> --id <id> [--ini <tharavu.ini>]
```

**Example:**

```bash
porool peek --db products --table manual --id 12
```

```
ID:     12
Source: docs/manual.pdf
Text:   The installation wizard supports Windows 10 and later...
```

---

### Interactive Mode

Run `porool` with no arguments to enter the interactive wizard.

```
porool RAG engine
-----------------

  [r] Retrieve   [i] Ingest   [q] Quit
Method:
```

Type `r` to query, `i` to ingest a file, `q` to quit.  
Enter `db.table` or `all` when prompted for the target table.

---

## Scoring

Results are re-ranked using a composite score read from `porool.ini`:

```
score = cosine_similarity * w1
      + length_score      * w2
      + keyword_density   * w3
```

| Component | What it measures |
|---|---|
| `cosine_similarity` | Semantic closeness of query and chunk embeddings |
| `length_score` | Gaussian reward centred on `optimal_chunk_len` (default 500 chars) |
| `keyword_density` | Fraction of distinct query words found verbatim in the chunk |

**Tuning tips:**

- If semantically correct chunks rank low → increase `w1`, decrease `w2`
- If length penalty hurts retrieval → set `w2 = 0.00`
- If exact-match queries rank poorly → increase `w3`

Edit `porool.ini` — no recompile needed.

---

## Chunking

Documents are split using paragraph-aware chunking:

1. Split on blank lines (`\n\n`) to respect natural paragraph boundaries
2. Paragraphs longer than `CHUNK_CHARS` (500) are sub-split at the last sentence boundary (`. ? !`) within the limit, then at the last whitespace, then hard-cut

This avoids cutting sentences mid-way and keeps related sentences together.

---

## Data Directory Layout

```
data_dir/           (default: ./data, set in tharavu.ini)
└── <db>/
    ├── <table>.odat    chunk text and source path
    └── <table>.ovec    embedding vectors
```

Tharavu creates `data_dir` and the `<db>` subdirectory automatically on first ingest.

---

## Supported File Types

| Extension | Method |
|---|---|
| `.txt` | Direct read |
| `.pdf` | Built-in PDF stream extractor (zlib-inflated) |
| `.docx` | Built-in DOCX extractor (ZIP + XML parse) |
| `.xlsx` | Built-in XLSX extractor (ZIP + XML parse) |
| `.jpg` `.jpeg` `.png` | Requires `porool_register_ocr()` callback (not in CLI) |

No external tools (pdftotext, LibreOffice, etc.) are required.

---

## Common Workflows

### First ingest

```bash
porool ingest report.pdf --db corp --table reports
```

### Add more documents to the same table

```bash
porool ingest update_q2.pdf --db corp --table reports
```

Existing chunks are preserved; new chunks are appended and the vector store is rebuilt.

### Re-ingest from scratch

Delete the `.odat` and `.ovec` files from `data_dir/corp/` then ingest again.

### Adjust ranking without re-ingesting

Edit `porool.ini` `[scoring]` weights and re-run the query — no rebuild needed.
