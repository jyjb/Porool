#ifndef POROOL_H
#define POROOL_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef POROOL_EXPORTS
#    define POROOL_API __declspec(dllexport)
#  else
#    define POROOL_API __declspec(dllimport)
#  endif
#else
#  define POROOL_API __attribute__((visibility("default")))
#endif

/*
 * porool_t — opaque instance handle.
 *
 * Multiple instances may coexist in the same process; each owns its own
 * config (data_dir, chunk/vector tables, scoring weights), chunk/vector
 * THARAVU handles, and phrasing cache.
 *
 * SORKUVAI (the embedding engine) is a process-global singleton shared
 * by all instances.  All instances MUST reference the same vocab_name
 * (read from the root section of porool.sxp).  Attempting to create a
 * second instance with a different vocab_name returns NULL.
 *
 * Lifecycle:
 *   porool_create()  — allocates and initialises one instance.
 *   porool_destroy() — releases instance resources; shuts down SORKUVAI
 *                      when the last instance is destroyed.
 */
typedef struct porool_s porool_t;

/*
 * ChunkMeta — optional metadata supplied at ingest time.
 * Pass NULL for any field to store an empty string.
 */
typedef struct {
    const char *concept;          /* e.g. "NERF"                           */
    const char *section;          /* e.g. "description"                    */
    const char *type;             /* e.g. "definition"                     */
    const char *tags;             /* comma-separated, e.g. "intro,basics"  */
    const char *importance;       /* "high" | "medium" | "low"             */
    const char *related_concepts; /* comma-separated related concept names */
    const char *language;         /* e.g. "english", "french"; NULL → use
                                     first language in [vocabularies] group */
} ChunkMeta;

/*
 * SearchResult — one retrieved and re-ranked chunk.
 *
 * All char* fields are heap-allocated by Porool; release the whole array with
 * porool_free_results().  Do not call free() on individual fields.
 */
typedef struct {
    uint32_t id;               /* row index in the .ovec vector store */
    float    score;            /* composite re-ranked score           */
    char    *text;             /* chunk content                       */
    char    *source;           /* source document path                */
    char    *chunk_id;         /* stable chunk identifier             */
    char    *concept;          /* primary concept tag                 */
    char    *section;          /* document section                    */
    char    *type;             /* chunk type (definition, example …)  */
    char    *tags;             /* comma-separated tags                */
    char    *importance;       /* high | medium | low                 */
    char    *related_concepts; /* comma-separated related concepts    */
    char    *language;         /* language key this chunk was indexed under */
} SearchResult;

/*
 * porool_config_t — explicit configuration for porool_create.
 * Call porool_config_defaults() first, then override individual fields.
 * THARAVU and SORKUVAI must be pre-configured by the caller before
 * porool_create() is called.
 */
typedef struct {
    const char *segment;       /* "knowledge" / "context" / "session"; NULL = default */
    int         top_k;         /* chunks to retrieve  (default 10)   */
    float       min_score;     /* minimum chunk score (default 0.10) */
    int         max_chars;     /* context char budget (default 2000) */
    float       w_cosine;      /* composite weight w1 (default 0.60) */
    float       w_length;      /* composite weight w2 (default 0.20) */
    float       w_source;      /* composite weight w3 (default 0.20) */
    float       w_def_bonus;   /* definitional bonus  (default 0.15) */
    float       w_overlap;     /* term-overlap bonus  (default 0.20) */
    /* Directory for phrasing tables (query_prefixes.odat, chunk_markers.odat).
     * NULL or "" → old behaviour: THARAVU logical-name resolver (writes to
     * knowledge/phrasing/ — architectural violation).
     * Set to e.g. "./arivular_log/porool" so phrasing stays out of knowledge/. */
    const char *phrasing_root;
} porool_config_t;

POROOL_API void porool_config_defaults(porool_config_t *cfg);

/*
 * Create a POROOL instance.
 * THARAVU must already be configured (tde_set_base_path called by host).
 * SORKUVAI must already be initialised (ve_init called by host).
 * cfg may be NULL to use all built-in defaults.
 * Returns a handle on success, NULL on failure.
 */
POROOL_API porool_t *porool_create(const porool_config_t *cfg);

/*
 * Destroy an instance and release its resources.
 * SORKUVAI is shut down when the last instance is destroyed.
 * Do not call while queries are in flight on this instance.
 */
POROOL_API void porool_destroy(porool_t *p);

/* Re-open the default chunk/vector handles so subsequent queries see an updated
 * index.  Call after an external vocabulary reload or after ODAT/OVEC files
 * are rebuilt by an ingest pipeline.
 * Returns 0 on success, -1 if the instance is not ready. */
POROOL_API int  porool_refresh(porool_t *p);

/* Embed a query text into a float vector (mean-pooled token embeddings).
 * *embedding is heap-allocated; free with porool_free() when done.
 * Returns 0 on success, negative on error. */
POROOL_API int  porool_embed_query(porool_t *p, const char *query,
                                    float **embedding, int *dim);

/* Retrieve the top_k chunks most similar to query_vector from the instance's
 * default chunk/vector tables.
 * *result_count receives the actual number of results returned (≤ top_k).
 * Returns a heap-allocated SearchResult[] array, or NULL on error.
 * Caller must release with porool_free_results(results, *result_count). */
POROOL_API SearchResult *porool_retrieve(porool_t *p, float *query_vector,
                                          int top_k, int *result_count);

/* Re-rank an existing result array in-place using the instance's scoring weights.
 * Combines cosine score, chunk-length score, and source priority score.
 * Results are sorted descending by the new composite score. */
POROOL_API void porool_rerank(porool_t *p, SearchResult *results, int count);

/* Re-rank in-place using the full query-aware scoring (w4 definitional bonus +
 * w5 term-overlap bonus).  Use this instead of porool_rerank whenever the
 * original query string is available. */
POROOL_API void porool_rerank_query(porool_t *p, SearchResult *results, int count,
                                     const char *query);

/* Build a deduplicated context string from a (re-ranked) SearchResult array.
 * max_chars limits the total character budget.  ctx_min_score filters out
 * results whose score is below the threshold (pass 0.0 to include all).
 * Returns a heap-allocated NUL-terminated string, or NULL on error.
 * Caller must release with porool_free(). */
POROOL_API char *porool_build_context(porool_t *p, SearchResult *results, int count,
                                       int max_chars, float ctx_min_score);

/* All-in-one pipeline: embed → retrieve → rerank → build context.
 * top_k  ≤ 0 → use porool.sxp default.
 * max_chars ≤ 0 → use porool.sxp default.
 * Returns a heap-allocated NUL-terminated context string, or NULL on error.
 * Caller must release with porool_free(). */
POROOL_API char *porool_query(porool_t *p, const char *query,
                               int top_k, int max_chars);

/* Ingest a document into db.table.
 * Supported types: .txt .pdf .jpg .jpeg .png .docx .xlsx
 * External tools required for non-.txt: pdftotext, tesseract, docx2txt, xlsx2csv
 * porool_create() must have been called first (provides the embedding engine).
 * Returns 0 on success, negative on error. */
POROOL_API int porool_ingest(porool_t *p, const char *file_path,
                              const char *db, const char *table);

/* Like porool_ingest but attaches metadata to every chunk produced from
 * file_path.  meta may be NULL (equivalent to calling porool_ingest).
 * Returns 0 on success, 1 if file_path is already present (skipped),
 * negative on error. */
POROOL_API int porool_ingest_with_meta(porool_t *p, const char *file_path,
                                        const char *db, const char *table,
                                        const ChunkMeta *meta);

/* Retrieve top_k results from db.table_name, or from every registered table
 * in db when table_name is NULL (results are merged and sorted by score).
 * *result_count receives the actual count returned (≤ top_k).
 * Returns heap-allocated SearchResult[], or NULL on error.
 * Caller must free with porool_free_results(). */
POROOL_API SearchResult *porool_retrieve_from(porool_t *p, float *query_vector,
                                               const char *db,
                                               const char *table_name,
                                               int top_k, int *result_count);

/* All-in-one pipeline for a specific table, or all registered tables when
 * table_name is NULL.
 * top_k ≤ 0 and max_chars ≤ 0 use the porool.sxp defaults.
 * Returns heap-allocated context string, or NULL on error.
 * Caller must free with porool_free(). */
POROOL_API char *porool_query_from(porool_t *p, const char *query,
                                    const char *db, const char *table_name,
                                    int top_k, int max_chars);

/* Unified-target retrieve: target may be "ALL", "db", or "db.table".
 * Behaviour matches porool_retrieve_from; see its docs for return/ownership. */
POROOL_API SearchResult *porool_retrieve_target(porool_t *p, float *query_vector,
                                                 const char *target,
                                                 int top_k, int *result_count);

/* Unified-target all-in-one pipeline: embed → retrieve → rerank → context.
 * target follows the same "ALL" / "db" / "db.table" convention.
 * top_k ≤ 0 and max_chars ≤ 0 use porool.sxp defaults.
 * Caller must free with porool_free(). */
POROOL_API char *porool_query_target(porool_t *p, const char *query,
                                      const char *target, int top_k, int max_chars);

/* Reload the phrasing cache for this instance from the phrasing.query_prefixes
 * and phrasing.chunk_markers ODAT tables. */
POROOL_API void porool_phrasing_reload(porool_t *p);

/* Add a pattern to a phrasing table and reload the instance's cache.
 * is_prefix=1 → phrasing.query_prefixes (e.g. "tell me about ").
 * is_prefix=0 → phrasing.chunk_markers  (e.g. " is defined as ").
 * Returns 0 on success, -1 if duplicate, -2 on error. */
POROOL_API int  porool_phrasing_add(porool_t *p, const char *pattern, int is_prefix);

/* Register a query-expansion synonym (process-global; linguistic, not per-segment).
 * Occurrences of `from` (whole-word, case-insensitive) in a query cause `to`
 * to be appended before embedding.
 * Returns 0 on success, -1 if already registered, -2 bad args, -3 table full. */
POROOL_API int  porool_synonym_add(const char *from, const char *to);

/* Return the embedding dimension (SORKUVAI global, same for all instances).
 * Returns 0 if no instance has been created yet. */
POROOL_API int  porool_dim(porool_t *p);

/* Free a string returned by porool_embed_query / porool_build_context /
 * porool_query / porool_query_from / porool_query_target. */
POROOL_API void porool_free(char *ptr);

/* Free a SearchResult array returned by porool_retrieve / porool_retrieve_from /
 * porool_retrieve_target. */
POROOL_API void porool_free_results(SearchResult *results, int count);

#endif /* POROOL_H */
