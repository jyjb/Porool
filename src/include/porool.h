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

/* Initialise the engine from a porool.ini config file.
 * Must be called before any other porool_* function.
 * Not thread-safe: call from a single thread before issuing concurrent queries.
 * Returns  0  on success.
 *          -1  config_path is NULL.
 *          -2  INI parse error.
 *          -3  Tharavu (tde_config_load) failed.
 *          -4  Sorkuvai (ve_init) failed. */
POROOL_API int  porool_init(const char *config_path);

/* Re-open the default chunk/vector handles so subsequent queries see an updated
 * index.  Call after an external vocabulary reload (Sorkuvai) or after the
 * ODAT/OVEC files are rebuilt by an ingest pipeline.
 * Returns 0 on success, -1 if porool_init() has not been called. */
POROOL_API int  porool_refresh(void);

/* Embed a query text into a float vector (mean-pooled token embeddings).
 * *embedding is heap-allocated; free with porool_free() when done.
 * Returns 0 on success, negative on error. */
POROOL_API int  porool_embed_query(const char *query, float **embedding, int *dim);

/* Retrieve the top_k chunks most similar to query_vector.
 * *result_count receives the actual number of results returned (≤top_k).
 * Returns a heap-allocated SearchResult[] array, or NULL on error.
 * Caller must release with porool_free_results(results, *result_count). */
POROOL_API SearchResult *porool_retrieve(float *query_vector, int top_k,
                                         int   *result_count);

/* Re-rank an existing result array in-place using the configured weights.
 * Combines cosine score, chunk-length score, and source priority score.
 * Results are sorted descending by the new composite score. */
POROOL_API void porool_rerank(SearchResult *results, int count);

/* Re-rank in-place using the full query-aware scoring (w4 definitional bonus +
 * w5 term-overlap bonus).  Use this instead of porool_rerank whenever the
 * original query string is available. */
POROOL_API void porool_rerank_query(SearchResult *results, int count,
                                     const char *query);

/* Build a deduplicated context string from a (re-ranked) SearchResult array.
 * max_chars limits the total character budget.  ctx_min_score filters out
 * results whose score is below the threshold (pass 0.0 to include all).
 * Returns a heap-allocated NUL-terminated string, or NULL on error.
 * Caller must release with porool_free(). */
POROOL_API char *porool_build_context(SearchResult *results, int count,
                                      int max_chars, float ctx_min_score);

/* All-in-one pipeline: embed → retrieve → rerank → build context.
 * top_k  ≤0 → use porool.ini default.
 * max_chars ≤0 → use porool.ini default.
 * Returns a heap-allocated NUL-terminated context string, or NULL on error.
 * Caller must release with porool_free(). */
POROOL_API char *porool_query(const char *query, int top_k, int max_chars);

/* Ingest a document into db.table.
 * Supported types: .txt .pdf .jpg .jpeg .png .docx .xlsx
 * External tools required for non-.txt: pdftotext, tesseract, docx2txt, xlsx2csv
 * Registers the table so porool_retrieve_from / porool_query_from with
 * table_name=NULL will include it in all-table searches.
 * porool_init() must be called first (provides the embedding engine).
 * Returns 0 on success, negative on error. */
POROOL_API int porool_ingest(const char *file_path,
                              const char *db, const char *table);

/* Like porool_ingest but attaches metadata to every chunk produced from
 * file_path.  meta may be NULL (equivalent to calling porool_ingest).
 * Returns 0 on success, 1 if file_path is already present (skipped),
 * negative on error. */
POROOL_API int porool_ingest_with_meta(const char *file_path,
                                        const char *db, const char *table,
                                        const ChunkMeta *meta);

/* Retrieve top_k results from db.table_name, or from every registered table
 * in db when table_name is NULL (results are merged and sorted by score).
 * *result_count receives the actual count returned (≤ top_k).
 * Returns heap-allocated SearchResult[], or NULL on error.
 * Caller must free with porool_free_results(). */
POROOL_API SearchResult *porool_retrieve_from(float      *query_vector,
                                               const char *db,
                                               const char *table_name,
                                               int         top_k,
                                               int        *result_count);

/* All-in-one pipeline for a specific table, or all registered tables when
 * table_name is NULL.
 * top_k ≤ 0 uses the porool.ini default; max_chars ≤ 0 uses the INI default.
 * Returns heap-allocated context string, or NULL on error.
 * Caller must free with porool_free(). */
POROOL_API char *porool_query_from(const char *query,
                                    const char *db,
                                    const char *table_name,
                                    int top_k, int max_chars);

/* Unified-target retrieve: target may be "ALL", "db", or "db.table".
 * Behaviour matches porool_retrieve_from; see its docs for return/ownership. */
POROOL_API SearchResult *porool_retrieve_target(float      *query_vector,
                                                 const char *target,
                                                 int         top_k,
                                                 int        *result_count);

/* Unified-target all-in-one pipeline: embed → retrieve → rerank → context.
 * target follows the same "ALL" / "db" / "db.table" convention.
 * top_k ≤ 0 and max_chars ≤ 0 use porool.ini defaults.
 * Caller must free with porool_free(). */
POROOL_API char *porool_query_target(const char *query,
                                      const char *target,
                                      int top_k, int max_chars);

/* Register a query-expansion synonym so occurrences of `from` (whole-word,
 * case-insensitive) in a query cause `to` to be appended before embedding.
 * Call before porool_init() or any query function; takes effect immediately.
 * Returns 0 on success, -1 if already registered, -2 bad args, -3 table full. */
POROOL_API int  porool_synonym_add(const char *from, const char *to);

/* Reload the phrasing cache from the phrasing.query_prefixes and
 * phrasing.chunk_markers ODAT tables (call after external edits to those tables). */
POROOL_API void porool_phrasing_reload(void);

/* Add a pattern to a phrasing table and reload the cache.
 * is_prefix=1 → phrasing.query_prefixes (e.g. "tell me about ").
 * is_prefix=0 → phrasing.chunk_markers  (e.g. " is defined as ").
 * Returns 0 on success, -1 if duplicate, -2 on error. */
POROOL_API int  porool_phrasing_add(const char *pattern, int is_prefix);

/* Free a string returned by porool_embed_query / porool_build_context /
 * porool_query / porool_query_from. */
POROOL_API void porool_free(char *ptr);

/* Free a SearchResult array returned by porool_retrieve. */
POROOL_API void porool_free_results(SearchResult *results, int count);

/* Return the embedding dimension currently in use (== sk_get_dim() of the
 * internal Sorkuvai instance).  Returns 0 if porool_init() has not been
 * called or if the dimension could not be determined. */
POROOL_API int  porool_dim(void);

/* Shut down the engine and release all resources. */
POROOL_API void porool_shutdown(void);

#endif /* POROOL_H */
