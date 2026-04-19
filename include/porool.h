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
 * SearchResult — one retrieved and re-ranked chunk.
 *
 * text and source are heap-allocated by Porool; release the whole array with
 * porool_free_results().  Do not call free() on individual fields.
 */
typedef struct {
    uint32_t id;     /* row index in the .ovec vector store */
    float    score;  /* composite re-ranked score           */
    char    *text;   /* chunk content  (owned by Porool)    */
    char    *source; /* source document (owned by Porool)   */
} SearchResult;

/* Initialise the engine from a porool.ini config file.
 * Must be called before any other porool_* function.
 * Returns 0 on success, negative on error. */
POROOL_API int  porool_init(const char *config_path);

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

/* Build a deduplicated context string from a (re-ranked) SearchResult array.
 * max_chars limits the total character budget.
 * Returns a heap-allocated NUL-terminated string, or NULL on error.
 * Caller must release with porool_free(). */
POROOL_API char *porool_build_context(SearchResult *results, int count,
                                      int max_chars);

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

/* Free a string returned by porool_embed_query / porool_build_context /
 * porool_query / porool_query_from. */
POROOL_API void porool_free(char *ptr);

/* Free a SearchResult array returned by porool_retrieve. */
POROOL_API void porool_free_results(SearchResult *results, int count);

/* Shut down the engine and release all resources. */
POROOL_API void porool_shutdown(void);

#endif /* POROOL_H */
