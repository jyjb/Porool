/*
 * porool.c — RAG Orchestration Engine (Porool.dll)
 *
 * Pure C99.  Orchestrates Sorkuvai.dll (embedding) and Tharavu.dll (vector
 * search + chunk storage) into a single query pipeline.
 *
 * Init order (mandatory):
 *   tde_config_load()  →  ve_init()  →  open chunk table + vector store
 *
 * Chunk table layout (ODAT, logical name from porool.ini):
 *   col 0  text   STRING  — chunk content
 *   col 1  source STRING  — source document name
 *   row i corresponds to vector store row i (same index).
 */

#ifndef POROOL_EXPORTS
#  define POROOL_EXPORTS
#endif
#include "../include/porool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <ctype.h>
#include "../include/porool_extract.h"

/* ── Sorkuvai.dll forward declarations ──────────────────────────────────────*/
#define VE_OK        0
#define VE_ERR_INVAL -4

#ifdef _WIN32
#  define SK_CALL __cdecl
#  define TDE_CALL __cdecl
#else
#  define SK_CALL
#  define TDE_CALL
#endif

extern int      SK_CALL ve_init(const char *vocab_logical_name, const char *ini_path);
extern void     SK_CALL ve_cleanup(void);
extern int      SK_CALL ve_process_text(const char *text, uint32_t **token_ids, int *count);
extern void     SK_CALL ve_free_ids(uint32_t *ids);
extern int      SK_CALL ve_get_vector(uint32_t token_id, float *out_vec, int dim);
extern int      SK_CALL sk_get_dim(void);

/* ── Tharavu.dll forward declarations ───────────────────────────────────────*/
typedef void *tde_handle_t;

#define TDE_OK       0

extern int          TDE_CALL tde_config_load(const char *ini_path);
extern tde_handle_t TDE_CALL tde_open_odat(const char *logical_name);
extern tde_handle_t TDE_CALL tde_open_ovec(const char *logical_name);
extern void         TDE_CALL tde_close(tde_handle_t h);
extern int          TDE_CALL tde_row_count(tde_handle_t h);
extern int          TDE_CALL tde_get_string(tde_handle_t h, int row, int col,
                                            char *buf, int buf_size);
extern int          TDE_CALL tde_vector_search_topk(tde_handle_t h,
                                                    const float *query_vec,
                                                    uint32_t dim, uint32_t k,
                                                    uint32_t *ids_out,
                                                    float    *scores_out);
extern tde_handle_t TDE_CALL tde_create(const char **col_names, int col_count);
extern int          TDE_CALL tde_save_logical(tde_handle_t h,
                                              const char *logical_name);
extern tde_handle_t TDE_CALL tde_row_begin(tde_handle_t table_h);
extern int          TDE_CALL tde_row_set_string(tde_handle_t row_h, int col,
                                                const char *val);
extern int          TDE_CALL tde_row_commit(tde_handle_t row_h);
extern int          TDE_CALL tde_vector_get_batch(tde_handle_t h,
                                                  const uint32_t *row_ids,
                                                  int count, float *out_buf,
                                                  uint32_t *out_dim);
extern int          TDE_CALL tde_build_vectors_logical(const char *logical_name,
                                                       const float *data,
                                                       int count, uint32_t dim);

/* ── Config ─────────────────────────────────────────────────────────────────*/

#define MAX_SOURCE_WEIGHTS 32

typedef struct { char name[64]; float weight; } source_weight_t;

typedef struct {
    char  tharavu_ini[512];
    char  data_dir[512];
    char  vocab_name[256];
    char  chunks_table[256];
    char  vectors_table[256];
    int   top_k_default;
    int   max_context_chars;
    float optimal_chunk_len;
    float length_penalty_sigma;
    float default_source_weight;
    float w1, w2, w3, w4, w5;
    source_weight_t source_weights[MAX_SOURCE_WEIGHTS];
    int             source_weight_count;
} PoroolConfig;

static PoroolConfig  g_cfg;
static tde_handle_t  g_chunks  = NULL;
static tde_handle_t  g_vectors = NULL;
static int           g_ready   = 0;

/* ── Phrasing cache ──────────────────────────────────────────────────────────*/

#define PHRASE_PREFIXES_LOGICAL "phrasing.query_prefixes"
#define PHRASE_MARKERS_LOGICAL  "phrasing.chunk_markers"
#define MAX_PHRASES 128

static char *g_prefixes[MAX_PHRASES]; static int g_nprefixes;
static char *g_markers [MAX_PHRASES]; static int g_nmarkers;

static const char *k_def_prefixes[] = {
    "what is", "what are", "what's", "define ", "explain ", "describe "
};
static const char *k_def_markers[] = {
    " is a ", " is an ", " are a ", " refers to ", " defined as ",
    " is defined ", " represents a ", " represent a ", " represent an ",
    " known as ", " stands for "
};

/* ── Directory helper ───────────────────────────────────────────────────────*/

#ifdef _WIN32
#  include <windows.h>
#  include <direct.h>
#  define p_mkdir(p) _mkdir(p)
#else
#  include <glob.h>
#  include <sys/stat.h>
#  define p_mkdir(p) mkdir((p), 0755)
#endif

static void ensure_db_subdir(const char *data_dir, const char *db)
{
    p_mkdir(data_dir);
    char path[640];
    snprintf(path, sizeof(path), "%s/%s", data_dir, db);
    p_mkdir(path);
}

/* ── INI writer (default file) ──────────────────────────────────────────────*/

static void write_default_ini(const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp,
        "; Porool configuration file\n"
        "\n"
        "[porool]\n"
        "tharavu_ini           = tharavu.ini\n"
        "data_dir              = ./data\n"
        "vocab_name            = porool.vocab\n"
        "chunks_table          = porool.chunks\n"
        "vectors_table         = porool.chunks_vec\n"
        "top_k_default         = 10\n"
        "max_context_chars     = 2000\n"
        "optimal_chunk_len     = 500.0\n"
        "length_penalty_sigma  = 200.0\n"
        "default_source_weight = 1.0\n"
        "vocabularies          = general\n"
        "\n"
        "[general]\n"
        "english = general/english.ovoc\n"
        "french  = general/french.ovoc\n"
        "\n"
        "[scoring]\n"
        "; Composite score = cosine*w1 + length_score*w2 + source_score*w3\n"
        ";   + definition_signal*w4  (only when query is definitional, e.g. \"what is X?\")\n"
        "; w1+w2+w3 should sum to 1.0; w4 is an additive bonus (0 = disabled)\n"
        "w1 = 0.60\n"
        "w2 = 0.20\n"
        "w3 = 0.20\n"
        "w4 = 0.15\n"
        "w5 = 0.20\n"
        "\n"
        "[source_weights]\n"
        "; Per-source priority multipliers.  Range [0.0, 2.0]; values outside are clamped.\n"
        "; Add entries in the form:  source_name = weight\n"
        "; Example:  my_docs = 1.5\n"
    );
    fclose(fp);
}

/* ── INI parser ─────────────────────────────────────────────────────────────*/

static void trim_inplace(char *s)
{
    /* strip leading whitespace */
    char *p = s;
    while (*p == ' ' || *p == '\t') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    /* strip trailing whitespace */
    p = s + strlen(s) - 1;
    while (p >= s && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n'))
        *p-- = '\0';
}

static int load_config(const char *path)
{
    /* Hard-coded defaults (applied even when file is missing) */
    strncpy(g_cfg.tharavu_ini,   "tharavu.ini",       sizeof(g_cfg.tharavu_ini)   - 1);
    strncpy(g_cfg.data_dir,      "./data",             sizeof(g_cfg.data_dir)      - 1);
    strncpy(g_cfg.vocab_name,    "porool.vocab",       sizeof(g_cfg.vocab_name)    - 1);
    strncpy(g_cfg.chunks_table,  "porool.chunks",      sizeof(g_cfg.chunks_table)  - 1);
    strncpy(g_cfg.vectors_table, "porool.chunks_vec",  sizeof(g_cfg.vectors_table) - 1);
    g_cfg.top_k_default         = 10;
    g_cfg.max_context_chars     = 2000;
    g_cfg.optimal_chunk_len     = 500.0f;
    g_cfg.length_penalty_sigma  = 200.0f;
    g_cfg.default_source_weight = 1.0f;
    g_cfg.w1 = 0.6f; g_cfg.w2 = 0.2f; g_cfg.w3 = 0.2f; g_cfg.w4 = 0.15f; g_cfg.w5 = 0.20f;
    g_cfg.source_weight_count   = 0;

    FILE *fp = fopen(path, "r");
    if (!fp) {
        write_default_ini(path);
        return 0;
    }

    char line[1024], section[64] = "";
    while (fgets(line, sizeof(line), fp)) {
        /* strip inline comments */
        char *c = strchr(line, ';');
        if (!c) c = strchr(line, '#');
        if (c) *c = '\0';
        line[strcspn(line, "\r\n")] = '\0';
        trim_inplace(line);
        if (!line[0]) continue;

        if (line[0] == '[') {
            char *e = strchr(line, ']');
            if (e) { *e = '\0'; snprintf(section, sizeof(section), "%.63s", line + 1); }
            continue;
        }

        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';

        char key[128], val[512];
        snprintf(key, sizeof(key), "%.127s", line);   trim_inplace(key);
        snprintf(val, sizeof(val), "%.511s", eq + 1); trim_inplace(val);
        if (!key[0] || !val[0]) continue;

        if (strcmp(section, "porool") == 0) {
            if      (!strcmp(key, "tharavu_ini"))
                snprintf(g_cfg.tharavu_ini,   sizeof(g_cfg.tharavu_ini),   "%.511s", val);
            else if (!strcmp(key, "data_dir"))
                snprintf(g_cfg.data_dir,      sizeof(g_cfg.data_dir),      "%.511s", val);
            else if (!strcmp(key, "vocab_name"))
                snprintf(g_cfg.vocab_name,    sizeof(g_cfg.vocab_name),    "%.255s", val);
            else if (!strcmp(key, "chunks_table"))
                snprintf(g_cfg.chunks_table,  sizeof(g_cfg.chunks_table),  "%.255s", val);
            else if (!strcmp(key, "vectors_table"))
                snprintf(g_cfg.vectors_table, sizeof(g_cfg.vectors_table), "%.255s", val);
            else if (!strcmp(key, "top_k_default"))
                g_cfg.top_k_default = atoi(val);
            else if (!strcmp(key, "max_context_chars"))
                g_cfg.max_context_chars = atoi(val);
            else if (!strcmp(key, "optimal_chunk_len"))
                g_cfg.optimal_chunk_len = (float)strtod(val, NULL);
            else if (!strcmp(key, "length_penalty_sigma"))
                g_cfg.length_penalty_sigma = (float)strtod(val, NULL);
            else if (!strcmp(key, "default_source_weight"))
                g_cfg.default_source_weight = (float)strtod(val, NULL);
        } else if (strcmp(section, "scoring") == 0) {
            if      (!strcmp(key, "w1")) g_cfg.w1 = (float)strtod(val, NULL);
            else if (!strcmp(key, "w2")) g_cfg.w2 = (float)strtod(val, NULL);
            else if (!strcmp(key, "w3")) g_cfg.w3 = (float)strtod(val, NULL);
            else if (!strcmp(key, "w4")) g_cfg.w4 = (float)strtod(val, NULL);
            else if (!strcmp(key, "w5")) g_cfg.w5 = (float)strtod(val, NULL);
        } else if (strcmp(section, "source_weights") == 0) {
            if (g_cfg.source_weight_count < MAX_SOURCE_WEIGHTS) {
                source_weight_t *sw = &g_cfg.source_weights[g_cfg.source_weight_count++];
                snprintf(sw->name, sizeof(sw->name), "%.63s", key);
                sw->weight = (float)strtod(val, NULL);
            }
        }
    }

    fclose(fp);
    return 0;
}

/* ── Phrasing helpers ────────────────────────────────────────────────────────*/

static void phrases_free(char **arr, int *count)
{
    for (int i = 0; i < *count; i++) { free(arr[i]); arr[i] = NULL; }
    *count = 0;
}

static void phrases_load(const char *logical, char **arr, int *count)
{
    *count = 0;
    tde_handle_t h = tde_open_odat(logical);
    if (!h) return;
    int n = tde_row_count(h);
    for (int i = 0; i < n && *count < MAX_PHRASES; i++) {
        int need = tde_get_string(h, i, 0, NULL, 0);
        if (need <= 0) continue;
        char *s = (char *)malloc((size_t)need);
        if (!s) continue;
        if (tde_get_string(h, i, 0, s, need) > 0 && s[0])
            arr[(*count)++] = s;
        else
            free(s);
    }
    tde_close(h);
}

static void phrases_ensure(const char *logical,
                            const char **defaults, int ndefaults)
{
    tde_handle_t h = tde_open_odat(logical);
    if (h) { tde_close(h); return; }
    const char *cols[] = { "pattern" };
    tde_handle_t tbl = tde_create(cols, 1);
    if (!tbl) return;
    for (int i = 0; i < ndefaults; i++) {
        tde_handle_t row = tde_row_begin(tbl);
        if (!row) continue;
        tde_row_set_string(row, 0, defaults[i]);
        tde_row_commit(row);
    }
    ensure_db_subdir(g_cfg.data_dir, "phrasing");
    tde_save_logical(tbl, logical);
    tde_close(tbl);
}

static void phrases_reload(void)
{
    phrases_free(g_prefixes, &g_nprefixes);
    phrases_free(g_markers,  &g_nmarkers);
    phrases_load(PHRASE_PREFIXES_LOGICAL, g_prefixes, &g_nprefixes);
    phrases_load(PHRASE_MARKERS_LOGICAL,  g_markers,  &g_nmarkers);
}

/* ── Init / shutdown ─────────────────────────────────────────────────────────*/

POROOL_API int porool_init(const char *config_path)
{
    if (!config_path) return -1;
    if (g_ready) porool_shutdown();

    memset(&g_cfg, 0, sizeof(g_cfg));
    if (load_config(config_path) != 0) return -2;

    /* 1. tde_config_load FIRST — sets data_dir and embedding dim for Sorkuvai */
    if (tde_config_load(g_cfg.tharavu_ini) != TDE_OK) return -3;

    /* 2. Init Sorkuvai vocabulary engine (inherits tharavu config via NULL) */
    if (ve_init(g_cfg.vocab_name, NULL) != VE_OK) return -4;

    /* 3+4. Open default chunk/vector tables — non-fatal if absent.
     * porool_retrieve() returns NULL when these are NULL; the multi-table
     * porool_retrieve_from/porool_retrieve_target open their own handles
     * per-call and are unaffected. */
    g_chunks  = tde_open_odat(g_cfg.chunks_table);
    g_vectors = g_chunks ? tde_open_ovec(g_cfg.vectors_table) : NULL;

    /* 5. Ensure phrasing tables exist then load into cache */
    phrases_ensure(PHRASE_PREFIXES_LOGICAL, k_def_prefixes,
                   (int)(sizeof(k_def_prefixes)/sizeof(k_def_prefixes[0])));
    phrases_ensure(PHRASE_MARKERS_LOGICAL,  k_def_markers,
                   (int)(sizeof(k_def_markers)/sizeof(k_def_markers[0])));
    phrases_reload();

    g_ready = 1;
    return 0;
}

POROOL_API void porool_shutdown(void)
{
    if (g_vectors) { tde_close(g_vectors); g_vectors = NULL; }
    if (g_chunks)  { tde_close(g_chunks);  g_chunks  = NULL; }
    phrases_free(g_prefixes, &g_nprefixes);
    phrases_free(g_markers,  &g_nmarkers);
    if (g_ready)   { ve_cleanup(); }
    g_ready = 0;
}

/* ── Utility: case-insensitive substring search ──────────────────────────────*/

/* needle must already be lowercased. */
static int ci_contains(const char *haystack, const char *needle)
{
    if (!haystack || !needle || !needle[0]) return 0;
    size_t nl = strlen(needle);
    for (; *haystack; haystack++) {
        size_t k = 0;
        while (k < nl && haystack[k] &&
               tolower((unsigned char)haystack[k]) == (unsigned char)needle[k])
            k++;
        if (k == nl) return 1;
    }
    return 0;
}

/* ── Synonym expansion ───────────────────────────────────────────────────────*/

#define MAX_SYNONYMS 96

typedef struct { char from[64]; char to[128]; } SynPair;

static SynPair g_synonyms[MAX_SYNONYMS];
static int     g_n_synonyms = 0;

static const char k_def_synonyms[][2][64] = {
    {"cancel",       "terminate"},
    {"cancellation", "termination"},
    {"end",          "terminate"},
    {"break",        "breach"},
    {"violate",      "breach"},
    {"code",         "work product"},
    {"software",     "work product"},
    {"work",         "work product"},
    {"pay",          "payment"},
    {"price",        "fee"},
    {"cost",         "fee"},
    {"hire",         "engage"},
    {"buy",          "purchase"},
    {"own",          "ownership"},
    {"owns",         "assigns"},
    {"own",          "assigns"},
    {"developed",    "created"},
    {"built",        "created"},
    {"wrote",        "created"},
    {"fired",        "terminated"},
    {"dispute",      "arbitration"},
    {"sue",          "arbitration"},
    {"deadline",     "notice"},
};
#define N_DEF_SYNONYMS ((int)(sizeof(k_def_synonyms)/sizeof(k_def_synonyms[0])))

static void synonyms_ensure_defaults(void)
{
    if (g_n_synonyms > 0) return;
    for (int i = 0; i < N_DEF_SYNONYMS && g_n_synonyms < MAX_SYNONYMS; i++) {
        snprintf(g_synonyms[g_n_synonyms].from, 64,  "%.63s",  k_def_synonyms[i][0]);
        snprintf(g_synonyms[g_n_synonyms].to,   128, "%.127s", k_def_synonyms[i][1]);
        g_n_synonyms++;
    }
}

/* Returns heap-allocated expanded query with synonym terms appended.
 * e.g. "how to cancel the agreement" → "…terminate"
 * Caller must free(). Returns NULL only on OOM. */
static char *synonym_expand(const char *query)
{
    if (!query) return NULL;
    synonyms_ensure_defaults();
    size_t ql  = strlen(query);
    size_t cap = ql + (size_t)g_n_synonyms * 130 + 4;
    char *out  = (char *)malloc(cap);
    if (!out) return NULL;
    memcpy(out, query, ql + 1);
    size_t used = ql;
    for (int s = 0; s < g_n_synonyms; s++) {
        const char *from = g_synonyms[s].from;
        const char *to   = g_synonyms[s].to;
        if (!from[0] || !to[0]) continue;
        if (ci_contains(query, to)) continue;
        size_t fl  = strlen(from);
        const char *p = query;
        int hit = 0;
        while (*p && !hit) {
            if (tolower((unsigned char)*p) == (unsigned char)from[0]) {
                size_t k = 0;
                while (k < fl && p[k] &&
                       tolower((unsigned char)p[k]) == (unsigned char)from[k])
                    k++;
                if (k == fl) {
                    unsigned char pre  = (p > query) ? (unsigned char)p[-1] : 0;
                    unsigned char post = (unsigned char)p[fl];
                    if (!isalnum(pre) && !isalnum(post)) hit = 1;
                }
            }
            if (!hit) p++;
        }
        if (hit) {
            size_t tl = strlen(to);
            if (used + 1 + tl < cap) {
                out[used++] = ' ';
                memcpy(out + used, to, tl);
                used += tl;
                out[used] = '\0';
            }
        }
    }
    return out;
}

/* ── Embedding (Sorkuvai) ────────────────────────────────────────────────────*/

/*
 * Mean-pool all token vectors to produce one query embedding.
 * Unknown tokens that have no stored vector still contribute via
 * ve_generate_vector's deterministic fallback inside ve_get_vector.
 */
POROOL_API int porool_embed_query(const char *query, float **embedding, int *dim)
{
    if (!query || !embedding || !dim) return -1;
    *embedding = NULL; *dim = 0;
    if (!g_ready) return -1;

    char *expanded = synonym_expand(query);
    const char *q  = expanded ? expanded : query;

    uint32_t *ids  = NULL;
    int       n    = 0;
    if (ve_process_text(q, &ids, &n) != VE_OK || n == 0) {
        ve_free_ids(ids);
        free(expanded);
        return -1;
    }
    free(expanded);

    int d = sk_get_dim();
    float *emb = (float *)calloc((size_t)d, sizeof(float));
    float *slot = (float *)malloc((size_t)d * sizeof(float));
    if (!emb || !slot) {
        free(emb); free(slot);
        ve_free_ids(ids);
        return -1;
    }

    int added = 0;
    for (int i = 0; i < n; i++) {
        if (ve_get_vector(ids[i], slot, d) == VE_OK) {
            for (int j = 0; j < d; j++) emb[j] += slot[j];
            added++;
        }
    }
    free(slot);
    ve_free_ids(ids);

    if (added == 0) { free(emb); return -1; }

    float inv = 1.0f / (float)added;
    for (int j = 0; j < d; j++) emb[j] *= inv;

    *embedding = emb;
    *dim = d;
    return 0;
}

/* ── Retrieval (Tharavu top-k) ───────────────────────────────────────────────*/

static char *fetch_string(tde_handle_t h, int row, int col)
{
    int need = tde_get_string(h, row, col, NULL, 0);
    if (need <= 0) return NULL;
    char *buf = (char *)malloc((size_t)need);
    if (!buf) return NULL;
    if (tde_get_string(h, row, col, buf, need) <= 0) { free(buf); return NULL; }
    return buf;
}

/* col indices in the chunks ODAT */
#define COL_TEXT             0
#define COL_SOURCE           1
#define COL_CHUNK_ID         2
#define COL_CONCEPT          3
#define COL_SECTION          4
#define COL_TYPE             5
#define COL_TAGS             6
#define COL_IMPORTANCE       7
#define COL_RELATED_CONCEPTS 8
#define ODAT_NCOLS           9

static void fill_result_meta(SearchResult *r, tde_handle_t h, int row)
{
    r->text             = fetch_string(h, row, COL_TEXT);
    r->source           = fetch_string(h, row, COL_SOURCE);
    r->chunk_id         = fetch_string(h, row, COL_CHUNK_ID);
    r->concept          = fetch_string(h, row, COL_CONCEPT);
    r->section          = fetch_string(h, row, COL_SECTION);
    r->type             = fetch_string(h, row, COL_TYPE);
    r->tags             = fetch_string(h, row, COL_TAGS);
    r->importance       = fetch_string(h, row, COL_IMPORTANCE);
    r->related_concepts = fetch_string(h, row, COL_RELATED_CONCEPTS);
}

static void free_result_fields(SearchResult *r)
{
    free(r->text);
    free(r->source);
    free(r->chunk_id);
    free(r->concept);
    free(r->section);
    free(r->type);
    free(r->tags);
    free(r->importance);
    free(r->related_concepts);
}

POROOL_API SearchResult *porool_retrieve(float *query_vector, int top_k,
                                          int   *result_count)
{
    if (!query_vector || top_k <= 0 || !result_count || !g_ready) return NULL;
    if (!g_vectors || !g_chunks) return NULL;
    *result_count = 0;

    int d = sk_get_dim();
    uint32_t *ids    = (uint32_t *)malloc((size_t)top_k * sizeof(uint32_t));
    float    *scores = (float *)   malloc((size_t)top_k * sizeof(float));
    if (!ids || !scores) { free(ids); free(scores); return NULL; }

    int found = tde_vector_search_topk(g_vectors, query_vector,
                                        (uint32_t)d, (uint32_t)top_k,
                                        ids, scores);
    if (found <= 0) { free(ids); free(scores); return NULL; }

    SearchResult *res = (SearchResult *)calloc((size_t)found, sizeof(SearchResult));
    if (!res) { free(ids); free(scores); return NULL; }

    for (int i = 0; i < found; i++) {
        res[i].id    = ids[i];
        res[i].score = scores[i];
        fill_result_meta(&res[i], g_chunks, (int)ids[i]);
    }

    free(ids);
    free(scores);
    *result_count = found;
    return res;
}

/* ── Re-ranking ──────────────────────────────────────────────────────────────*/

static float source_weight_lookup(const char *source)
{
    if (!source) return g_cfg.default_source_weight;
    for (int i = 0; i < g_cfg.source_weight_count; i++)
        if (strcmp(g_cfg.source_weights[i].name, source) == 0)
            return g_cfg.source_weights[i].weight;
    return g_cfg.default_source_weight;
}

static float importance_score(const char *imp)
{
    if (!imp || imp[0] == '\0') return 0.5f;
    if (strcmp(imp, "high")   == 0) return 1.0f;
    if (strcmp(imp, "medium") == 0) return 0.5f;
    if (strcmp(imp, "low")    == 0) return 0.2f;
    return 0.5f;
}

static float length_score(const char *text)
{
    if (!text) return 0.0f;
    float len   = (float)strlen(text);
    float sigma = g_cfg.length_penalty_sigma > 0.0f
                  ? g_cfg.length_penalty_sigma : 1.0f;
    float dev = (len - g_cfg.optimal_chunk_len) / sigma;
    return expf(-0.5f * dev * dev);
}

static int cmp_results_desc(const void *a, const void *b)
{
    float sa = ((const SearchResult *)a)->score;
    float sb = ((const SearchResult *)b)->score;
    return (sa > sb) ? -1 : (sa < sb) ? 1 : 0;
}

/* Returns 1 if the query has a definitional prefix ("what is", "define", etc.)
 * and writes the topic term — everything after the prefix, lowercased, '?' stripped —
 * into out[0..out_sz-1].  Used to gate the w4 bonus so off-topic chunks
 * with incidental "is a / is an" phrasing don't outrank on-topic results. */
static int extract_query_topic(const char *q, char *out, int out_sz)
{
    if (!q || !out || out_sz <= 0) return 0;
    out[0] = '\0';
    while (*q == ' ' || *q == '\t') q++;
    char buf[512];
    int i = 0;
    while (i < (int)sizeof(buf) - 1 && q[i]) {
        buf[i] = (char)tolower((unsigned char)q[i]); i++;
    }
    buf[i] = '\0';
    int n = g_nprefixes > 0 ? g_nprefixes
                            : (int)(sizeof(k_def_prefixes)/sizeof(k_def_prefixes[0]));
    const char **list = g_nprefixes > 0 ? (const char **)g_prefixes : k_def_prefixes;
    for (int p = 0; p < n; p++) {
        size_t pl = strlen(list[p]);
        if (strncmp(buf, list[p], pl) != 0) continue;
        const char *rest = buf + pl;
        while (*rest == ' ') rest++;
        int rlen = (int)strlen(rest);
        while (rlen > 0 && (rest[rlen-1] == '?' || rest[rlen-1] == ' ')) rlen--;
        int copy = rlen < out_sz - 1 ? rlen : out_sz - 1;
        memcpy(out, rest, (size_t)copy);
        out[copy] = '\0';
        return out[0] != '\0' ? 1 : 0;
    }
    return 0;
}

/* Returns [0.0, 1.0] measuring how many definitional markers appear in first 400 chars. */
static float definition_content_score(const char *text)
{
    if (!text) return 0.0f;
    char buf[401];
    int i = 0;
    while (i < 400 && text[i]) { buf[i] = (char)tolower((unsigned char)text[i]); i++; }
    buf[i] = '\0';
    int n          = g_nmarkers > 0 ? g_nmarkers
                                    : (int)(sizeof(k_def_markers)/sizeof(k_def_markers[0]));
    const char **list = g_nmarkers > 0 ? (const char **)g_markers : k_def_markers;
    int hits = 0;
    for (int m = 0; m < n; m++)
        if (strstr(buf, list[m])) hits++;
    if (hits == 0) return 0.0f;
    return 1.0f;
}

static int is_stopword(const char *w)
{
    static const char *s[] = {
        "the","and","for","are","was","is","in","to","of","a","an","or",
        "this","that","with","by","from","who","how","what","when","where",
        "why","which","can","not","all","any","its","may","if","be","as",
        NULL
    };
    for (int i = 0; s[i]; i++) if (strcmp(w, s[i]) == 0) return 1;
    return 0;
}

/* Fraction of distinct non-stopword query words (>2 chars) found in chunk text.
 * Uses synonym-expanded query so "work product" and "assigns" count as hits. */
static float term_overlap_score(const char *query, const char *text)
{
    if (!query || !text) return 0.0f;
    char qbuf[4096]; int i = 0;
    while (i < (int)sizeof(qbuf)-1 && query[i])
        { qbuf[i] = (char)tolower((unsigned char)query[i]); i++; }
    qbuf[i] = '\0';
    char tbuf[8192]; i = 0;
    while (i < (int)sizeof(tbuf)-1 && text[i])
        { tbuf[i] = (char)tolower((unsigned char)text[i]); i++; }
    tbuf[i] = '\0';
    int total = 0, found = 0;
    char *tok = strtok(qbuf, " \t\n\r.,;:!?\"'()[]{}");
    while (tok) {
        if (strlen(tok) > 2 && !is_stopword(tok)) {
            total++;
            if (strstr(tbuf, tok)) found++;
        }
        tok = strtok(NULL, " \t\n\r.,;:!?\"'()[]{}");
    }
    return (total > 0) ? ((float)found / (float)total) : 0.0f;
}

/* Like porool_rerank but incorporates query intent when w4 > 0. Used internally. */
static void rerank_internal(SearchResult *results, int count, const char *query)
{
    if (!results || count <= 0) return;
    float w1 = g_cfg.w1, w2 = g_cfg.w2, w3 = g_cfg.w3, w4 = g_cfg.w4, w5 = g_cfg.w5;
    char topic[256] = "";
    int is_def = (w4 > 0.0f) ? extract_query_topic(query, topic, sizeof(topic)) : 0;
    char *expanded = (w5 > 0.0f) ? synonym_expand(query) : NULL;
    const char *eq  = expanded ? expanded : query;
    for (int i = 0; i < count; i++) {
        float cosine = results[i].score;
        float lscore = length_score(results[i].text);
        float sscore;
        const char *imp = results[i].importance;
        if (imp && imp[0] != '\0') {
            sscore = importance_score(imp);
        } else {
            float wsrc = source_weight_lookup(results[i].source);
            sscore = wsrc > 2.0f ? 1.0f : wsrc * 0.5f;
        }
        /* Definitional bonus only fires when the chunk actually mentions the
         * topic term — prevents off-topic chunks with incidental "is a / is an"
         * phrasing from outranking on-topic results. */
        float dscore = (is_def && ci_contains(results[i].text, topic))
                       ? definition_content_score(results[i].text) : 0.0f;
        float oscore = (w5 > 0.0f)
                       ? term_overlap_score(eq, results[i].text) : 0.0f;
        results[i].score = cosine*w1 + lscore*w2 + sscore*w3 + dscore*w4 + oscore*w5;
    }
    free(expanded);
    qsort(results, (size_t)count, sizeof(SearchResult), cmp_results_desc);
}

POROOL_API void porool_rerank(SearchResult *results, int count)
{
    if (!results || count <= 0) return;
    float w1 = g_cfg.w1, w2 = g_cfg.w2, w3 = g_cfg.w3;
    for (int i = 0; i < count; i++) {
        float cosine = results[i].score;
        float lscore = length_score(results[i].text);
        float wsrc   = source_weight_lookup(results[i].source);
        /* Normalise source weight into [0,1] for the composite score:
         * clamp to [0,2] then divide by 2 so weights up to 2.0 are valid. */
        float sscore = wsrc > 2.0f ? 1.0f : wsrc * 0.5f;
        results[i].score = cosine * w1 + lscore * w2 + sscore * w3;
    }
    qsort(results, (size_t)count, sizeof(SearchResult), cmp_results_desc);
}

POROOL_API void porool_rerank_query(SearchResult *results, int count,
                                     const char *query)
{
    rerank_internal(results, count, query);
}

/* ── Context builder ─────────────────────────────────────────────────────────*/

POROOL_API char *porool_build_context(SearchResult *results, int count, int max_chars)
{
    if (!results || count <= 0 || max_chars <= 0) return NULL;

    size_t capacity = 4096;
    char  *ctx = (char *)malloc(capacity);
    if (!ctx) return NULL;
    ctx[0] = '\0';
    size_t used = 0;

    /* Track seen texts to deduplicate identical chunks */
    const char **seen = (const char **)calloc((size_t)count, sizeof(char *));
    int seen_n = 0;

    for (int i = 0; i < count; i++) {
        if (!results[i].text) continue;

        /* Dedup check */
        int dup = 0;
        for (int j = 0; j < seen_n; j++)
            if (strcmp(results[i].text, seen[j]) == 0) { dup = 1; break; }
        if (dup) continue;

        const char *src  = results[i].source ? results[i].source : "unknown";
        int needed = snprintf(NULL, 0, "[Source: %s]\n%s\n\n", src, results[i].text);
        if (needed <= 0) continue;

        if (used + (size_t)needed > (size_t)max_chars) break;

        /* Grow buffer if required */
        if (used + (size_t)needed + 1 > capacity) {
            capacity = capacity * 2 + (size_t)needed + 1024;
            char *tmp = (char *)realloc(ctx, capacity);
            if (!tmp) { free(seen); free(ctx); return NULL; }
            ctx = tmp;
        }

        snprintf(ctx + used, capacity - used, "[Source: %s]\n%s\n\n",
                 src, results[i].text);
        used += (size_t)needed;
        seen[seen_n++] = results[i].text;
    }

    free(seen);
    return ctx;
}

/* ── All-in-one pipeline ─────────────────────────────────────────────────────*/

POROOL_API char *porool_query(const char *query, int top_k, int max_chars)
{
    if (!query || !g_ready) return NULL;
    if (top_k     <= 0) top_k     = g_cfg.top_k_default;
    if (max_chars <= 0) max_chars = g_cfg.max_context_chars;

    float *emb = NULL;
    int    dim = 0;
    if (porool_embed_query(query, &emb, &dim) != 0) return NULL;

    int          n   = 0;
    SearchResult *res = porool_retrieve(emb, top_k, &n);
    free(emb);
    if (!res) return NULL;

    rerank_internal(res, n, query);

    char *ctx = porool_build_context(res, n, max_chars);
    porool_free_results(res, n);
    return ctx;
}

/* ── Table discovery (folder = db, *.odat files inside = tables) ────────────*/

/* Scan data_dir/db/ for *.odat files; fill out[] with "db.tablename" entries.
 * Returns count. */
static int registry_read(const char *db, char out[][256], int max_out)
{
    int n = 0;
#ifdef _WIN32
    char pat[640];
    snprintf(pat, sizeof(pat), "%s/%s/*.odat", g_cfg.data_dir, db);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        const char *fn = fd.cFileName;
        int fl = (int)strlen(fn);
        if (fl <= 5) continue;          /* ".odat" = 5 chars */
        if (n >= max_out) break;
        snprintf(out[n], 256, "%s.%.*s", db, fl - 5, fn);
        n++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    char pat[640];
    snprintf(pat, sizeof(pat), "%s/%s/*.odat", g_cfg.data_dir, db);
    glob_t g = {0};
    if (glob(pat, 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc && n < max_out; i++) {
            const char *base = strrchr(g.gl_pathv[i], '/');
            base = base ? base + 1 : g.gl_pathv[i];
            int bl = (int)strlen(base);
            if (bl <= 5) continue;
            snprintf(out[n], 256, "%s.%.*s", db, bl - 5, base);
            n++;
        }
        globfree(&g);
    }
#endif
    return n;
}

/* Scan all subdirectories of data_dir; for each subdir (= db), scan *.odat.
 * Returns total count. */
static int registry_read_all(char out[][256], int max_out)
{
    int n = 0;
#ifdef _WIN32
    char pat[640];
    snprintf(pat, sizeof(pat), "%s/*", g_cfg.data_dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (fd.cFileName[0] == '.') continue;
        n += registry_read(fd.cFileName, out + n, max_out - n);
    } while (n < max_out && FindNextFileA(h, &fd));
    FindClose(h);
#else
    char pat[640];
    snprintf(pat, sizeof(pat), "%s/*/*.odat", g_cfg.data_dir);
    glob_t g = {0};
    if (glob(pat, 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc && n < max_out; i++) {
            const char *path = g.gl_pathv[i];
            const char *slash = strrchr(path, '/');
            if (!slash) continue;
            const char *base = slash + 1;
            int bl = (int)strlen(base);
            if (bl <= 5) continue;
            const char *slash2 = slash - 1;
            while (slash2 > path && *slash2 != '/') slash2--;
            if (*slash2 == '/') slash2++;
            int dl = (int)(slash - slash2);
            if (dl <= 0 || dl >= 256) continue;
            char db[256];
            memcpy(db, slash2, (size_t)dl); db[dl] = '\0';
            snprintf(out[n], 256, "%s.%.*s", db, bl - 5, base);
            n++;
        }
        globfree(&g);
    }
#endif
    return n;
}

/* Parse a target string into db and optional table components.
 * Returns 2 = all dbs, 1 = all tables in db, 0 = specific db.table. */
static int parse_target(const char *target, char *db_out, int db_sz,
                         char *tbl_out, int tbl_sz)
{
    db_out[0] = tbl_out[0] = '\0';
    if (!target || !*target || !strcasecmp(target, "all") || !strcmp(target, "*"))
        return 2;
    const char *dot = strchr(target, '.');
    if (!dot) {
        snprintf(db_out, (size_t)db_sz, "%s", target);
        return 1;
    }
    int dl = (int)(dot - target);
    snprintf(db_out,  (size_t)db_sz,  "%.*s", dl, target);
    snprintf(tbl_out, (size_t)tbl_sz, "%s",   dot + 1);
    return 0;
}

/* ── Ingest helpers ──────────────────────────────────────────────────────────*/

#define POROOL_CHUNK_CHARS   500
#define POROOL_CHUNK_OVERLAP 150

static void p_normalize(char *s)
{
    char *r = s, *w = s;
    int space = 0;
    while (*r) {
        unsigned char c = (unsigned char)*r++;
        if (c < 0x20 || c == 0x7f) {
            if (!space) { *w++ = ' '; space = 1; }
        } else { *w++ = (char)c; space = 0; }
    }
    *w = '\0';
    char *p = s;
    while (*p == ' ') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    int len = (int)strlen(s);
    while (len > 0 && s[len - 1] == ' ') s[--len] = '\0';
}

static char **p_chunk_text(const char *text, int *out_n)
{
    int len = (int)strlen(text);
    if (len == 0) { *out_n = 0; return NULL; }
    int cap = len / (POROOL_CHUNK_CHARS - POROOL_CHUNK_OVERLAP) + 4;
    char **chunks = (char **)malloc((size_t)cap * sizeof(char *));
    if (!chunks) return NULL;
    int n = 0, start = 0;
    while (start < len && n < cap - 1) {
        int end = start + POROOL_CHUNK_CHARS;
        if (end >= len) {
            end = len;
        } else {
            int probe = end;
            while (probe > start + POROOL_CHUNK_CHARS / 2 &&
                   !isspace((unsigned char)text[probe]))
                probe--;
            if (probe > start + POROOL_CHUNK_CHARS / 2) end = probe;
        }
        int clen = end - start;
        if (clen <= 0) break;
        chunks[n] = (char *)malloc((size_t)clen + 1);
        if (!chunks[n]) break;
        memcpy(chunks[n], text + start, (size_t)clen);
        chunks[n][clen] = '\0';
        n++;
        start = end - POROOL_CHUNK_OVERLAP;
        if (start <= 0 || end >= len) break;
    }
    *out_n = n;
    return chunks;
}

static void p_free_chunks(char **chunks, int n)
{
    if (!chunks) return;
    for (int i = 0; i < n; i++) free(chunks[i]);
    free(chunks);
}

/* Mean-pool token vectors for one chunk; returns caller-owned float[dim]. */
static float *p_embed_chunk(const char *text)
{
    int d = sk_get_dim();
    uint32_t *ids = NULL;
    int n = 0;
    if (ve_process_text(text, &ids, &n) != VE_OK || n == 0) {
        ve_free_ids(ids);
        return (float *)calloc((size_t)d, sizeof(float));
    }
    float *vec  = (float *)calloc((size_t)d, sizeof(float));
    float *slot = (float *)malloc((size_t)d * sizeof(float));
    if (!vec || !slot) { free(vec); free(slot); ve_free_ids(ids); return NULL; }
    int added = 0;
    for (int i = 0; i < n; i++) {
        if (ve_get_vector(ids[i], slot, d) == VE_OK) {
            for (int j = 0; j < d; j++) vec[j] += slot[j];
            added++;
        }
    }
    free(slot);
    ve_free_ids(ids);
    if (added > 0) {
        float inv = 1.0f / (float)added;
        for (int j = 0; j < d; j++) vec[j] *= inv;
    }
    return vec;
}

/* ── Ingest ──────────────────────────────────────────────────────────────────*/

POROOL_API int porool_ingest_with_meta(const char *file_path,
                                        const char *db, const char *table,
                                        const ChunkMeta *meta)
{
    if (!file_path || !db || !table || !g_ready) return -1;

    /* 1. Extract plain text */
    char *raw = porool_extract(file_path);
    if (!raw) return -2;

    p_normalize(raw);

    /* 2. Chunk */
    int nc = 0;
    char **chunks = p_chunk_text(raw, &nc);
    free(raw);
    if (!chunks || nc == 0) { p_free_chunks(chunks, nc); return -3; }

    ensure_db_subdir(g_cfg.data_dir, db);

    char cl[256];
    snprintf(cl, sizeof(cl), "%s.%s", db, table);
    int d = sk_get_dim();

    /* 3. Load existing rows for append support */
    tde_handle_t old_tbl = tde_open_odat(cl);
    int old_n = old_tbl ? tde_row_count(old_tbl) : 0;

    /* Deduplication: skip if file_path is already present in this table */
    if (old_tbl) {
        for (int i = 0; i < old_n; i++) {
            char *src = fetch_string(old_tbl, i, COL_SOURCE);
            int dup = src && strcmp(src, file_path) == 0;
            free(src);
            if (dup) {
                tde_close(old_tbl);
                p_free_chunks(chunks, nc);
                return 1;
            }
        }
    }

    /* 4. Build new ODAT: existing rows + new chunks */
    const char *cols[] = {
        "text", "source", "chunk_id",
        "concept", "section", "type",
        "tags", "importance", "related_concepts"
    };
    tde_handle_t new_tbl = tde_create(cols, ODAT_NCOLS);
    if (!new_tbl) {
        if (old_tbl) tde_close(old_tbl);
        p_free_chunks(chunks, nc);
        return -5;
    }

    /* Copy existing rows (older tables may have fewer cols — fetch returns NULL safely) */
    for (int i = 0; i < old_n; i++) {
        char *f[ODAT_NCOLS];
        for (int c = 0; c < ODAT_NCOLS; c++) f[c] = fetch_string(old_tbl, i, c);
        tde_handle_t row = tde_row_begin(new_tbl);
        if (row) {
            for (int c = 0; c < ODAT_NCOLS; c++)
                tde_row_set_string(row, c, f[c] ? f[c] : "");
            tde_row_commit(row);
        }
        for (int c = 0; c < ODAT_NCOLS; c++) free(f[c]);
    }
    if (old_tbl) { tde_close(old_tbl); old_tbl = NULL; }

    /* Meta field helpers — empty string when not provided */
    const char *m_concept  = (meta && meta->concept)          ? meta->concept          : "";
    const char *m_section  = (meta && meta->section)          ? meta->section          : "";
    const char *m_type     = (meta && meta->type)             ? meta->type             : "";
    const char *m_tags     = (meta && meta->tags)             ? meta->tags             : "";
    const char *m_imp      = (meta && meta->importance)       ? meta->importance       : "";
    const char *m_related  = (meta && meta->related_concepts) ? meta->related_concepts : "";

    for (int i = 0; i < nc; i++) {
        char chunk_id[128];
        snprintf(chunk_id, sizeof(chunk_id), "%s_%s_%04d", db, table, old_n + i);
        tde_handle_t row = tde_row_begin(new_tbl);
        if (row) {
            tde_row_set_string(row, COL_TEXT,             chunks[i]);
            tde_row_set_string(row, COL_SOURCE,           file_path);
            tde_row_set_string(row, COL_CHUNK_ID,         chunk_id);
            tde_row_set_string(row, COL_CONCEPT,          m_concept);
            tde_row_set_string(row, COL_SECTION,          m_section);
            tde_row_set_string(row, COL_TYPE,             m_type);
            tde_row_set_string(row, COL_TAGS,             m_tags);
            tde_row_set_string(row, COL_IMPORTANCE,       m_imp);
            tde_row_set_string(row, COL_RELATED_CONCEPTS, m_related);
            tde_row_commit(row);
        }
    }

    if (tde_save_logical(new_tbl, cl) != TDE_OK) {
        tde_close(new_tbl);
        p_free_chunks(chunks, nc);
        return -6;
    }
    tde_close(new_tbl);

    /* 5. Build combined vector store */
    int total = old_n + nc;
    float *vecs = (float *)calloc((size_t)total * (size_t)d, sizeof(float));
    if (!vecs) { p_free_chunks(chunks, nc); return -7; }

    if (old_n > 0) {
        tde_handle_t old_ov = tde_open_ovec(cl);
        if (old_ov) {
            uint32_t *ids = (uint32_t *)malloc((size_t)old_n * sizeof(uint32_t));
            if (ids) {
                for (int i = 0; i < old_n; i++) ids[i] = (uint32_t)i;
                uint32_t dim_out = 0;
                tde_vector_get_batch(old_ov, ids, old_n, vecs, &dim_out);
                free(ids);
            }
            tde_close(old_ov);
        }
    }

    for (int i = 0; i < nc; i++) {
        float *v = p_embed_chunk(chunks[i]);
        if (v) {
            memcpy(vecs + (size_t)(old_n + i) * (size_t)d, v,
                   (size_t)d * sizeof(float));
            free(v);
        }
    }
    p_free_chunks(chunks, nc);

    int rc = tde_build_vectors_logical(cl, vecs, total, (uint32_t)d);
    free(vecs);
    if (rc != TDE_OK) return -8;

    return 0;
}

POROOL_API int porool_ingest(const char *file_path,
                              const char *db, const char *table)
{
    return porool_ingest_with_meta(file_path, db, table, NULL);
}

/* ── Multi-table retrieval ───────────────────────────────────────────────────*/

POROOL_API SearchResult *porool_retrieve_from(float      *query_vector,
                                               const char *db,
                                               const char *table_name,
                                               int         top_k,
                                               int        *result_count)
{
    if (!query_vector || !db || top_k <= 0 || !result_count || !g_ready) return NULL;
    *result_count = 0;

    int d = sk_get_dim();

    /* Build list of logical names to query */
#define MAX_TABLES 64
    char tables[MAX_TABLES][256];
    int  ntables = 0;

    if (table_name) {
        snprintf(tables[0], 256, "%.127s.%.127s", db, table_name);
        ntables = 1;
    } else {
        ntables = registry_read(db, tables, MAX_TABLES);
        if (ntables == 0) return NULL;
    }

    int total_cap = top_k * ntables;
    SearchResult *all = (SearchResult *)calloc((size_t)total_cap,
                                               sizeof(SearchResult));
    if (!all) return NULL;
    int total_found = 0;

    uint32_t *ids    = (uint32_t *)malloc((size_t)top_k * sizeof(uint32_t));
    float    *scores = (float *)   malloc((size_t)top_k * sizeof(float));
    if (!ids || !scores) { free(ids); free(scores); free(all); return NULL; }

    for (int t = 0; t < ntables && total_found < total_cap; t++) {
        tde_handle_t chunks_h = tde_open_odat(tables[t]);
        tde_handle_t vecs_h   = tde_open_ovec(tables[t]);
        if (!chunks_h || !vecs_h) {
            if (chunks_h) tde_close(chunks_h);
            if (vecs_h)   tde_close(vecs_h);
            continue;
        }

        int found = tde_vector_search_topk(vecs_h, query_vector,
                                            (uint32_t)d, (uint32_t)top_k,
                                            ids, scores);
        if (found > 0) {
            int space = total_cap - total_found;
            int take  = found < space ? found : space;
            for (int i = 0; i < take; i++) {
                all[total_found + i].id    = ids[i];
                all[total_found + i].score = scores[i];
                fill_result_meta(&all[total_found + i], chunks_h, (int)ids[i]);
            }
            total_found += take;
        }

        tde_close(chunks_h);
        tde_close(vecs_h);
    }

    free(ids);
    free(scores);

    if (total_found == 0) { free(all); return NULL; }

    /* Sort merged results best-first and trim to top_k */
    qsort(all, (size_t)total_found, sizeof(SearchResult), cmp_results_desc);
    if (total_found > top_k) {
        for (int i = top_k; i < total_found; i++)
            free_result_fields(&all[i]);
        total_found = top_k;
    }

    *result_count = total_found;
    return all;
#undef MAX_TABLES
}

/* ── All-in-one with explicit table ──────────────────────────────────────────*/

POROOL_API char *porool_query_from(const char *query,
                                    const char *db,
                                    const char *table_name,
                                    int top_k, int max_chars)
{
    if (!query || !db || !g_ready) return NULL;
    if (top_k     <= 0) top_k     = g_cfg.top_k_default;
    if (max_chars <= 0) max_chars = g_cfg.max_context_chars;

    float *emb = NULL;
    int    dim = 0;
    if (porool_embed_query(query, &emb, &dim) != 0) return NULL;

    int           n   = 0;
    SearchResult *res = porool_retrieve_from(emb, db, table_name, top_k, &n);
    free(emb);
    if (!res) return NULL;

    rerank_internal(res, n, query);
    char *ctx = porool_build_context(res, n, max_chars);
    porool_free_results(res, n);
    return ctx;
}

/* ── Target-string API ───────────────────────────────────────────────────────*/

POROOL_API SearchResult *porool_retrieve_target(float      *query_vector,
                                                 const char *target,
                                                 int         top_k,
                                                 int        *result_count)
{
    if (!query_vector || top_k <= 0 || !result_count || !g_ready) return NULL;
    char db[256], tbl[256];
    int mode = parse_target(target, db, sizeof(db), tbl, sizeof(tbl));

    if (mode == 0) return porool_retrieve_from(query_vector, db, tbl,  top_k, result_count);
    if (mode == 1) return porool_retrieve_from(query_vector, db, NULL, top_k, result_count);

    /* mode 2: all databases */
#define ALL_MAX (64 * 4)
    char tables[ALL_MAX][256];
    int ntables = registry_read_all(tables, ALL_MAX);
    if (ntables == 0) { *result_count = 0; return NULL; }

    int d = sk_get_dim();
    int cap = top_k * ntables;
    SearchResult *all = (SearchResult *)calloc((size_t)cap, sizeof(SearchResult));
    if (!all) { *result_count = 0; return NULL; }
    uint32_t *ids    = (uint32_t *)malloc((size_t)top_k * sizeof(uint32_t));
    float    *scores = (float *)   malloc((size_t)top_k * sizeof(float));
    if (!ids || !scores) { free(ids); free(scores); free(all); *result_count = 0; return NULL; }

    int total = 0;
    for (int t = 0; t < ntables && total < cap; t++) {
        tde_handle_t ch = tde_open_odat(tables[t]);
        tde_handle_t vh = tde_open_ovec(tables[t]);
        if (!ch || !vh) { if (ch) tde_close(ch); if (vh) tde_close(vh); continue; }
        int found = tde_vector_search_topk(vh, query_vector, (uint32_t)d, (uint32_t)top_k, ids, scores);
        int space = cap - total;
        int take  = found < space ? found : space;
        for (int i = 0; i < take; i++) {
            all[total + i].id    = ids[i];
            all[total + i].score = scores[i];
            fill_result_meta(&all[total + i], ch, (int)ids[i]);
        }
        total += take;
        tde_close(ch); tde_close(vh);
    }
    free(ids); free(scores);
    if (total == 0) { free(all); *result_count = 0; return NULL; }

    qsort(all, (size_t)total, sizeof(SearchResult), cmp_results_desc);
    for (int i = top_k; i < total; i++) free_result_fields(&all[i]);
    *result_count = total < top_k ? total : top_k;
    return all;
#undef ALL_MAX
}

POROOL_API char *porool_query_target(const char *query, const char *target,
                                      int top_k, int max_chars)
{
    if (!query || !g_ready) return NULL;
    if (top_k     <= 0) top_k     = g_cfg.top_k_default;
    if (max_chars <= 0) max_chars = g_cfg.max_context_chars;

    float *emb = NULL; int dim = 0;
    if (porool_embed_query(query, &emb, &dim) != 0) return NULL;

    int n = 0;
    SearchResult *res = porool_retrieve_target(emb, target, top_k, &n);
    free(emb);
    if (!res) return NULL;

    rerank_internal(res, n, query);
    char *ctx = porool_build_context(res, n, max_chars);
    porool_free_results(res, n);
    return ctx;
}

/* ── Phrasing API ────────────────────────────────────────────────────────────*/

POROOL_API void porool_phrasing_reload(void)
{
    if (!g_ready) return;
    phrases_reload();
}

/*
 * Add a pattern to the phrasing table and reload the cache.
 * is_prefix=1 → query_prefixes table; is_prefix=0 → chunk_markers table.
 * Returns 0 on success, -1 if duplicate, -2 on error.
 */
POROOL_API int porool_phrasing_add(const char *pattern, int is_prefix)
{
    if (!pattern || !pattern[0] || !g_ready) return -2;
    const char *logical = is_prefix ? PHRASE_PREFIXES_LOGICAL : PHRASE_MARKERS_LOGICAL;
    char **cache        = is_prefix ? g_prefixes : g_markers;
    int   *count        = is_prefix ? &g_nprefixes : &g_nmarkers;

    for (int i = 0; i < *count; i++)
        if (cache[i] && strcmp(cache[i], pattern) == 0) return -1;

    tde_handle_t old   = tde_open_odat(logical);
    int          old_n = old ? tde_row_count(old) : 0;
    const char  *cols[] = { "pattern" };
    tde_handle_t tbl   = tde_create(cols, 1);
    if (!tbl) { if (old) tde_close(old); return -2; }

    for (int i = 0; i < old_n; i++) {
        int need = tde_get_string(old, i, 0, NULL, 0);
        if (need <= 0) continue;
        char *s = (char *)malloc((size_t)need);
        if (!s) continue;
        if (tde_get_string(old, i, 0, s, need) > 0) {
            tde_handle_t row = tde_row_begin(tbl);
            if (row) { tde_row_set_string(row, 0, s); tde_row_commit(row); }
        }
        free(s);
    }
    if (old) tde_close(old);

    tde_handle_t row = tde_row_begin(tbl);
    if (row) { tde_row_set_string(row, 0, pattern); tde_row_commit(row); }

    if (tde_save_logical(tbl, logical) != TDE_OK) { tde_close(tbl); return -2; }
    tde_close(tbl);
    phrases_reload();
    return 0;
}

/* ── Synonym API ─────────────────────────────────────────────────────────────*/

/*
 * Register a query-expansion synonym: occurrences of `from` (whole-word,
 * case-insensitive) in a query will cause `to` to be appended before embedding.
 * Returns 0 on success, -1 if `from` already registered, -2 on bad args,
 * -3 if the synonym table is full.
 */
POROOL_API int porool_synonym_add(const char *from, const char *to)
{
    if (!from || !from[0] || !to || !to[0]) return -2;
    synonyms_ensure_defaults();
    for (int i = 0; i < g_n_synonyms; i++)
        if (strcmp(g_synonyms[i].from, from) == 0) return -1;
    if (g_n_synonyms >= MAX_SYNONYMS) return -3;
    snprintf(g_synonyms[g_n_synonyms].from, 64,  "%.63s",  from);
    snprintf(g_synonyms[g_n_synonyms].to,   128, "%.127s", to);
    g_n_synonyms++;
    return 0;
}

/* ── Memory management ───────────────────────────────────────────────────────*/

POROOL_API void porool_free(char *ptr)
{
    free(ptr);
}

POROOL_API void porool_free_results(SearchResult *results, int count)
{
    if (!results) return;
    for (int i = 0; i < count; i++)
        free_result_fields(&results[i]);
    free(results);
}
