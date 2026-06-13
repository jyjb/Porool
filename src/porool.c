/*
 * porool.c — Context Retrieval Engine (Porool.dll)
 *
 * Pure C99.  Orchestrates Sorkuvai.dll (embedding) and Tharavu.dll (vector
 * search + chunk storage) into a single query pipeline.
 *
 * Multiple porool_t instances may coexist.  Each instance owns:
 *   - its own PoroolConfig (data_dir, table names, scoring weights)
 *   - its own chunk/vector THARAVU handles
 *   - its own phrasing cache
 *
 * SORKUVAI (embedding) and THARAVU (storage) are pre-configured by the
 * caller before porool_create().  POROOL does not call ve_init() or
 * tde_config_load() — those are the initiator's responsibility.
 *
 * Init order (caller's responsibility):
 *   tde_set_base_path() → ve_init() → porool_create() → open chunk + vector tables
 */

#ifndef POROOL_EXPORTS
#  define POROOL_EXPORTS
#endif
#include "include/porool.h"
#include "tharavu_dll.h"
#include "slispmanager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <ctype.h>
#include <stdatomic.h>
#include "include/porool_extract.h"

/* ── Sorkuvai.dll forward declarations ──────────────────────────────────────*/
#define VE_OK        0
#define VE_ERR_INVAL -4

#ifdef _WIN32
#  include <windows.h>
#  define SK_CALL __cdecl
#  define TDE_CALL __cdecl
#else
#  define SK_CALL
#  define TDE_CALL
#endif

/* ve_init/ve_cleanup removed 2026-05-30 — caller manages SORKUVAI lifecycle */
extern int      SK_CALL ve_process_text(const char *text, uint32_t **token_ids, int *count);
extern void     SK_CALL ve_free_ids(uint32_t *ids);
extern int      SK_CALL ve_get_vector(uint32_t token_id, float *out_vec, int dim);
extern int      SK_CALL sk_get_dim(void);

/* ── Tharavu.dll forward declarations ───────────────────────────────────────*/
typedef void *tde_handle_t;

#define TDE_OK       0

extern const char  *TDE_CALL tde_get_base_path(void);
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
extern const float *TDE_CALL tde_vector_get_ptr(tde_handle_t h,
                                                uint32_t     row_id,
                                                uint32_t    *out_dim);

/* ── Config struct ───────────────────────────────────────────────────────────*/

#define MAX_SOURCE_WEIGHTS 32

typedef struct { char name[64]; float weight; } source_weight_t;

typedef struct {
    char  data_dir[512];
    char  phrasing_root[512]; /* explicit dir for phrasing tables; "" = old logical-name path */
    char  vocab_name[256];
    char  vocab_group[128];
    char  vocab_first_lang[128];
    char  chunks_table[256];
    char  vectors_table[256];
    int   top_k_default;
    int   max_context_chars;
    float optimal_chunk_len;
    float length_penalty_sigma;
    float default_source_weight;
    float w1, w2, w3, w4, w5;
    float ctx_min_score;
    source_weight_t source_weights[MAX_SOURCE_WEIGHTS];
    int             source_weight_count;
} PoroolConfig;

/* ── Phrasing cache constants ────────────────────────────────────────────────*/

#define PHRASE_PREFIXES_LOGICAL "phrasing.query_prefixes"
#define PHRASE_MARKERS_LOGICAL  "phrasing.chunk_markers"
#define MAX_PHRASES 128

/* Build explicit filesystem path for a phrasing ODAT.
 * "phrasing.query_prefixes" + root → "{root}/query_prefixes.odat" */
static void phrase_explicit_path(const char *phrasing_root,
                                  const char *logical,
                                  char *out, size_t cap)
{
    const char *last_dot = strrchr(logical, '.');
    const char *table    = last_dot ? last_dot + 1 : logical;
    snprintf(out, cap, "%s/%s.odat", phrasing_root, table);
}

/* ── Instance struct ─────────────────────────────────────────────────────────*/

struct porool_s {
    PoroolConfig  cfg;
    tde_handle_t  chunks;
    tde_handle_t  vectors;
    int           ready;
    char         *prefixes[MAX_PHRASES];
    int           nprefixes;
    char         *markers[MAX_PHRASES];
    int           nmarkers;
#ifdef _WIN32
    SRWLOCK       phrase_lock;
#else
    pthread_rwlock_t phrase_lock;
#endif
};

/* ── Per-instance phrase lock macros ─────────────────────────────────────────*/

#ifdef _WIN32
#  define PHRASE_WLOCK(p)   AcquireSRWLockExclusive(&(p)->phrase_lock)
#  define PHRASE_WUNLOCK(p) ReleaseSRWLockExclusive(&(p)->phrase_lock)
#  define PHRASE_RLOCK(p)   AcquireSRWLockShared(&(p)->phrase_lock)
#  define PHRASE_RUNLOCK(p) ReleaseSRWLockShared(&(p)->phrase_lock)
#else
#  include <pthread.h>
#  define PHRASE_WLOCK(p)   pthread_rwlock_wrlock(&(p)->phrase_lock)
#  define PHRASE_WUNLOCK(p) pthread_rwlock_unlock(&(p)->phrase_lock)
#  define PHRASE_RLOCK(p)   pthread_rwlock_rdlock(&(p)->phrase_lock)
#  define PHRASE_RUNLOCK(p) pthread_rwlock_unlock(&(p)->phrase_lock)
#endif

#ifndef _WIN32
#  include <pthread.h>
#endif

/* ── Directory helper ───────────────────────────────────────────────────────*/

#ifdef _WIN32
#  include <direct.h>
static void p_mkdirs(const char *path) {
    char tmp[640]; size_t len = strlen(path);
    if (!len || len >= sizeof(tmp)) return;
    memcpy(tmp, path, len + 1);
    for (size_t i = 1; i <= len; i++) {
        if (tmp[i] == '/' || tmp[i] == '\\' || tmp[i] == '\0') {
            char c = tmp[i]; tmp[i] = '\0'; _mkdir(tmp); tmp[i] = c;
        }
    }
}
#  define p_mkdir(p) p_mkdirs(p)
#else
#  include <glob.h>
#  include <sys/stat.h>
static void p_mkdirs(const char *path) {
    char tmp[640]; size_t len = strlen(path);
    if (!len || len >= sizeof(tmp)) return;
    memcpy(tmp, path, len + 1);
    for (size_t i = 1; i <= len; i++) {
        if (tmp[i] == '/' || tmp[i] == '\0') {
            char c = tmp[i]; tmp[i] = '\0'; mkdir(tmp, 0755); tmp[i] = c;
        }
    }
}
#  define p_mkdir(p) p_mkdirs(p)
#endif

static void ensure_db_subdir(const char *data_dir, const char *db)
{
    p_mkdir(data_dir);
    char path[640];
    snprintf(path, sizeof(path), "%s/%s", data_dir, db);
    p_mkdir(path);
}

/* ── SXP writer (default file) ──────────────────────────────────────────────*/

static void write_default_sxp(const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp,
        "; Porool configuration file\n"
        "; host must call tde_config_load() with tharavu.sxp before porool_create()\n"
        "(porool\n"
        "  (data_dir              \"./data\")\n"
        "  (chunks_table          \"porool.chunks\")\n"
        "  (vectors_table         \"porool.chunks\")\n"
        "  (top_k_default         10)\n"
        "  (max_context_chars     2000)\n"
        "  (ctx_min_score         0.0)\n"
        "  (optimal_chunk_len     500.0)\n"
        "  (length_penalty_sigma  200.0)\n"
        "  (default_source_weight 1.0)\n"
        "  (vocab_name            \"general.english\")\n"
        "  ; Composite score = cosine*w1 + length_score*w2 + source_score*w3\n"
        "  ;   + definition_signal*w4  (only when query is definitional)\n"
        "  ; w1+w2+w3 should sum to 1.0; w4 is an additive bonus (0 = disabled)\n"
        "  (scoring (w1 0.60) (w2 0.20) (w3 0.20) (w4 0.15) (w5 0.20))\n"
        "  ; Per-source priority multipliers. Range [0.0, 2.0]; values outside are clamped.\n"
        "  ; Add entries as (source_name weight) inside source_weights.\n"
        "  (source_weights))\n"
    );
    fclose(fp);
}

/* ── SXP parser ─────────────────────────────────────────────────────────────*/

static void read_config_node(PoroolConfig *cfg, slm_node_t *node)
{
    const char *s;

    s = slm_attr(node, "data_dir");
    if (s) snprintf(cfg->data_dir, sizeof(cfg->data_dir), "%.511s", s);

    s = slm_attr(node, "chunks_table");
    if (s) snprintf(cfg->chunks_table, sizeof(cfg->chunks_table), "%.255s", s);

    s = slm_attr(node, "vectors_table");
    if (s) snprintf(cfg->vectors_table, sizeof(cfg->vectors_table), "%.255s", s);

    s = slm_attr(node, "vocab_name");
    if (s) snprintf(cfg->vocab_name, sizeof(cfg->vocab_name), "%.255s", s);

    cfg->top_k_default         = (int)  slm_attr_i(node, "top_k_default",         cfg->top_k_default);
    cfg->max_context_chars     = (int)  slm_attr_i(node, "max_context_chars",      cfg->max_context_chars);
    cfg->ctx_min_score         = (float)slm_attr_f(node, "ctx_min_score",          cfg->ctx_min_score);
    cfg->optimal_chunk_len     = (float)slm_attr_f(node, "optimal_chunk_len",      cfg->optimal_chunk_len);
    cfg->length_penalty_sigma  = (float)slm_attr_f(node, "length_penalty_sigma",   cfg->length_penalty_sigma);
    cfg->default_source_weight = (float)slm_attr_f(node, "default_source_weight",  cfg->default_source_weight);

    slm_node_t *scoring = slm_find(node, "scoring");
    if (scoring) {
        cfg->w1 = (float)slm_attr_f(scoring, "w1", cfg->w1);
        cfg->w2 = (float)slm_attr_f(scoring, "w2", cfg->w2);
        cfg->w3 = (float)slm_attr_f(scoring, "w3", cfg->w3);
        cfg->w4 = (float)slm_attr_f(scoring, "w4", cfg->w4);
        cfg->w5 = (float)slm_attr_f(scoring, "w5", cfg->w5);
    }

    slm_node_t *sw = slm_find(node, "source_weights");
    if (sw) {
        cfg->source_weight_count = 0;
        slm_node_t *c = slm_first_child(sw);
        if (c) c = slm_next(c); /* skip "source_weights" tag symbol */
        while (c && cfg->source_weight_count < MAX_SOURCE_WEIGHTS) {
            if (slm_type(c) == SLM_LIST) {
                slm_node_t *k = slm_first_child(c);
                slm_node_t *v = k ? slm_next(k) : NULL;
                if (k && v) {
                    source_weight_t *entry = &cfg->source_weights[cfg->source_weight_count++];
                    snprintf(entry->name, sizeof(entry->name), "%.63s", slm_value(k));
                    entry->weight = (float)slm_number(v);
                }
            }
            c = slm_next(c);
        }
    }
}

/* Load config from path into cfg.  If segment_name is non-NULL, read global
 * fields from the root node first, then override with the named segment node. */
static int load_config(PoroolConfig *cfg, const char *path,
                        const char *segment_name)
{
    strncpy(cfg->data_dir,      "./data",        sizeof(cfg->data_dir)      - 1);
    strncpy(cfg->chunks_table,  "porool.chunks", sizeof(cfg->chunks_table)  - 1);
    strncpy(cfg->vectors_table, "porool.chunks", sizeof(cfg->vectors_table) - 1);
    cfg->vocab_name[0]          = '\0';
    cfg->vocab_group[0]         = '\0';
    cfg->vocab_first_lang[0]    = '\0';
    cfg->top_k_default          = 10;
    cfg->max_context_chars      = 2000;
    cfg->optimal_chunk_len      = 500.0f;
    cfg->length_penalty_sigma   = 200.0f;
    cfg->default_source_weight  = 1.0f;
    cfg->w1 = 0.6f; cfg->w2 = 0.2f; cfg->w3 = 0.2f;
    cfg->w4 = 0.15f; cfg->w5 = 0.20f;
    cfg->ctx_min_score          = 0.0f;
    cfg->source_weight_count    = 0;

    FILE *fp = fopen(path, "r");
    if (!fp) {
        write_default_sxp(path);
        strncpy(cfg->vocab_name, "porool.vocab", sizeof(cfg->vocab_name) - 1);
        return 0;
    }
    fclose(fp);

    slm_node_t *root = slm_load(path);
    if (!root) {
        fprintf(stderr, "porool: error: failed to parse config '%s'\n", path);
        return -1;
    }

    read_config_node(cfg, root);

    if (segment_name && *segment_name) {
        slm_node_t *seg = slm_find(root, segment_name);
        if (!seg) {
            fprintf(stderr, "porool: warning: segment '%s' not found in '%s'; using root config\n",
                    segment_name, path);
        } else {
            read_config_node(cfg, seg);
        }
    }

    slm_free(root);

    if (!cfg->vocab_name[0]) {
        fprintf(stderr,
            "porool: warning: no 'vocab_name' in %s; defaulting to 'porool.vocab'\n", path);
        strncpy(cfg->vocab_name, "porool.vocab", sizeof(cfg->vocab_name) - 1);
    }

    if (cfg->w1 + cfg->w2 + cfg->w3 > 1.001f)
        fprintf(stderr,
            "porool: warning: w1+w2+w3 = %.3f; downstream normalisation assumes sum <= 1.0\n",
            cfg->w1 + cfg->w2 + cfg->w3);
    return 0;
}

/* ── Phrasing helpers ────────────────────────────────────────────────────────*/

static const char *k_def_prefixes[] = {
    "what is", "what are", "what's", "define ", "explain ", "describe "
};
static const char *k_def_markers[] = {
    " is a ", " is an ", " are a ", " refers to ", " defined as ",
    " is defined ", " represents a ", " represent a ", " represent an ",
    " known as ", " stands for "
};

static void phrases_free_arrays(char **arr, int *count)
{
    for (int i = 0; i < *count; i++) { free(arr[i]); arr[i] = NULL; }
    *count = 0;
}

static void phrases_load_from(const char *phrasing_root,
                               const char *logical,
                               char **arr, int *count)
{
    *count = 0;
    tde_handle_t h;
    if (phrasing_root && phrasing_root[0]) {
        char path[512 + 64];
        phrase_explicit_path(phrasing_root, logical, path, sizeof path);
        h = tde_open(path);
    } else {
        h = tde_open_odat(logical);
    }
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

static void phrases_ensure_table(const char *phrasing_root,
                                   const char *data_dir,
                                   const char *logical,
                                   const char **defaults, int ndefaults)
{
    const char *cols[] = { "pattern" };
    if (phrasing_root && phrasing_root[0]) {
        /* Explicit path — bypass THARAVU logical-name resolver */
        char path[512 + 64];
        phrase_explicit_path(phrasing_root, logical, path, sizeof path);
        tde_handle_t h = tde_open(path);
        if (h) { tde_close(h); return; }
        tde_handle_t tbl = tde_create(cols, 1);
        if (!tbl) return;
        for (int i = 0; i < ndefaults; i++) {
            tde_handle_t row = tde_row_begin(tbl);
            if (!row) continue;
            tde_row_set_string(row, 0, defaults[i]);
            tde_row_commit(row);
        }
        p_mkdir(phrasing_root);
        tde_save(tbl, path);
        tde_close(tbl);
        return;
    }
    /* Old behaviour: THARAVU logical-name resolver (resolves under g_base_path) */
    tde_handle_t h = tde_open_odat(logical);
    if (h) { tde_close(h); return; }
    tde_handle_t tbl = tde_create(cols, 1);
    if (!tbl) return;
    for (int i = 0; i < ndefaults; i++) {
        tde_handle_t row = tde_row_begin(tbl);
        if (!row) continue;
        tde_row_set_string(row, 0, defaults[i]);
        tde_row_commit(row);
    }
    ensure_db_subdir(data_dir, "phrasing");
    tde_save_logical(tbl, logical);
    tde_close(tbl);
}

static void phrases_reload(porool_t *p)
{
    PHRASE_WLOCK(p);
    phrases_free_arrays(p->prefixes, &p->nprefixes);
    phrases_free_arrays(p->markers,  &p->nmarkers);
    phrases_load_from(p->cfg.phrasing_root, PHRASE_PREFIXES_LOGICAL, p->prefixes, &p->nprefixes);
    phrases_load_from(p->cfg.phrasing_root, PHRASE_MARKERS_LOGICAL,  p->markers,  &p->nmarkers);
    PHRASE_WUNLOCK(p);
}

/* ── Init / destroy ──────────────────────────────────────────────────────────*/

POROOL_API void porool_config_defaults(porool_config_t *cfg) {
    if (!cfg) return;
    cfg->segment       = NULL;
    cfg->top_k         = 10;
    cfg->min_score     = 0.10f;
    cfg->max_chars     = 2000;
    cfg->w_cosine      = 0.60f;
    cfg->w_length      = 0.20f;
    cfg->w_source      = 0.20f;
    cfg->w_def_bonus   = 0.15f;
    cfg->w_overlap     = 0.20f;
    cfg->phrasing_root = NULL;  /* NULL = old logical-name path (knowledge/phrasing/) */
}

POROOL_API porool_t *porool_create(const porool_config_t *cfg)
{
    /* Caller is responsible for pre-configuring THARAVU (tde_set_base_path)
     * and SORKUVAI (ve_init) before calling porool_create. */
    porool_config_t defaults;
    porool_config_defaults(&defaults);
    const porool_config_t *c = cfg ? cfg : &defaults;

    porool_t *p = (porool_t *)calloc(1, sizeof(porool_t));
    if (!p) return NULL;

    /* Fill internal config from explicit values — no config file read. */
    const char *base = tde_get_base_path();
    strncpy(p->cfg.data_dir,     base ? base : "./knowledge", sizeof(p->cfg.data_dir)     - 1);
    strncpy(p->cfg.chunks_table,  "porool.chunks", sizeof(p->cfg.chunks_table)  - 1);
    strncpy(p->cfg.vectors_table, "porool.chunks", sizeof(p->cfg.vectors_table) - 1);
    strncpy(p->cfg.vocab_name,    "general.english", sizeof(p->cfg.vocab_name)  - 1);
    p->cfg.top_k_default         = c->top_k      > 0    ? c->top_k      : 10;
    p->cfg.max_context_chars     = c->max_chars   > 0    ? c->max_chars   : 2000;
    p->cfg.ctx_min_score         = c->min_score;
    p->cfg.optimal_chunk_len     = 500.0f;
    p->cfg.length_penalty_sigma  = 200.0f;
    p->cfg.default_source_weight = 1.0f;
    p->cfg.w1 = c->w_cosine    > 0.0f ? c->w_cosine    : 0.60f;
    p->cfg.w2 = c->w_length    > 0.0f ? c->w_length    : 0.20f;
    p->cfg.w3 = c->w_source    > 0.0f ? c->w_source    : 0.20f;
    p->cfg.w4 = c->w_def_bonus > 0.0f ? c->w_def_bonus : 0.15f;
    p->cfg.w5 = c->w_overlap   > 0.0f ? c->w_overlap   : 0.20f;
    if (c->phrasing_root && c->phrasing_root[0] != '\0')
        strncpy(p->cfg.phrasing_root, c->phrasing_root, sizeof(p->cfg.phrasing_root) - 1);
    /* else: phrasing_root stays zero-filled (calloc) — old logical-name path preserved */

#ifdef _WIN32
    InitializeSRWLock(&p->phrase_lock);
#else
    pthread_rwlock_init(&p->phrase_lock, NULL);
#endif

    /* Open default chunk/vector tables only when the file already exists.
     * tde_open_odat triggers THARAVU's ensure_dir_recursive even on a
     * read-only path, creating a phantom knowledge/porool/ directory when
     * no chunks have been ingested yet. */
    {
        char probe[640];
        snprintf(probe, sizeof probe, "%s/porool/chunks.odat", p->cfg.data_dir);
        FILE *f = fopen(probe, "rb");
        if (f) {
            fclose(f);
            p->chunks  = tde_open_odat(p->cfg.chunks_table);
            p->vectors = p->chunks ? tde_open_ovec(p->cfg.vectors_table) : NULL;
        }
    }

    /* Ensure phrasing tables exist then load into instance cache. */
    phrases_ensure_table(p->cfg.phrasing_root, p->cfg.data_dir, PHRASE_PREFIXES_LOGICAL,
                         k_def_prefixes,
                         (int)(sizeof(k_def_prefixes)/sizeof(k_def_prefixes[0])));
    phrases_ensure_table(p->cfg.phrasing_root, p->cfg.data_dir, PHRASE_MARKERS_LOGICAL,
                         k_def_markers,
                         (int)(sizeof(k_def_markers)/sizeof(k_def_markers[0])));
    phrases_reload(p);

    p->ready = 1;
    return p;
}

POROOL_API void porool_destroy(porool_t *p)
{
    if (!p) return;

    if (p->vectors) { tde_close(p->vectors); p->vectors = NULL; }
    if (p->chunks)  { tde_close(p->chunks);  p->chunks  = NULL; }

    PHRASE_WLOCK(p);
    phrases_free_arrays(p->prefixes, &p->nprefixes);
    phrases_free_arrays(p->markers,  &p->nmarkers);
    PHRASE_WUNLOCK(p);

#ifdef _WIN32
    /* No explicit destroy for SRWLOCK on Windows */
#else
    pthread_rwlock_destroy(&p->phrase_lock);
#endif

    /* SORKUVAI lifecycle is managed by the caller — no ve_cleanup here. */
    free(p);
}

POROOL_API int porool_dim(porool_t *p)
{
    return (p && p->ready) ? sk_get_dim() : 0;
}

POROOL_API int porool_refresh(porool_t *p)
{
    if (!p || !p->ready) return -1;
    if (p->vectors) { tde_close(p->vectors); p->vectors = NULL; }
    if (p->chunks)  { tde_close(p->chunks);  p->chunks  = NULL; }
    p->chunks  = tde_open_odat(p->cfg.chunks_table);
    p->vectors = p->chunks ? tde_open_ovec(p->cfg.vectors_table) : NULL;
    return 0;
}

/* ── Utility: case-insensitive substring search ──────────────────────────────*/

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

/* ── Synonym expansion (process-global) ─────────────────────────────────────*/

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
        const char *q = query;
        int hit = 0;
        while (*q && !hit) {
            if (tolower((unsigned char)*q) == (unsigned char)from[0]) {
                size_t k = 0;
                while (k < fl && q[k] &&
                       tolower((unsigned char)q[k]) == (unsigned char)from[k])
                    k++;
                if (k == fl) {
                    unsigned char pre  = (q > query) ? (unsigned char)q[-1] : 0;
                    unsigned char post = (unsigned char)q[fl];
                    if (!isalnum(pre) && !isalnum(post)) hit = 1;
                }
            }
            if (!hit) q++;
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

POROOL_API int porool_embed_query(porool_t *p, const char *query,
                                   float **embedding, int *dim)
{
    if (!p || !query || !embedding || !dim) return -1;
    *embedding = NULL; *dim = 0;
    if (!p->ready) return -1;

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

#define COL_TEXT             0
#define COL_SOURCE           1
#define COL_CHUNK_ID         2
#define COL_CONCEPT          3
#define COL_SECTION          4
#define COL_TYPE             5
#define COL_TAGS             6
#define COL_IMPORTANCE       7
#define COL_RELATED_CONCEPTS 8
#define COL_LANGUAGE         9
#define ODAT_NCOLS           10

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
    r->language         = fetch_string(h, row, COL_LANGUAGE);
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
    free(r->language);
}

POROOL_API SearchResult *porool_retrieve(porool_t *p, float *query_vector,
                                          int top_k, int *result_count)
{
    if (!p || !query_vector || top_k <= 0 || !result_count || !p->ready) return NULL;
    if (!p->vectors || !p->chunks) return NULL;
    *result_count = 0;

    int d = sk_get_dim();

    if (tde_row_count(p->chunks) > 0) {
        uint32_t stored_dim = 0;
        tde_vector_get_ptr(p->vectors, 0, &stored_dim);
        if (stored_dim != 0 && stored_dim != (uint32_t)d) return NULL;
    }

    uint32_t *ids    = (uint32_t *)malloc((size_t)top_k * sizeof(uint32_t));
    float    *scores = (float *)   malloc((size_t)top_k * sizeof(float));
    if (!ids || !scores) { free(ids); free(scores); return NULL; }

    int found = tde_vector_search_topk(p->vectors, query_vector,
                                        (uint32_t)d, (uint32_t)top_k,
                                        ids, scores);
    if (found <= 0) { free(ids); free(scores); return NULL; }

    SearchResult *res = (SearchResult *)calloc((size_t)found, sizeof(SearchResult));
    if (!res) { free(ids); free(scores); return NULL; }

    for (int i = 0; i < found; i++) {
        res[i].id    = ids[i];
        res[i].score = scores[i];
        fill_result_meta(&res[i], p->chunks, (int)ids[i]);
    }

    free(ids);
    free(scores);
    *result_count = found;
    return res;
}

/* ── Re-ranking ──────────────────────────────────────────────────────────────*/

static float source_weight_lookup(const PoroolConfig *cfg, const char *source)
{
    if (!source) return cfg->default_source_weight;
    for (int i = 0; i < cfg->source_weight_count; i++)
        if (strcmp(cfg->source_weights[i].name, source) == 0)
            return cfg->source_weights[i].weight;
    return cfg->default_source_weight;
}

static float importance_score(const char *imp)
{
    if (!imp || imp[0] == '\0') return 0.5f;
    if (strcmp(imp, "high")   == 0) return 1.0f;
    if (strcmp(imp, "medium") == 0) return 0.5f;
    if (strcmp(imp, "low")    == 0) return 0.2f;
    return 0.5f;
}

static float length_score(const PoroolConfig *cfg, const char *text)
{
    if (!text) return 0.0f;
    float len   = (float)strlen(text);
    float sigma = cfg->length_penalty_sigma > 0.0f
                  ? cfg->length_penalty_sigma : 1.0f;
    float dev = (len - cfg->optimal_chunk_len) / sigma;
    return expf(-0.5f * dev * dev);
}

static int cmp_results_desc(const void *a, const void *b)
{
    float sa = ((const SearchResult *)a)->score;
    float sb = ((const SearchResult *)b)->score;
    return (sa > sb) ? -1 : (sa < sb) ? 1 : 0;
}

static int extract_query_topic(porool_t *p, const char *q, char *out, int out_sz)
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
    PHRASE_RLOCK(p);
    int n = p->nprefixes > 0 ? p->nprefixes
                              : (int)(sizeof(k_def_prefixes)/sizeof(k_def_prefixes[0]));
    const char **list = p->nprefixes > 0 ? (const char **)p->prefixes : k_def_prefixes;
    int found = 0;
    for (int pp = 0; pp < n; pp++) {
        size_t pl = strlen(list[pp]);
        if (strncmp(buf, list[pp], pl) != 0) continue;
        const char *rest = buf + pl;
        while (*rest == ' ') rest++;
        int rlen = (int)strlen(rest);
        while (rlen > 0 && (rest[rlen-1] == '?' || rest[rlen-1] == ' ')) rlen--;
        int copy = rlen < out_sz - 1 ? rlen : out_sz - 1;
        memcpy(out, rest, (size_t)copy);
        out[copy] = '\0';
        found = out[0] != '\0' ? 1 : 0;
        break;
    }
    PHRASE_RUNLOCK(p);
    return found;
}

static float definition_content_score(porool_t *p, const char *text)
{
    if (!text) return 0.0f;
    char buf[401];
    int i = 0;
    while (i < 400 && text[i]) { buf[i] = (char)tolower((unsigned char)text[i]); i++; }
    buf[i] = '\0';
    PHRASE_RLOCK(p);
    int n          = p->nmarkers > 0 ? p->nmarkers
                                     : (int)(sizeof(k_def_markers)/sizeof(k_def_markers[0]));
    const char **list = p->nmarkers > 0 ? (const char **)p->markers : k_def_markers;
    int hits = 0;
    for (int m = 0; m < n; m++)
        if (strstr(buf, list[m])) hits++;
    PHRASE_RUNLOCK(p);
    return hits > 0 ? 1.0f : 0.0f;
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

static void rerank_internal(porool_t *p, SearchResult *results, int count,
                              const char *query)
{
    if (!results || count <= 0) return;
    float w1 = p->cfg.w1, w2 = p->cfg.w2, w3 = p->cfg.w3,
          w4 = p->cfg.w4, w5 = p->cfg.w5;
    char topic[256] = "";
    int is_def = (w4 > 0.0f) ? extract_query_topic(p, query, topic, sizeof(topic)) : 0;
    char *expanded = (w5 > 0.0f) ? synonym_expand(query) : NULL;
    const char *eq  = expanded ? expanded : query;
    for (int i = 0; i < count; i++) {
        float cosine = results[i].score;
        float lscore = length_score(&p->cfg, results[i].text);
        float sscore;
        const char *imp = results[i].importance;
        if (imp && imp[0] != '\0') {
            sscore = importance_score(imp);
        } else {
            float wsrc = source_weight_lookup(&p->cfg, results[i].source);
            sscore = wsrc > 2.0f ? 1.0f : wsrc * 0.5f;
        }
        float dscore = (is_def && ci_contains(results[i].text, topic))
                       ? definition_content_score(p, results[i].text) : 0.0f;
        float oscore = (w5 > 0.0f)
                       ? term_overlap_score(eq, results[i].text) : 0.0f;
        results[i].score = cosine*w1 + lscore*w2 + sscore*w3 + dscore*w4 + oscore*w5;
    }
    free(expanded);
    qsort(results, (size_t)count, sizeof(SearchResult), cmp_results_desc);
}

POROOL_API void porool_rerank(porool_t *p, SearchResult *results, int count)
{
    if (!p || !results || count <= 0) return;
    for (int i = 0; i < count; i++) {
        float cosine = results[i].score;
        float lscore = length_score(&p->cfg, results[i].text);
        float wsrc   = source_weight_lookup(&p->cfg, results[i].source);
        float sscore = wsrc > 2.0f ? 1.0f : wsrc * 0.5f;
        results[i].score = cosine * p->cfg.w1 + lscore * p->cfg.w2 + sscore * p->cfg.w3;
    }
    qsort(results, (size_t)count, sizeof(SearchResult), cmp_results_desc);
}

POROOL_API void porool_rerank_query(porool_t *p, SearchResult *results, int count,
                                     const char *query)
{
    if (p) rerank_internal(p, results, count, query);
}

/* ── Context builder ─────────────────────────────────────────────────────────*/

POROOL_API char *porool_build_context(porool_t *p, SearchResult *results, int count,
                                       int max_chars, float ctx_min_score)
{
    (void)p;
    if (!results || count <= 0 || max_chars <= 0) return NULL;

    size_t capacity = 4096;
    char  *ctx = (char *)malloc(capacity);
    if (!ctx) return NULL;
    ctx[0] = '\0';
    size_t used = 0;

    const char **seen = (const char **)calloc((size_t)count, sizeof(char *));
    int seen_n = 0;

    for (int i = 0; i < count; i++) {
        if (results[i].score < ctx_min_score) continue;
        if (!results[i].text) continue;

        int dup = 0;
        for (int j = 0; j < seen_n; j++)
            if (strcmp(results[i].text, seen[j]) == 0) { dup = 1; break; }
        if (dup) continue;

        const char *src  = results[i].source ? results[i].source : "unknown";
        int needed = snprintf(NULL, 0, "[Source: %s]\n%s\n\n", src, results[i].text);
        if (needed <= 0) continue;

        if (used + (size_t)needed > (size_t)max_chars) break;

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

POROOL_API char *porool_query(porool_t *p, const char *query, int top_k, int max_chars)
{
    if (!p || !query || !p->ready) return NULL;
    if (top_k     <= 0) top_k     = p->cfg.top_k_default;
    if (max_chars <= 0) max_chars = p->cfg.max_context_chars;

    float *emb = NULL;
    int    dim = 0;
    if (porool_embed_query(p, query, &emb, &dim) != 0) return NULL;

    int          n   = 0;
    SearchResult *res = porool_retrieve(p, emb, top_k, &n);
    free(emb);
    if (!res) return NULL;

    rerank_internal(p, res, n, query);

    char *ctx = porool_build_context(p, res, n, max_chars, p->cfg.ctx_min_score);
    porool_free_results(res, n);
    return ctx;
}

/* ── Table discovery ─────────────────────────────────────────────────────────*/

static int registry_read(const char *data_dir, const char *db,
                           char out[][256], int max_out)
{
    int n = 0;
#ifdef _WIN32
    char pat[640];
    snprintf(pat, sizeof(pat), "%s/%s/*.odat", data_dir, db);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        const char *fn = fd.cFileName;
        int fl = (int)strlen(fn);
        if (fl <= 5) continue;
        if (n >= max_out) break;
        snprintf(out[n], 256, "%s.%.*s", db, fl - 5, fn);
        n++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    char pat[640];
    snprintf(pat, sizeof(pat), "%s/%s/*.odat", data_dir, db);
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

static int registry_read_all(const char *data_dir, char out[][256], int max_out)
{
    int n = 0;
#ifdef _WIN32
    char pat[640];
    snprintf(pat, sizeof(pat), "%s/*", data_dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (fd.cFileName[0] == '.') continue;
        n += registry_read(data_dir, fd.cFileName, out + n, max_out - n);
    } while (n < max_out && FindNextFileA(h, &fd));
    FindClose(h);
#else
    char pat[640];
    snprintf(pat, sizeof(pat), "%s/*/*.odat", data_dir);
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
    char *pp = s;
    while (*pp == ' ') pp++;
    if (pp != s) memmove(s, pp, strlen(pp) + 1);
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

POROOL_API int porool_ingest_with_meta(porool_t *p, const char *file_path,
                                        const char *db, const char *table,
                                        const ChunkMeta *meta)
{
    if (!p || !file_path || !db || !table || !p->ready) return -1;

    char *raw = porool_extract(file_path);
    if (!raw) return -2;

    p_normalize(raw);

    int nc = 0;
    char **chunks = p_chunk_text(raw, &nc);
    free(raw);
    if (!chunks || nc == 0) { p_free_chunks(chunks, nc); return -3; }

    ensure_db_subdir(p->cfg.data_dir, db);

    char cl[256];
    snprintf(cl, sizeof(cl), "%s.%s", db, table);
    int d = sk_get_dim();

    tde_handle_t old_tbl = tde_open_odat(cl);
    int old_n = old_tbl ? tde_row_count(old_tbl) : 0;

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

    const char *cols[] = {
        "text", "source", "chunk_id",
        "concept", "section", "type",
        "tags", "importance", "related_concepts",
        "language"
    };

    tde_handle_t new_tbl = tde_create(cols, ODAT_NCOLS);
    if (!new_tbl) {
        if (old_tbl) tde_close(old_tbl);
        p_free_chunks(chunks, nc);
        return -5;
    }

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

    const char *m_concept  = (meta && meta->concept)          ? meta->concept          : "";
    const char *m_section  = (meta && meta->section)          ? meta->section          : "";
    const char *m_type     = (meta && meta->type)             ? meta->type             : "";
    const char *m_tags     = (meta && meta->tags)             ? meta->tags             : "";
    const char *m_imp      = (meta && meta->importance)       ? meta->importance       : "";
    const char *m_related  = (meta && meta->related_concepts) ? meta->related_concepts : "";
    const char *m_language = (meta && meta->language && meta->language[0])
                             ? meta->language
                             : p->cfg.vocab_first_lang;

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
            tde_row_set_string(row, COL_LANGUAGE,         m_language);
            tde_row_commit(row);
        }
    }

    if (tde_save_logical(new_tbl, cl) != TDE_OK) {
        tde_close(new_tbl);
        p_free_chunks(chunks, nc);
        return -6;
    }
    tde_close(new_tbl);

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

POROOL_API int porool_ingest(porool_t *p, const char *file_path,
                              const char *db, const char *table)
{
    return porool_ingest_with_meta(p, file_path, db, table, NULL);
}

/* ── Multi-table retrieval ───────────────────────────────────────────────────*/

POROOL_API SearchResult *porool_retrieve_from(porool_t *p, float *query_vector,
                                               const char *db,
                                               const char *table_name,
                                               int top_k, int *result_count)
{
    if (!p || !query_vector || !db || top_k <= 0 || !result_count || !p->ready)
        return NULL;
    *result_count = 0;

    int d = sk_get_dim();

#define MAX_TABLES 64
    char tables[MAX_TABLES][256];
    int  ntables = 0;

    if (table_name) {
        snprintf(tables[0], 256, "%.127s.%.127s", db, table_name);
        ntables = 1;
    } else {
        ntables = registry_read(p->cfg.data_dir, db, tables, MAX_TABLES);
        if (ntables == 0) return NULL;
    }

    int total_cap = top_k * ntables;
    SearchResult *all = (SearchResult *)calloc((size_t)total_cap, sizeof(SearchResult));
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

        if (tde_row_count(chunks_h) > 0) {
            uint32_t stored_dim = 0;
            tde_vector_get_ptr(vecs_h, 0, &stored_dim);
            if (stored_dim != 0 && stored_dim != (uint32_t)d) {
                tde_close(chunks_h); tde_close(vecs_h);
                continue;
            }
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

POROOL_API char *porool_query_from(porool_t *p, const char *query,
                                    const char *db, const char *table_name,
                                    int top_k, int max_chars)
{
    if (!p || !query || !db || !p->ready) return NULL;
    if (top_k     <= 0) top_k     = p->cfg.top_k_default;
    if (max_chars <= 0) max_chars = p->cfg.max_context_chars;

    float *emb = NULL;
    int    dim = 0;
    if (porool_embed_query(p, query, &emb, &dim) != 0) return NULL;

    int           n   = 0;
    SearchResult *res = porool_retrieve_from(p, emb, db, table_name, top_k, &n);
    free(emb);
    if (!res) return NULL;

    rerank_internal(p, res, n, query);
    char *ctx = porool_build_context(p, res, n, max_chars, p->cfg.ctx_min_score);
    porool_free_results(res, n);
    return ctx;
}

/* ── Target-string API ───────────────────────────────────────────────────────*/

POROOL_API SearchResult *porool_retrieve_target(porool_t *p, float *query_vector,
                                                 const char *target,
                                                 int top_k, int *result_count)
{
    if (!p || !query_vector || top_k <= 0 || !result_count || !p->ready) return NULL;
    char db[256], tbl[256];
    int mode = parse_target(target, db, sizeof(db), tbl, sizeof(tbl));

    if (mode == 0) return porool_retrieve_from(p, query_vector, db, tbl,  top_k, result_count);
    if (mode == 1) return porool_retrieve_from(p, query_vector, db, NULL, top_k, result_count);

#define ALL_MAX (64 * 4)
    char tables[ALL_MAX][256];
    int ntables = registry_read_all(p->cfg.data_dir, tables, ALL_MAX);
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
        if (tde_row_count(ch) > 0) {
            uint32_t stored_dim = 0;
            tde_vector_get_ptr(vh, 0, &stored_dim);
            if (stored_dim != 0 && stored_dim != (uint32_t)d) {
                tde_close(ch); tde_close(vh); continue;
            }
        }
        int found = tde_vector_search_topk(vh, query_vector, (uint32_t)d, (uint32_t)top_k,
                                            ids, scores);
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

POROOL_API char *porool_query_target(porool_t *p, const char *query,
                                      const char *target, int top_k, int max_chars)
{
    if (!p || !query || !p->ready) return NULL;
    if (top_k     <= 0) top_k     = p->cfg.top_k_default;
    if (max_chars <= 0) max_chars = p->cfg.max_context_chars;

    float *emb = NULL; int dim = 0;
    if (porool_embed_query(p, query, &emb, &dim) != 0) return NULL;

    int n = 0;
    SearchResult *res = porool_retrieve_target(p, emb, target, top_k, &n);
    free(emb);
    if (!res) return NULL;

    rerank_internal(p, res, n, query);
    char *ctx = porool_build_context(p, res, n, max_chars, p->cfg.ctx_min_score);
    porool_free_results(res, n);
    return ctx;
}

/* ── Phrasing API ────────────────────────────────────────────────────────────*/

POROOL_API void porool_phrasing_reload(porool_t *p)
{
    if (p && p->ready) phrases_reload(p);
}

POROOL_API int porool_phrasing_add(porool_t *p, const char *pattern, int is_prefix)
{
    if (!p || !pattern || !pattern[0] || !p->ready) return -2;
    const char *logical = is_prefix ? PHRASE_PREFIXES_LOGICAL : PHRASE_MARKERS_LOGICAL;
    char **cache        = is_prefix ? p->prefixes : p->markers;
    int   *count        = is_prefix ? &p->nprefixes : &p->nmarkers;

    PHRASE_RLOCK(p);
    for (int i = 0; i < *count; i++)
        if (cache[i] && strcmp(cache[i], pattern) == 0) {
            PHRASE_RUNLOCK(p);
            return -1;
        }
    PHRASE_RUNLOCK(p);

    char explicit_path[512 + 64];
    int  use_explicit = p->cfg.phrasing_root[0] != '\0';
    if (use_explicit)
        phrase_explicit_path(p->cfg.phrasing_root, logical, explicit_path, sizeof explicit_path);

    tde_handle_t old   = use_explicit ? tde_open(explicit_path) : tde_open_odat(logical);
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

    int save_rc = use_explicit ? tde_save(tbl, explicit_path)
                               : tde_save_logical(tbl, logical);
    tde_close(tbl);
    if (save_rc != TDE_OK) return -2;
    phrases_reload(p);
    return 0;
}

/* ── Synonym API ─────────────────────────────────────────────────────────────*/

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
