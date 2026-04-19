/*
 * porool_cli.c — RAG command-line tool
 *
 * Commands:
 *   porool ingest <file> --db <db> --table <table> [--ini <tharavu.ini>]
 *   porool query  <text> --db <db> --table <table> [--topk N] [--max-chars N]
 *   porool stats  --db <db> --table <table> [--ini <tharavu.ini>]
 *   porool peek   --db <db> --table <table> --id <id> [--ini <tharavu.ini>]
 *
 * Supported file types (no external tools required):
 *   .txt  .pdf  .docx  .xlsx
 * Image files (.jpg .jpeg .png) require an OCR callback — see porool_register_ocr().
 *
 * Build:
 *   gcc scr/porool_cli.c scr/porool_extract.c -std=c99 -Iinclude \
 *       -Llib -lsorkuvai_dll -ltharavu_dll -lz -lm -o bin/porool.exe
 */

#include "tharavu_dll.h"
#include "sorkuvai_dll.h"
#include "porool_extract.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>

#ifdef _WIN32
#  include <windows.h>
#  define strcasecmp _stricmp
#else
#  include <glob.h>
#endif

/* ── Constants ──────────────────────────────────────────────────────────── */

#define DEFAULT_INI       "tharavu.ini"
#define DEFAULT_DATA_DIR  "./data"
#define DEFAULT_TOPK      5
#define DEFAULT_MAX_CHARS 2000
#define CHUNK_CHARS       500
#define CHUNK_OVERLAP     50
#define REG_MAX_TABLES    64
#define DEFAULT_POROOL_INI "porool.ini"
#define DEFAULT_VOCAB_NAME "general.english"

/* ── Logical name builders ──────────────────────────────────────────────── */

static void lname_chunks(char *out, int sz, const char *db, const char *tbl)
{ snprintf(out, sz, "%s.%s", db, tbl); }

/*
 * Scan {data_dir} for subdirectories (each is a DB), then list *.odat files
 * inside each — but skip *_vec.odat (those are vector index companions).
 * Fills logical[] with "db.table" strings. Returns count.
 */
static int cli_show_tables(const char *data_dir, char logical[][512], int max)
{
    int n = 0;
#ifdef _WIN32
    char db_pat[1024];
    snprintf(db_pat, sizeof(db_pat), "%s/*", data_dir);
    WIN32_FIND_DATAA db_fd;
    HANDLE db_h = FindFirstFileA(db_pat, &db_fd);
    if (db_h == INVALID_HANDLE_VALUE) return 0;
    do {
        if (!(db_fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (!strcmp(db_fd.cFileName, ".") || !strcmp(db_fd.cFileName, "..")) continue;
        const char *db = db_fd.cFileName;
        char tbl_pat[1024];
        snprintf(tbl_pat, sizeof(tbl_pat), "%s/%s/*.odat", data_dir, db);
        WIN32_FIND_DATAA tf;
        HANDLE th = FindFirstFileA(tbl_pat, &tf);
        if (th == INVALID_HANDLE_VALUE) continue;
        do {
            if (tf.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            char *fname = tf.cFileName;
            int fl = (int)strlen(fname);
            int el = (int)strlen(".odat");
            if (fl <= el) continue;
            fname[fl - el] = '\0'; /* strip .odat */
            if (n < max) {
                snprintf(logical[n], 512, "%.255s.%.255s", db, fname);
                n++;
            }
        } while (n < max && FindNextFileA(th, &tf));
        FindClose(th);
    } while (n < max && FindNextFileA(db_h, &db_fd));
    FindClose(db_h);
#else
    /* Linux/macOS */
    char db_pat[1024];
    snprintf(db_pat, sizeof(db_pat), "%s/*/", data_dir);
    glob_t dg = {0};
    if (glob(db_pat, GLOB_MARK, NULL, &dg) != 0) return 0;
    for (size_t di = 0; di < dg.gl_pathc && n < max; di++) {
        const char *dpath = dg.gl_pathv[di];
        const char *db = strrchr(dpath, '/');
        if (!db || db == dpath) continue;
        db--; while (db > dpath && *db != '/') db--; if (*db == '/') db++;
        char tbl_pat[1024];
        snprintf(tbl_pat, sizeof(tbl_pat), "%s*.odat", dpath);
        glob_t fg = {0};
        if (glob(tbl_pat, 0, NULL, &fg) == 0) {
            for (size_t fi = 0; fi < fg.gl_pathc && n < max; fi++) {
                const char *base = strrchr(fg.gl_pathv[fi], '/');
                base = base ? base + 1 : fg.gl_pathv[fi];
                int bl = (int)strlen(base);
                int el = (int)strlen(".odat");
                if (bl <= el) continue;
                char tname[256]; strncpy(tname, base, bl - el); tname[bl - el] = '\0';
                snprintf(logical[n], 256, "%s.%s", db, tname); n++;
            }
            globfree(&fg);
        }
    }
    globfree(&dg);
#endif
    return n;
}

typedef struct { char *text; char *source; float score; } QHit;
static int qhit_cmp(const void *a, const void *b) {
    float d = ((const QHit *)b)->score - ((const QHit *)a)->score;
    return (d > 0.0f) ? 1 : (d < 0.0f) ? -1 : 0;
}

/* ── Text normalization ─────────────────────────────────────────────────── */

static void normalize(char *s)
{
    char *r = s, *w = s;
    int space = 0;
    while (*r) {
        unsigned char c = (unsigned char)*r++;
        if (c < 0x20 || c == 0x7f) {
            if (!space) { *w++ = ' '; space = 1; }
        } else {
            *w++ = (char)c;
            space = 0;
        }
    }
    *w = '\0';
    /* trim leading */
    char *p = s;
    while (*p == ' ') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    /* trim trailing */
    int len = (int)strlen(s);
    while (len > 0 && s[len - 1] == ' ') s[--len] = '\0';
}

/* ── Chunking ───────────────────────────────────────────────────────────── */

/* Split oversized text at the last sentence boundary before `limit`.
 * Falls back to the last whitespace, then hard-cuts at limit. */
static int find_split(const char *text, int limit)
{
    /* prefer sentence end: '. ', '? ', '! ' */
    for (int i = limit - 1; i > limit / 2; i--) {
        if ((text[i] == '.' || text[i] == '?' || text[i] == '!')
                && isspace((unsigned char)text[i + 1]))
            return i + 1;
    }
    /* fall back to whitespace */
    for (int i = limit - 1; i > limit / 2; i--) {
        if (isspace((unsigned char)text[i]))
            return i;
    }
    return limit;
}

static char **chunk_text(const char *text, int *out_n)
{
    int len = (int)strlen(text);
    if (len == 0) { *out_n = 0; return NULL; }

    int cap = len / (CHUNK_CHARS / 2) + 8;
    char **chunks = malloc((size_t)cap * sizeof(char *));
    if (!chunks) return NULL;
    int n = 0;

    /* Split into paragraphs on blank lines, then size-check each one. */
    int pos = 0;
    while (pos < len && n < cap - 1) {
        /* skip leading blank lines */
        while (pos < len && text[pos] == '\n') pos++;
        if (pos >= len) break;

        /* find end of paragraph (blank line or EOF) */
        int end = pos;
        while (end < len) {
            if (text[end] == '\n' && end + 1 < len && text[end + 1] == '\n')
                break;
            end++;
        }

        int plen = end - pos;
        if (plen <= 0) { pos = end + 2; continue; }

        if (plen <= CHUNK_CHARS) {
            /* whole paragraph fits in one chunk */
            chunks[n] = malloc((size_t)plen + 1);
            if (!chunks[n]) break;
            memcpy(chunks[n], text + pos, (size_t)plen);
            chunks[n][plen] = '\0';
            n++;
        } else {
            /* paragraph too long — split at sentence/word boundaries */
            int sub = pos;
            while (sub < end && n < cap - 1) {
                int avail = end - sub;
                int cut = (avail <= CHUNK_CHARS) ? avail
                                                 : find_split(text + sub, CHUNK_CHARS);
                chunks[n] = malloc((size_t)cut + 1);
                if (!chunks[n]) goto done;
                memcpy(chunks[n], text + sub, (size_t)cut);
                chunks[n][cut] = '\0';
                n++;
                sub += cut;
                /* skip whitespace between sub-chunks */
                while (sub < end && isspace((unsigned char)text[sub])) sub++;
            }
        }

        pos = end + 2;
    }
done:
    *out_n = n;
    return chunks;
}

static void free_chunks(char **chunks, int n)
{
    if (!chunks) return;
    for (int i = 0; i < n; i++) free(chunks[i]);
    free(chunks);
}

/* ── Embedding ──────────────────────────────────────────────────────────── */

/* Mean-pool all token vectors. Returns caller-owned float[dim], never NULL. */
static float *embed(const char *text)
{
    int d = sk_get_dim();
    uint32_t *ids = NULL;
    int n = 0;

    if (ve_process_text(text, &ids, &n) != VE_OK || n == 0) {
        ve_free_ids(ids);
        return calloc((size_t)d, sizeof(float));
    }

    float *vec  = calloc((size_t)d, sizeof(float));
    float *slot = malloc((size_t)d * sizeof(float));
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

/* ── ODAT string fetch (two-call pattern) ───────────────────────────────── */

static char *get_str(tde_handle_t h, int row, int col)
{
    int need = tde_get_string(h, row, col, NULL, 0);
    if (need <= 0) return NULL;
    char *buf = malloc((size_t)need);
    if (!buf) return NULL;
    if (tde_get_string(h, row, col, buf, need) <= 0) { free(buf); return NULL; }
    return buf;
}

/* ── INI bootstrap ──────────────────────────────────────────────────────── */

static void ensure_porool_ini(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (fp) { fclose(fp); return; }

    fp = fopen(path, "w");
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
        "; Weights should sum to 1.0\n"
        "w1 = 0.60\n"
        "w2 = 0.20\n"
        "w3 = 0.20\n"
        "\n"
        "[source_weights]\n"
        "; Per-source priority multipliers.  Range [0.0, 2.0]; values outside are clamped.\n"
        "; Add entries in the form:  source_name = weight\n"
        "; Example:  my_docs = 1.5\n"
    );
    fclose(fp);
    fprintf(stderr, "porool: created default %s\n", path);
}

/* ── Engine init helpers ────────────────────────────────────────────────── */

/* Read the first vocab logical name from porool.ini.
 * Reads [porool] vocabularies = <group>, then finds [<group>] section and
 * takes the first entry's value (e.g. "general/english.ovoc"), strips the
 * .ovoc extension, and converts '/' to '.' to form a logical name.
 * Falls back to DEFAULT_VOCAB_NAME if anything is missing. */
static void read_vocab_name(const char *porool_ini, char *out, int outsz)
{
    strncpy(out, DEFAULT_VOCAB_NAME, outsz - 1); out[outsz - 1] = '\0';
    FILE *f = fopen(porool_ini, "r");
    if (!f) return;

    char group[64] = "";
    char line[512];
    int  in_porool = 0, in_group = 0;

    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ';' || *p == '#' || *p == '\r' || *p == '\n' || *p == '\0') continue;

        if (*p == '[') {
            char *end = strchr(p + 1, ']');
            if (!end) continue;
            char sec[64]; int sl = (int)(end - p - 1);
            if (sl >= (int)sizeof(sec)) sl = (int)sizeof(sec) - 1;
            strncpy(sec, p + 1, sl); sec[sl] = '\0';
            in_porool = (!group[0] && strcmp(sec, "porool") == 0);
            in_group  = (group[0] && strcmp(sec, group) == 0);
            continue;
        }

        char *eq = strchr(p, '=');
        if (!eq) continue;
        char *val = eq + 1;
        while (*val == ' ' || *val == '\t') val++;
        int vlen = (int)strlen(val);
        while (vlen > 0 && (val[vlen-1] == '\n' || val[vlen-1] == '\r' || val[vlen-1] == ' '))
            val[--vlen] = '\0';

        if (in_porool) {
            /* trim key */
            char key[64]; int klen = (int)(eq - p);
            while (klen > 0 && (p[klen-1] == ' ' || p[klen-1] == '\t')) klen--;
            if (klen >= (int)sizeof(key)) klen = (int)sizeof(key) - 1;
            strncpy(key, p, klen); key[klen] = '\0';
            if (strcmp(key, "vocabularies") == 0 && vlen > 0 && vlen < (int)sizeof(group)) {
                strncpy(group, val, sizeof(group) - 1); group[sizeof(group)-1] = '\0';
                /* re-scan from top to find the group section */
                rewind(f); in_porool = 0; in_group = 0;
            }
            continue;
        }

        if (in_group && vlen > 0) {
            /* strip .ovoc extension if present */
            if (vlen > 5 && strcmp(val + vlen - 5, ".ovoc") == 0) val[vlen -= 5] = '\0';
            /* convert '/' to '.' for logical name */
            for (char *c = val; *c; c++) if (*c == '/') *c = '.';
            strncpy(out, val, outsz - 1); out[outsz - 1] = '\0';
            break;
        }
    }
    fclose(f);
}

static int init_engines(const char *ini)
{
    if (tde_config_load(ini) != TDE_OK) {
        printf("porool: tde_config_load(%s): %s\n",
               ini, tde_strerror(tde_last_error()));
        return -1;
    }
    char vocab[256];
    read_vocab_name(DEFAULT_POROOL_INI, vocab, sizeof(vocab));

    if (ve_init(vocab, NULL) != VE_OK) {
        printf("porool: ve_init(%s): %s\n", vocab, ve_strerror(ve_last_error()));
        return -1;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ingest
 * ══════════════════════════════════════════════════════════════════════════ */

static int cmd_ingest(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: porool ingest <file> --db <db> --table <table> [--ini <ini>]\n");
        return 1;
    }

    const char *infile = argv[2];
    const char *db = NULL, *table = NULL, *ini = DEFAULT_INI;

    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i], "--db")    && i+1 < argc) db    = argv[++i];
        else if (!strcmp(argv[i], "--table") && i+1 < argc) table = argv[++i];
        else if (!strcmp(argv[i], "--ini")   && i+1 < argc) ini   = argv[++i];
    }
    if (!db || !table) {
        fprintf(stderr, "porool: --db and --table required\n");
        return 1;
    }

    /* 1. Extract text (built-in: no external tools required) */
    char *raw = porool_extract(infile);
    if (!raw) {
        fprintf(stderr, "porool: failed to extract text from %s\n", infile);
        return 1;
    }

    normalize(raw);

    /* 2. Chunk */
    int nc = 0;
    char **chunks = chunk_text(raw, &nc);
    free(raw);
    if (!chunks || nc == 0) {
        fprintf(stderr, "porool: no content extracted from %s\n", infile);
        free_chunks(chunks, nc);
        return 1;
    }
    printf("porool: %d chunks extracted from %s\n", nc, infile);
    fflush(stdout);

    /* 3. Init */
    if (init_engines(ini) != 0) { free_chunks(chunks, nc); return 1; }

    char cl[512];
    lname_chunks(cl, sizeof(cl), db, table);
    int d = sk_get_dim();

    /* 4. Load existing rows (append support) */
    tde_handle_t old_tbl = tde_open_odat(cl);
    int old_n = old_tbl ? tde_row_count(old_tbl) : 0;

    /* 5. Build new ODAT: existing rows + new chunks */
    const char *cols[] = { "text", "source" };
    tde_handle_t new_tbl = tde_create(cols, 2);
    if (!new_tbl) {
        fprintf(stderr, "porool: tde_create failed\n");
        if (old_tbl) tde_close(old_tbl);
        free_chunks(chunks, nc);
        ve_cleanup();
        return 1;
    }

    for (int i = 0; i < old_n; i++) {
        char *txt = get_str(old_tbl, i, 0);
        char *src = get_str(old_tbl, i, 1);
        tde_handle_t row = tde_row_begin(new_tbl);
        if (row) {
            tde_row_set_string(row, 0, txt ? txt : "");
            tde_row_set_string(row, 1, src ? src : "");
            tde_row_commit(row);
        }
        free(txt); free(src);
    }
    if (old_tbl) { tde_close(old_tbl); old_tbl = NULL; }

    for (int i = 0; i < nc; i++) {
        tde_handle_t row = tde_row_begin(new_tbl);
        if (row) {
            tde_row_set_string(row, 0, chunks[i]);
            tde_row_set_string(row, 1, infile);
            tde_row_commit(row);
        }
    }

    if (tde_save_logical(new_tbl, cl) != TDE_OK) {
        printf("porool: tde_save_logical(%s): %s\n",
               cl, tde_strerror(tde_last_error()));
        tde_close(new_tbl);
        free_chunks(chunks, nc);
        ve_cleanup();
        return 1;
    }
    tde_close(new_tbl);

    /* 6. Build combined vector store */
    int total = old_n + nc;
    float *vecs = calloc((size_t)total * (size_t)d, sizeof(float));
    if (!vecs) {
        fprintf(stderr, "porool: out of memory (%d vectors x %d dim)\n", total, d);
        free_chunks(chunks, nc);
        ve_cleanup();
        return 1;
    }

    if (old_n > 0) {
        tde_handle_t old_ov = tde_open_ovec(cl);
        if (old_ov) {
            uint32_t *ids = malloc((size_t)old_n * sizeof(uint32_t));
            if (ids) {
                for (int i = 0; i < old_n; i++) ids[i] = (uint32_t)i;
                uint32_t dim_out = 0;
                tde_vector_get_batch(old_ov, ids, old_n, vecs, &dim_out);
                free(ids);
            }
            tde_close(old_ov);
        } else {
            /* Vector store missing — re-embed existing chunks from their text. */
            fprintf(stderr,
                "porool: warning: no existing vector store at %s — "
                "re-embedding %d chunk(s)\n", cl, old_n);
            for (int i = 0; i < old_n; i++) {
                char *old_txt = get_str(new_tbl, i, 0);
                if (old_txt) {
                    float *v = embed(old_txt);
                    if (v) {
                        memcpy(vecs + (size_t)i * (size_t)d, v,
                               (size_t)d * sizeof(float));
                        free(v);
                    }
                    free(old_txt);
                }
            }
        }
    }

    printf("porool: embedding %d chunk(s)...\n", nc);
    for (int i = 0; i < nc; i++) {
        float *v = embed(chunks[i]);
        if (v) {
            memcpy(vecs + (size_t)(old_n + i) * (size_t)d, v,
                   (size_t)d * sizeof(float));
            free(v);
        }
        if ((i + 1) % 20 == 0 || i == nc - 1)
            printf("  %d/%d\r", i + 1, nc);
    }
    printf("\n");

    free_chunks(chunks, nc);

    if (tde_build_vectors_logical(cl, vecs, total, (uint32_t)d) != TDE_OK) {
        printf("porool: tde_build_vectors_logical(%s): %s\n",
               cl, tde_strerror(tde_last_error()));
        free(vecs);
        ve_cleanup();
        return 1;
    }
    free(vecs);
    ve_cleanup();

    printf("porool: stored %d new chunk(s), total %d in %s\n", nc, total, cl);
    fflush(stdout);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  query
 * ══════════════════════════════════════════════════════════════════════════ */

/* Collect up to topk hits from one open table into hits[]. Returns added count. */
static int collect_hits(tde_handle_t chunks, tde_handle_t vecs,
                         float *qv, int d, int topk,
                         QHit *hits, int nhits, int hits_cap)
{
    uint32_t *ids    = malloc((size_t)topk * sizeof(uint32_t));
    float    *scores = malloc((size_t)topk * sizeof(float));
    if (!ids || !scores) { free(ids); free(scores); return 0; }

    int found = tde_vector_search_topk(vecs, qv, (uint32_t)d,
                                       (uint32_t)topk, ids, scores);
    int added = 0;
    for (int i = 0; i < found && nhits + added < hits_cap; i++) {
        char *txt = get_str(chunks, (int)ids[i], 0);
        if (txt) {
            hits[nhits + added].text   = txt;
            hits[nhits + added].source = get_str(chunks, (int)ids[i], 1);
            hits[nhits + added].score  = scores[i];
            added++;
        }
    }
    free(ids); free(scores);
    return added;
}

static void json_print_str(const char *s)
{
    putchar('"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if      (c == '"')  printf("\\\"");
        else if (c == '\\') printf("\\\\");
        else if (c == '\n') printf("\\n");
        else if (c == '\r') printf("\\r");
        else if (c == '\t') printf("\\t");
        else if (c < 0x20)  printf("\\u%04x", c);
        else                putchar((char)c);
    }
    putchar('"');
}

static void print_hits(QHit *hits, int nhits, int topk, int max_chars,
                       const char *query)
{
    qsort(hits, (size_t)nhits, sizeof(QHit), qhit_cmp);

    printf("{\n  \"query\": ");
    json_print_str(query);
    printf(",\n  \"results\": [");

    int nseen = 0, chars_used = 0, first = 1;
    for (int i = 0; i < nhits && nseen < topk && chars_used < max_chars; i++) {
        char *txt = hits[i].text;
        if (!txt) continue;
        int dup = 0;
        for (int j = 0; j < i; j++)
            if (hits[j].text && strcmp(hits[j].text, txt) == 0) { dup = 1; break; }
        if (dup) continue;
        int tlen = (int)strlen(txt);
        if (chars_used + tlen > max_chars) break;

        if (!first) printf(",");
        first = 0;
        printf("\n    {\n");
        printf("      \"rank\": %d,\n", nseen + 1);
        printf("      \"score\": %.6f,\n", hits[i].score);
        printf("      \"source\": ");
        json_print_str(hits[i].source ? hits[i].source : "");
        printf(",\n      \"text\": ");
        json_print_str(txt);
        printf("\n    }");

        chars_used += tlen;
        nseen++;
    }

    printf("\n  ]\n}\n");
}

static int cmd_query(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: porool query <text> --db <db> --table <table|all>"
            " [--topk N] [--max-chars N] [--ini <ini>] [--data-dir <dir>]\n");
        return 1;
    }

    const char *query_text = argv[2];
    const char *db = NULL, *table = NULL, *ini = DEFAULT_INI;
    const char *data_dir = DEFAULT_DATA_DIR;
    int topk = DEFAULT_TOPK, max_chars = DEFAULT_MAX_CHARS;

    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i], "--db")        && i+1 < argc) db       = argv[++i];
        else if (!strcmp(argv[i], "--table")     && i+1 < argc) table    = argv[++i];
        else if (!strcmp(argv[i], "--ini")       && i+1 < argc) ini      = argv[++i];
        else if (!strcmp(argv[i], "--data-dir")  && i+1 < argc) data_dir = argv[++i];
        else if (!strcmp(argv[i], "--topk")      && i+1 < argc) topk     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-chars") && i+1 < argc) max_chars= atoi(argv[++i]);
    }
    if (!table) table = "all";
    if (topk <= 0)      topk      = DEFAULT_TOPK;
    if (max_chars <= 0) max_chars = DEFAULT_MAX_CHARS;

    /* Bare "all" with no db = search every table in data_dir */
    int all_mode = (!db && !strcasecmp(table, "all"))
                || ( db && !strcasecmp(db,    "all"))
                || (!db);

    /* ── single table ── */
    if (!all_mode) {
        if (init_engines(ini) != 0) return 1;
        int   d  = sk_get_dim();
        float *qv = embed(query_text);
        if (!qv) { fprintf(stderr, "porool: embedding failed\n"); ve_cleanup(); return 1; }

        char cl[512];
        lname_chunks(cl, sizeof(cl), db, table);
        tde_handle_t chunks = tde_open_odat(cl);
        tde_handle_t vecs   = tde_open_ovec(cl);
        if (!chunks || !vecs) {
            fprintf(stderr, "porool: cannot open %s - run ingest first\n", cl);
            if (chunks) tde_close(chunks);
            if (vecs)   tde_close(vecs);
            free(qv); ve_cleanup(); return 1;
        }
        QHit *hits = malloc((size_t)topk * sizeof(QHit));
        if (!hits) {
            fprintf(stderr, "porool: out of memory\n");
            tde_close(chunks); tde_close(vecs);
            free(qv); ve_cleanup(); return 1;
        }
        int nhits = collect_hits(chunks, vecs, qv, d, topk, hits, 0, topk);
        print_hits(hits, nhits, topk, max_chars, query_text);
        for (int i = 0; i < nhits; i++) { free(hits[i].text); free(hits[i].source); }
        free(hits); free(qv);
        tde_close(chunks); tde_close(vecs); ve_cleanup();
        return 0;
    }

    /* ── all tables: scan data_dir subdirs for *.odat files ── */
    char logical[REG_MAX_TABLES * 4][512];
    int  ntables;

    if (db && strcasecmp(db, "all")) {
        /* specific db, all tables: scan data_dir/db/ for .odat files */
        ntables = 0;
#ifdef _WIN32
        char pat[1280];
        snprintf(pat, sizeof(pat), "%s/%s/*.odat", data_dir, db);
        WIN32_FIND_DATAA fd2;
        HANDLE h2 = FindFirstFileA(pat, &fd2);
        if (h2 != INVALID_HANDLE_VALUE) {
            do {
                char *fn = fd2.cFileName;
                int fl = (int)strlen(fn);
                if (fl > 5 && ntables < REG_MAX_TABLES * 4) {
                    fn[fl - 5] = '\0';
                    snprintf(logical[ntables++], 512, "%.255s.%.255s", db, fn);
                }
            } while (ntables < REG_MAX_TABLES * 4 && FindNextFileA(h2, &fd2));
            FindClose(h2);
        }
#else
        char gpat[1280];
        snprintf(gpat, sizeof(gpat), "%s/%s/*.odat", data_dir, db);
        glob_t gg = {0};
        if (glob(gpat, 0, NULL, &gg) == 0) {
            for (size_t gi = 0; gi < gg.gl_pathc && ntables < REG_MAX_TABLES*4; gi++) {
                const char *base = strrchr(gg.gl_pathv[gi], '/');
                base = base ? base+1 : gg.gl_pathv[gi];
                int bl = (int)strlen(base);
                if (bl <= 5) continue;
                char tn[256]; strncpy(tn, base, bl-5); tn[bl-5]='\0';
                snprintf(logical[ntables++], 512, "%.255s.%.255s", db, tn);
            }
            globfree(&gg);
        }
#endif
    } else {
        ntables = cli_show_tables(data_dir, logical, REG_MAX_TABLES * 4);
    }

    if (ntables == 0) {
        fprintf(stderr, "porool: no tables found in %s\n", data_dir);
        return 1;
    }

    /* init engines with first table's db */
    char first_db[256] = "";
    const char *dot0 = strchr(logical[0], '.');
    if (dot0) { int dl = (int)(dot0-logical[0]); strncpy(first_db, logical[0], dl); first_db[dl]='\0'; }
    if (init_engines(ini) != 0) return 1;

    int   d  = sk_get_dim();
    float *qv = embed(query_text);
    if (!qv) { fprintf(stderr, "porool: embedding failed\n"); ve_cleanup(); return 1; }

    int   hits_cap = topk * ntables;
    QHit *hits     = malloc((size_t)hits_cap * sizeof(QHit));
    if (!hits) {
        fprintf(stderr, "porool: out of memory\n");
        free(qv); ve_cleanup(); return 1;
    }
    int   nhits    = 0;

    for (int t = 0; t < ntables; t++) {
        tde_handle_t chunks = tde_open_odat(logical[t]);
        tde_handle_t vecs   = tde_open_ovec(logical[t]);
        if (!chunks || !vecs) {
            if (chunks) tde_close(chunks);
            if (vecs)   tde_close(vecs);
            continue;
        }
        nhits += collect_hits(chunks, vecs, qv, d, topk, hits, nhits, hits_cap);
        tde_close(chunks); tde_close(vecs);
    }

    print_hits(hits, nhits, topk, max_chars, query_text);
    for (int i = 0; i < nhits; i++) { free(hits[i].text); free(hits[i].source); }
    free(hits); free(qv);
    ve_cleanup();
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  stats
 * ══════════════════════════════════════════════════════════════════════════ */

static int cmd_stats(int argc, char **argv)
{
    const char *db = NULL, *table = NULL, *ini = DEFAULT_INI;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--db")    && i+1 < argc) db    = argv[++i];
        else if (!strcmp(argv[i], "--table") && i+1 < argc) table = argv[++i];
        else if (!strcmp(argv[i], "--ini")   && i+1 < argc) ini   = argv[++i];
    }
    if (!db || !table) {
        fprintf(stderr, "Usage: porool stats --db <db> --table <table>\n");
        return 1;
    }

    char cl[512];
    lname_chunks(cl, sizeof(cl), db, table);

    if (tde_config_load(ini) != TDE_OK) {
        fprintf(stderr, "porool: tde_config_load(%s) failed\n", ini);
        return 1;
    }

    tde_handle_t ch = tde_open_odat(cl);
    tde_handle_t ov = tde_open_ovec(cl);

    int  chunk_n = ch ? tde_row_count(ch) : -1;
    uint32_t dim = 0;
    if (ov && chunk_n > 0) tde_vector_get_ptr(ov, 0, &dim);

    printf("Table:     %s\n", cl);
    printf("Chunks:    %d\n", chunk_n);
    printf("Vectors:   %d\n", chunk_n);  /* always in sync with chunks */
    if (dim) printf("Dim:       %u\n", dim);

    if (ch) tde_close(ch);
    if (ov) tde_close(ov);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  peek
 * ══════════════════════════════════════════════════════════════════════════ */

static int cmd_peek(int argc, char **argv)
{
    const char *db = NULL, *table = NULL, *ini = DEFAULT_INI;
    int id = -1;

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--db")    && i+1 < argc) db    = argv[++i];
        else if (!strcmp(argv[i], "--table") && i+1 < argc) table = argv[++i];
        else if (!strcmp(argv[i], "--ini")   && i+1 < argc) ini   = argv[++i];
        else if (!strcmp(argv[i], "--id")    && i+1 < argc) id    = atoi(argv[++i]);
    }
    if (!db || !table || id < 0) {
        fprintf(stderr,
            "Usage: porool peek --db <db> --table <table> --id <id>\n");
        return 1;
    }

    char cl[256];
    lname_chunks(cl, sizeof(cl), db, table);

    if (tde_config_load(ini) != TDE_OK) {
        fprintf(stderr, "porool: tde_config_load(%s) failed\n", ini);
        return 1;
    }

    tde_handle_t ch = tde_open_odat(cl);
    if (!ch) {
        fprintf(stderr, "porool: cannot open %s\n", cl);
        return 1;
    }

    int total = tde_row_count(ch);
    if (id >= total) {
        fprintf(stderr, "porool: id %d out of range [0, %d)\n", id, total);
        tde_close(ch);
        return 1;
    }

    char *txt = get_str(ch, id, 0);
    char *src = get_str(ch, id, 1);

    printf("ID:     %d\n", id);
    printf("Source: %s\n", src ? src : "unknown");
    printf("Text:   %s\n", txt ? txt : "(empty)");

    free(txt); free(src);
    tde_close(ch);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  interactive wizard
 * ══════════════════════════════════════════════════════════════════════════ */

static void trim_line(char *s)
{
    int n = (int)strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' || s[n-1] == ' ')) s[--n] = '\0';
    char *p = s;
    while (*p == ' ') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    /* strip surrounding quotes typed by user */
    n = (int)strlen(s);
    if (n >= 2 && s[0] == '"' && s[n-1] == '"') {
        memmove(s, s+1, (size_t)(n-2)); s[n-2] = '\0';
    }
}

/* Read a line from stdin into buf[size]. Returns 0 on EOF. */
static int read_line(const char *prompt, char *buf, int size)
{
    printf("%s", prompt);
    fflush(stdout);
    if (!fgets(buf, size, stdin)) return 0;
    trim_line(buf);
    return 1;
}

/* Parse "db.table" into separate db[] and table[] buffers. */
static int parse_dbtable(const char *s, char *db, int dbsz, char *tbl, int tblsz)
{
    const char *dot = strchr(s, '.');
    if (!dot || dot == s) return 0;
    int dlen = (int)(dot - s);
    if (dlen >= dbsz || (int)strlen(dot+1) >= tblsz) return 0;
    memcpy(db,  s,     (size_t)dlen); db[dlen] = '\0';
    memcpy(tbl, dot+1, (size_t)(tblsz-1)); tbl[tblsz-1] = '\0';
    return db[0] && tbl[0];
}

static void cmd_interactive(void)
{
    char ini[512] = DEFAULT_INI;
    int  topk     = DEFAULT_TOPK;
    int  max_chars = DEFAULT_MAX_CHARS;

    printf("porool RAG engine\n");
    printf("-----------------\n\n");

    char choice[64];
    for (;;) {
        printf("  [r] Retrieve   [i] Ingest   [q] Quit\n");
        if (!read_line("Method: ", choice, sizeof(choice))) break;
        printf("\n");

        if (!choice[0]) continue;

        /* ── Quit ── */
        if (choice[0] == 'q' || choice[0] == 'Q') {
            printf("bye\n");
            break;
        }

        /* ── Retrieve ── */
        if (choice[0] == 'r' || choice[0] == 'R') {
            char query[4096];
            if (!read_line("Query: ", query, sizeof(query))) break;
            if (!query[0]) { printf("(cancelled)\n\n"); continue; }

            char dbtbl[512];
            for (;;) {
                if (!read_line("DB.Table (or 'all'): ", dbtbl, sizeof(dbtbl))) goto done;
                if (dbtbl[0]) break;
                printf("  (required)\n");
            }
            printf("\n");

            char db[256] = "", table[256] = "";
            if (!strcasecmp(dbtbl, "all")) {
                strncpy(db,    "all", sizeof(db));
                strncpy(table, "all", sizeof(table));
            } else {
                char *dot = strchr(dbtbl, '.');
                if (dot && dot != dbtbl && *(dot+1)) {
                    int dlen = (int)(dot - dbtbl);
                    strncpy(db, dbtbl, dlen < 255 ? dlen : 255); db[dlen < 255 ? dlen : 255] = '\0';
                    strncpy(table, dot+1, 255);
                } else {
                    printf("  Enter db.table or 'all'\n\n");
                    continue;
                }
            }

            char topk_s[16], maxc_s[16];
            snprintf(topk_s, sizeof(topk_s), "%d", topk);
            snprintf(maxc_s, sizeof(maxc_s), "%d", max_chars);

            char *av[] = { "porool", "query", query,
                           "--db", db, "--table", table,
                           "--topk", topk_s, "--max-chars", maxc_s,
                           "--ini", ini, "--data-dir", ini /* reuse slot; see below */ };
            av[13] = (char *)DEFAULT_DATA_DIR;
            cmd_query(15, av);
            printf("\n");
            continue;
        }

        /* ── Ingest ── */
        if (choice[0] == 'i' || choice[0] == 'I') {
            char filepath[4096];
            if (!read_line("File: ", filepath, sizeof(filepath))) break;
            if (!filepath[0]) { printf("(cancelled)\n\n"); continue; }

            char dbtbl[512];
            if (!read_line("DB.Table (e.g. mydb.docs): ", dbtbl, sizeof(dbtbl))) break;
            printf("\n");

            char db[256] = "", table[256] = "";
            if (!parse_dbtable(dbtbl, db, sizeof(db), table, sizeof(table))) {
                printf("Error: enter DB.Table in the form  dbname.tablename\n\n");
                continue;
            }

            char *av[] = { "porool", "ingest", filepath,
                           "--db", db, "--table", table, "--ini", ini };
            cmd_ingest(9, av);
            printf("\n");
            continue;
        }

        printf("Type r, i, or q.\n\n");
    }
done:;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  main
 * ══════════════════════════════════════════════════════════════════════════ */

static void usage(void)
{
    fprintf(stderr,
        "porool - RAG document ingestion and query\n\n"
        "  porool ingest <file> --db <db> --table <table> [--ini <ini>]\n"
        "  porool query  <text> --db <db> --table <table>"
             " [--topk N] [--max-chars N] [--ini <ini>]\n"
        "  porool stats  --db <db> --table <table> [--ini <ini>]\n"
        "  porool peek   --db <db> --table <table> --id <id> [--ini <ini>]\n\n"
        "Run with no arguments to enter interactive mode.\n\n"
        "Supported types (built-in, no external tools):\n"
        "  .txt  .pdf  .docx  .xlsx\n"
        "  .jpg  .jpeg  .png  - requires porool_register_ocr() callback\n"
    );
}

#ifdef _WIN32
static int launched_from_explorer(void)
{
    DWORD procs[2];
    return GetConsoleProcessList(procs, 2) == 1;
}
#endif

int main(int argc, char **argv)
{
    ensure_porool_ini(DEFAULT_POROOL_INI);

    if (argc < 2) {
#ifdef _WIN32
        /* When double-clicked, Windows allocates a console but there is no
           parent terminal — interactive mode is exactly what the user wants. */
        (void)launched_from_explorer();
#endif
        cmd_interactive();
        return 0;
    }

    if (!strcmp(argv[1], "ingest")) return cmd_ingest(argc, argv);
    if (!strcmp(argv[1], "query"))  return cmd_query (argc, argv);
    if (!strcmp(argv[1], "stats"))  return cmd_stats (argc, argv);
    if (!strcmp(argv[1], "peek"))   return cmd_peek  (argc, argv);

    fprintf(stderr, "porool: unknown command '%s'\n\n", argv[1]);
    usage();
    return 1;
}
