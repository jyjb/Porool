/*
 * porool_cli.c — RAG command-line tool (thin wrapper over porool.dll)
 *
 * Commands:
 *   porool ingest <file> --db <db> --table <table>
 *          [--concept <c>] [--section <s>] [--type <t>]
 *          [--tags <t>] [--importance high|medium|low] [--related <r>]
 *          [--ini <porool.ini>]
 *   porool query  <text> [--db <db>] [--table <table>]
 *          [--topk N] [--max-chars N] [--concept <c>] [--ini <porool.ini>]
 *   porool stats  --db <db> --table <table> [--ini <tharavu.ini>]
 *   porool peek   --db <db> --table <table> --id <id> [--ini <tharavu.ini>]
 *   porool phrasing list [--ini <tharavu.ini>]
 *   porool phrasing add-prefix|add-marker <pattern> [--ini <porool.ini>]
 *
 * Build:
 *   gcc src/tools/porool_cli.c -std=c99 -Isrc/include -Lbuild -lporool -ltharavu_dll \
 *       -lm -o build/porool.exe
 */

#include "../include/porool.h"
#include "../include/tharavu_dll.h"   /* stats / peek / phrasing-list use tde directly */
#include "slispmanager.h"             /* read porool.ocfg for tde_set_base_path */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#ifdef _WIN32
#  include <windows.h>
#  define SK_CALL __cdecl
#  define strcasecmp _stricmp
#else
#  define SK_CALL
#endif

/* SORKUVAI forward declarations — caller manages lifecycle per directive 2026-05-30 */
extern int  SK_CALL ve_init(const char *logical_name, const char *ini_path);
extern void SK_CALL ve_cleanup(void);

/* THARAVU additional forward declaration — tde_get_base_path not in porool's tharavu_dll.h */
#ifdef _WIN32
#  define TDE_CALL __cdecl
#else
#  define TDE_CALL
#endif
extern const char *TDE_CALL tde_get_base_path(void);

#define DEFAULT_INI        "tharavu.ini"
#define DEFAULT_TOPK       5
#define DEFAULT_MAX_CHARS  2000

/* ODAT column indices — must match the schema written by porool_ingest_with_meta */
#define COL_TEXT             0
#define COL_SOURCE           1
#define COL_CHUNK_ID         2
#define COL_CONCEPT          3
#define COL_SECTION          4
#define COL_TYPE             5
#define COL_TAGS             6
#define COL_IMPORTANCE       7
#define COL_RELATED_CONCEPTS 8

#define PHRASE_PREFIXES_LOGICAL "phrasing.query_prefixes"
#define PHRASE_MARKERS_LOGICAL  "phrasing.chunk_markers"

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void json_print_str(const char *s)
{
    if (!s) s = "";
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

static char *get_str(tde_handle_t h, int row, int col)
{
    int need = tde_get_string(h, row, col, NULL, 0);
    if (need <= 0) return NULL;
    char *buf = malloc((size_t)need);
    if (!buf) return NULL;
    if (tde_get_string(h, row, col, buf, need) <= 0) { free(buf); return NULL; }
    return buf;
}

static void lname(char *out, int sz, const char *db, const char *tbl)
{ snprintf(out, sz, "%s.%s", db, tbl); }

/* Read porool.ocfg via SLispManager — configure THARAVU base path + vocab name.
 * Falls back to ./knowledge if ocfg is absent or missing the section. */
#define DEFAULT_POROOL_OCFG "porool.ocfg"
static char s_vocab_name[128] = "general.languages";

static void configure_from_ocfg(const char *ocfg_path) {
    slm_node_t *cfg = slm_load(ocfg_path);
    if (!cfg) {
        tde_set_base_path("./knowledge");
        return;
    }
    slm_node_t *pr = slm_find(cfg, "porool");
    if (pr) {
        const char *dd = slm_attr(pr, "data_dir");
        if (dd && dd[0]) tde_set_base_path(dd);
        else             tde_set_base_path("./knowledge");
        const char *vn = slm_attr(pr, "vocab_name");
        if (vn && vn[0]) strncpy(s_vocab_name, vn, sizeof(s_vocab_name) - 1);
    } else {
        tde_set_base_path("./knowledge");
    }
    slm_free(cfg);
}

/* ── print_results ───────────────────────────────────────────────────────── */

static void print_results(SearchResult *results, int count, int topk,
                           int max_chars, const char *query,
                           const char *concept_filter)
{
    printf("{\n  \"query\": ");
    json_print_str(query);
    printf(",\n  \"results\": [");

    int nseen = 0, chars_used = 0, first = 1;
    for (int i = 0; i < count && nseen < topk && chars_used < max_chars; i++) {
        const char *txt = results[i].text;
        if (!txt) continue;

        if (concept_filter && concept_filter[0] &&
            (!results[i].concept ||
             strcasecmp(results[i].concept, concept_filter) != 0))
            continue;

        int dup = 0;
        for (int j = 0; j < i && !dup; j++)
            if (results[j].text && strcmp(results[j].text, txt) == 0) dup = 1;
        if (dup) continue;

        int tlen = (int)strlen(txt);
        if (chars_used + tlen > max_chars) break;

        if (!first) printf(",");
        first = 0;
        printf("\n    {\n");
        printf("      \"rank\": %d,\n", nseen + 1);
        printf("      \"score\": %.6f,\n", results[i].score);
        printf("      \"chunk_id\": ");          json_print_str(results[i].chunk_id);          printf(",\n");
        printf("      \"source\": ");            json_print_str(results[i].source);            printf(",\n");
        printf("      \"concept\": ");           json_print_str(results[i].concept);           printf(",\n");
        printf("      \"section\": ");           json_print_str(results[i].section);           printf(",\n");
        printf("      \"type\": ");              json_print_str(results[i].type);              printf(",\n");
        printf("      \"tags\": ");              json_print_str(results[i].tags);              printf(",\n");
        printf("      \"importance\": ");        json_print_str(results[i].importance);        printf(",\n");
        printf("      \"related_concepts\": ");  json_print_str(results[i].related_concepts);  printf(",\n");
        printf("      \"text\": ");
        json_print_str(txt);
        printf("\n    }");

        chars_used += tlen;
        nseen++;
    }
    printf("\n  ]\n}\n");
}

/* ── cmd_ingest ──────────────────────────────────────────────────────────── */

static int cmd_ingest(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: porool ingest <file> --db <db> --table <table>\n"
            "         [--concept <c>] [--section <s>] [--type <t>]\n"
            "         [--tags <t>] [--importance high|medium|low]\n"
            "         [--related <r>] [--ini <porool.ini>]\n");
        return 1;
    }

    const char *infile = argv[2];
    const char *db = NULL, *table = NULL, *ini = DEFAULT_POROOL_OCFG;
    ChunkMeta meta = {0};

    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i], "--db")        && i+1 < argc) db                    = argv[++i];
        else if (!strcmp(argv[i], "--table")      && i+1 < argc) table                 = argv[++i];
        else if (!strcmp(argv[i], "--ini")        && i+1 < argc) ini                   = argv[++i];
        else if (!strcmp(argv[i], "--concept")    && i+1 < argc) meta.concept          = argv[++i];
        else if (!strcmp(argv[i], "--section")    && i+1 < argc) meta.section          = argv[++i];
        else if (!strcmp(argv[i], "--type")       && i+1 < argc) meta.type             = argv[++i];
        else if (!strcmp(argv[i], "--tags")       && i+1 < argc) meta.tags             = argv[++i];
        else if (!strcmp(argv[i], "--importance") && i+1 < argc) meta.importance       = argv[++i];
        else if (!strcmp(argv[i], "--related")    && i+1 < argc) meta.related_concepts = argv[++i];
    }
    if (!db || !table) {
        fprintf(stderr, "porool: --db and --table required\n");
        return 1;
    }

    porool_config_t pcfg;
    porool_config_defaults(&pcfg);
    porool_t *p = porool_create(&pcfg);
    if (!p) {
        fprintf(stderr, "porool: init failed\n");
        return 1;
    }

    int rc = porool_ingest_with_meta(p, infile, db, table, &meta);
    porool_destroy(p);

    if (rc < 0) {
        fprintf(stderr, "porool: ingest failed (%d)\n", rc);
        return 1;
    }
    if (rc == 1) {
        printf("porool: %s already ingested into %s.%s — skipping\n", infile, db, table);
        return 0;
    }
    printf("porool: ingested %s into %s.%s\n", infile, db, table);
    return 0;
}

/* ── cmd_query ───────────────────────────────────────────────────────────── */

static int cmd_query(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: porool query <text> [--db <db>] [--table <table>]"
            " [--topk N] [--max-chars N] [--concept <c>] [--ini <porool.ini>]\n"
            "       Omit --table to search all tables in the db.\n"
            "       Omit --db to search every registered table.\n");
        return 1;
    }

    const char *query_text     = argv[2];
    const char *db = NULL, *table = NULL, *ini = DEFAULT_POROOL_OCFG;
    const char *concept_filter = NULL;
    int topk = DEFAULT_TOPK, max_chars = DEFAULT_MAX_CHARS;

    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i], "--db")        && i+1 < argc) db             = argv[++i];
        else if (!strcmp(argv[i], "--table")     && i+1 < argc) table          = argv[++i];
        else if (!strcmp(argv[i], "--ini")       && i+1 < argc) ini            = argv[++i];
        else if (!strcmp(argv[i], "--concept")   && i+1 < argc) concept_filter = argv[++i];
        else if (!strcmp(argv[i], "--topk")      && i+1 < argc) topk           = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-chars") && i+1 < argc) max_chars      = atoi(argv[++i]);
    }
    if (topk <= 0)      topk      = DEFAULT_TOPK;
    if (max_chars <= 0) max_chars = DEFAULT_MAX_CHARS;

    porool_config_t pcfg;
    porool_config_defaults(&pcfg);
    porool_t *p = porool_create(&pcfg);
    if (!p) {
        fprintf(stderr, "porool: init failed\n");
        return 1;
    }

    /* Build target string: "ALL", "db", or "db.table" */
    char target[512] = "ALL";
    if (db && *db && strcasecmp(db, "all") != 0) {
        if (table && *table && strcasecmp(table, "all") != 0)
            snprintf(target, sizeof(target), "%s.%s", db, table);
        else
            snprintf(target, sizeof(target), "%s", db);
    }

    float *qv = NULL;
    int    dim = 0;
    if (porool_embed_query(p, query_text, &qv, &dim) != 0 || !qv) {
        fprintf(stderr, "porool: embedding failed\n");
        porool_destroy(p);
        return 1;
    }

    int count = 0;
    SearchResult *results = porool_retrieve_target(p, qv, target, topk, &count);
    free(qv);  /* allocated by porool_embed_query via malloc */

    if (!results || count == 0) {
        printf("{\"query\": ");
        json_print_str(query_text);
        printf(", \"results\": []}\n");
        if (results) porool_free_results(results, count);
        porool_destroy(p);
        return 0;
    }

    porool_rerank_query(p, results, count, query_text);
    print_results(results, count, topk, max_chars, query_text, concept_filter);
    porool_free_results(results, count);
    porool_destroy(p);
    return 0;
}

/* ── cmd_stats ───────────────────────────────────────────────────────────── */

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
    lname(cl, sizeof(cl), db, table);

    /* tde_config_load removed 2026-05-29; use tde default base path "./data" */
    (void)ini;

    tde_handle_t ch = tde_open_odat(cl);
    tde_handle_t ov = tde_open_ovec(cl);

    int      chunk_n = ch ? tde_row_count(ch) : -1;
    uint32_t dim     = 0;
    if (ov && chunk_n > 0) tde_vector_get_ptr(ov, 0, &dim);

    printf("Table:   %s\n", cl);
    printf("Chunks:  %d\n", chunk_n);
    printf("Vectors: %d\n", chunk_n);
    if (dim) printf("Dim:     %u\n", dim);

    if (ch) tde_close(ch);
    if (ov) tde_close(ov);
    return 0;
}

/* ── cmd_peek ────────────────────────────────────────────────────────────── */

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
        fprintf(stderr, "Usage: porool peek --db <db> --table <table> --id <id>\n");
        return 1;
    }

    char cl[256];
    lname(cl, sizeof(cl), db, table);

    /* tde_config_load removed 2026-05-29; use tde default base path "./data" */
    (void)ini;

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

    char *txt     = get_str(ch, id, COL_TEXT);
    char *src     = get_str(ch, id, COL_SOURCE);
    char *cid     = get_str(ch, id, COL_CHUNK_ID);
    char *concept = get_str(ch, id, COL_CONCEPT);
    char *section = get_str(ch, id, COL_SECTION);
    char *type    = get_str(ch, id, COL_TYPE);
    char *tags    = get_str(ch, id, COL_TAGS);
    char *imp     = get_str(ch, id, COL_IMPORTANCE);
    char *related = get_str(ch, id, COL_RELATED_CONCEPTS);

    printf("ID:               %d\n",  id);
    printf("Chunk ID:         %s\n",  cid     ? cid     : "");
    printf("Source:           %s\n",  src     ? src     : "unknown");
    printf("Concept:          %s\n",  concept ? concept : "");
    printf("Section:          %s\n",  section ? section : "");
    printf("Type:             %s\n",  type    ? type    : "");
    printf("Tags:             %s\n",  tags    ? tags    : "");
    printf("Importance:       %s\n",  imp     ? imp     : "");
    printf("Related Concepts: %s\n",  related ? related : "");
    printf("Text:             %s\n",  txt     ? txt     : "(empty)");

    free(txt); free(src); free(cid); free(concept); free(section);
    free(type); free(tags); free(imp); free(related);
    tde_close(ch);
    return 0;
}

/* ── cmd_phrasing ────────────────────────────────────────────────────────── */

static int cmd_phrasing(int argc, char **argv)
{
    const char *action  = argc >= 3 ? argv[2] : "list";
    const char *pattern = argc >= 4 ? argv[3] : NULL;
    const char *ini     = DEFAULT_INI;  /* list uses tde directly */

    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i], "--ini")     && i+1 < argc) ini     = argv[++i];
        else if (!strcmp(argv[i], "--pattern") && i+1 < argc) pattern = argv[++i];
    }

    if (!strcmp(action, "list")) {
        /* tde_config_load removed 2026-05-29; use tde default base path */
        (void)ini;
        const char *tables[] = { PHRASE_PREFIXES_LOGICAL, PHRASE_MARKERS_LOGICAL };
        const char *labels[] = { "query_prefixes", "chunk_markers" };
        for (int t = 0; t < 2; t++) {
            printf("[%s]\n", labels[t]);
            tde_handle_t h = tde_open_odat(tables[t]);
            if (!h) { printf("  (table not found)\n\n"); continue; }
            int n = tde_row_count(h);
            for (int i = 0; i < n; i++) {
                int need = tde_get_string(h, i, 0, NULL, 0);
                if (need <= 0) continue;
                char *s = malloc((size_t)need);
                if (!s) continue;
                if (tde_get_string(h, i, 0, s, need) > 0) printf("  [%d] %s\n", i, s);
                free(s);
            }
            tde_close(h);
            printf("\n");
        }
        return 0;
    }

    if (!pattern) {
        fprintf(stderr, "porool phrasing %s: pattern required\n", action);
        return 1;
    }

    int is_prefix;
    if      (!strcmp(action, "add-prefix")) is_prefix = 1;
    else if (!strcmp(action, "add-marker")) is_prefix = 0;
    else {
        fprintf(stderr,
            "porool phrasing: unknown action '%s'\n"
            "  Usage: porool phrasing list\n"
            "         porool phrasing add-prefix \"pattern\"\n"
            "         porool phrasing add-marker \"marker\"\n", action);
        return 1;
    }

    /* add-* needs porool instance */
    porool_config_t pcfg;
    porool_config_defaults(&pcfg);
    porool_t *p = porool_create(&pcfg);
    if (!p) {
        fprintf(stderr, "porool: init failed\n");
        return 1;
    }

    int rc = porool_phrasing_add(p, pattern, is_prefix);
    porool_destroy(p);

    if      (rc == -1) { fprintf(stderr, "porool: duplicate — '%s' already exists\n", pattern); return 1; }
    else if (rc == -2) { fprintf(stderr, "porool: failed to save phrasing table\n"); return 1; }
    printf("porool: added '%s' to %s\n", pattern,
           is_prefix ? PHRASE_PREFIXES_LOGICAL : PHRASE_MARKERS_LOGICAL);
    return 0;
}

/* ── interactive wizard ──────────────────────────────────────────────────── */

static void trim_line(char *s)
{
    int n = (int)strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' || s[n-1] == ' ')) s[--n] = '\0';
    char *p = s;
    while (*p == ' ') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    n = (int)strlen(s);
    if (n >= 2 && s[0] == '"' && s[n-1] == '"') {
        memmove(s, s+1, (size_t)(n-2)); s[n-2] = '\0';
    }
}

static int read_line(const char *prompt, char *buf, int size)
{
    printf("%s", prompt);
    fflush(stdout);
    if (!fgets(buf, size, stdin)) return 0;
    trim_line(buf);
    return 1;
}

static int parse_dbtable(const char *s, char *db, int dbsz, char *tbl, int tblsz)
{
    const char *dot = strchr(s, '.');
    if (!dot || dot == s) return 0;
    int dlen = (int)(dot - s);
    if (dlen >= dbsz || (int)strlen(dot+1) >= tblsz) return 0;
    memcpy(db, s, (size_t)dlen); db[dlen] = '\0';
    strncpy(tbl, dot+1, (size_t)(tblsz-1)); tbl[tblsz-1] = '\0';
    return db[0] && tbl[0];
}

static void cmd_interactive(void)
{
    int topk      = DEFAULT_TOPK;
    int max_chars = DEFAULT_MAX_CHARS;

    printf("porool RAG engine\n");
    printf("-----------------\n\n");

    char choice[64];
    for (;;) {
        printf("  [r] Retrieve   [i] Ingest   [q] Quit\n");
        if (!read_line("Method: ", choice, sizeof(choice))) break;
        printf("\n");
        if (!choice[0]) continue;

        if (choice[0] == 'q' || choice[0] == 'Q') { printf("bye\n"); break; }

        if (choice[0] == 'r' || choice[0] == 'R') {
            char query[4096];
            if (!read_line("Query: ", query, sizeof(query))) break;
            if (!query[0]) { printf("(cancelled)\n\n"); continue; }

            char dbtbl[512];
            for (;;) {
                if (!read_line("DB.Table (db / db.table / all): ", dbtbl, sizeof(dbtbl))) goto done;
                if (dbtbl[0]) break;
                printf("  (required)\n");
            }
            printf("\n");

            char db[256] = "", table[256] = "";
            if (!strcasecmp(dbtbl, "all")) {
                /* both empty → ALL target */
            } else {
                char *dot = strchr(dbtbl, '.');
                if (dot && dot != dbtbl && *(dot+1)) {
                    int dlen = (int)(dot - dbtbl);
                    if (dlen > 255) dlen = 255;
                    strncpy(db, dbtbl, dlen); db[dlen] = '\0';
                    strncpy(table, dot+1, 255); table[255] = '\0';
                } else {
                    strncpy(db, dbtbl, 255); db[255] = '\0';
                }
            }

            char concept[256] = "";
            read_line("Concept filter (Enter to skip): ", concept, sizeof(concept));

            char topk_s[16], maxc_s[16];
            snprintf(topk_s, sizeof(topk_s), "%d", topk);
            snprintf(maxc_s, sizeof(maxc_s), "%d", max_chars);

            const char *av[20];
            int ac = 0;
            av[ac++] = "porool"; av[ac++] = "query"; av[ac++] = query;
            if (db[0])      { av[ac++] = "--db";        av[ac++] = db;      }
            if (table[0])   { av[ac++] = "--table";     av[ac++] = table;   }
            av[ac++] = "--topk";       av[ac++] = topk_s;
            av[ac++] = "--max-chars";  av[ac++] = maxc_s;
            if (concept[0]) { av[ac++] = "--concept";   av[ac++] = concept; }
            cmd_query(ac, (char **)av);
            printf("\n");
            continue;
        }

        if (choice[0] == 'i' || choice[0] == 'I') {
            char filepath[4096];
            if (!read_line("File: ", filepath, sizeof(filepath))) break;
            if (!filepath[0]) { printf("(cancelled)\n\n"); continue; }

            char dbtbl[512];
            if (!read_line("DB.Table (e.g. mydb.docs): ", dbtbl, sizeof(dbtbl))) break;

            char db[256] = "", table[256] = "";
            if (!parse_dbtable(dbtbl, db, sizeof(db), table, sizeof(table))) {
                printf("Error: enter DB.Table in the form  dbname.tablename\n\n");
                continue;
            }

            printf("\nMetadata (press Enter to skip any field):\n");
            char concept[256]="", section[256]="", mtype[256]="";
            char tags[512]="", imp[32]="", related[512]="";
            read_line("  Concept (e.g. NERF): ",          concept, sizeof(concept));
            read_line("  Section (e.g. description): ",   section, sizeof(section));
            read_line("  Type (e.g. definition): ",       mtype,   sizeof(mtype));
            read_line("  Tags (comma-separated): ",       tags,    sizeof(tags));
            read_line("  Importance (high/medium/low): ", imp,     sizeof(imp));
            read_line("  Related concepts (comma-sep): ", related, sizeof(related));
            printf("\n");

            const char *av[] = {
                "porool", "ingest", filepath,
                "--db",         db,
                "--table",      table,
                "--concept",    concept,
                "--section",    section,
                "--type",       mtype,
                "--tags",       tags,
                "--importance", imp,
                "--related",    related
            };
            cmd_ingest((int)(sizeof(av)/sizeof(av[0])), (char **)av);
            printf("\n");
            continue;
        }

        printf("Type r, i, or q.\n\n");
    }
done:;
}

/* ── usage / main ────────────────────────────────────────────────────────── */

static void usage(void)
{
    fprintf(stderr,
        "porool - RAG document ingestion and query\n\n"
        "  porool ingest <file> --db <db> --table <table>\n"
        "                [--concept <c>] [--section <s>] [--type <t>]\n"
        "                [--tags <t>] [--importance high|medium|low]\n"
        "                [--related <r>] [--ini <porool.ini>]\n"
        "  porool query  <text> [--db <db>] [--table <table>]\n"
        "                [--topk N] [--max-chars N] [--concept <c>]\n"
        "                [--ini <porool.ini>]\n"
        "  porool stats  --db <db> --table <table> [--ini <tharavu.ini>]\n"
        "  porool peek   --db <db> --table <table> --id <id> [--ini <tharavu.ini>]\n"
        "  porool phrasing list [--ini <tharavu.ini>]\n"
        "  porool phrasing add-prefix|add-marker <pattern> [--ini <porool.ini>]\n\n"
        "Run with no arguments to enter interactive mode.\n\n"
        "Supported file types (via porool.dll):\n"
        "  .txt  .pdf  .docx  .xlsx  .jpg  .jpeg  .png\n"
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
    /* Read porool.ocfg — sets tde base path + vocab name.  No porool.ini created. */
    configure_from_ocfg(DEFAULT_POROOL_OCFG);
    if (ve_init(s_vocab_name, NULL) != 0) {
        fprintf(stderr, "porool: SORKUVAI init failed\n");
        return 1;
    }

    int rc = 1;
    if (argc < 2) {
#ifdef _WIN32
        (void)launched_from_explorer();
#endif
        cmd_interactive();
        rc = 0;
    } else if (!strcmp(argv[1], "ingest"))   { rc = cmd_ingest   (argc, argv); }
    else if   (!strcmp(argv[1], "query"))    { rc = cmd_query    (argc, argv); }
    else if   (!strcmp(argv[1], "stats"))    { rc = cmd_stats    (argc, argv); }
    else if   (!strcmp(argv[1], "peek"))     { rc = cmd_peek     (argc, argv); }
    else if   (!strcmp(argv[1], "phrasing")) { rc = cmd_phrasing (argc, argv); }
    else {
        fprintf(stderr, "porool: unknown command '%s'\n\n", argv[1]);
        usage();
    }

    ve_cleanup();
    return rc;
}
