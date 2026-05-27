/*
 * test/test_porool.c — Porool integration test suite
 *
 * Run via: make test  (executes from build/ directory)
 *
 * Isolated data: table prefix "test_p" → build/data/test_p/
 * That subtree is created before tests and removed at the end.
 * Existing build/data/ content is not touched.
 *
 * Covers:
 *   porool_init / porool_shutdown
 *   porool_ingest_with_meta (dedup check)
 *   porool_embed_query
 *   porool_retrieve + porool_rerank_query
 *   porool_build_context
 *   porool_query / porool_query_from / porool_query_target
 *   porool_synonym_add
 *   porool_phrasing_add + porool_phrasing_reload
 *   porool_refresh
 *   ve_reload_vocab → porool_refresh → porool_query  (Gap-F pipeline)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../src/include/porool.h"
#include "../src/include/sorkuvai_dll.h"   /* ve_reload_vocab() */

/* ── Counters ────────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;

#define CHECK(label, expr) do {                                             \
    int _ok = !!(expr);                                                     \
    printf("  %s  %s\n", _ok ? "PASS" : "FAIL", (label));                  \
    if (_ok) g_pass++; else g_fail++;                                       \
} while (0)

/* ── Test corpus ─────────────────────────────────────────────────────────── */

static const char *k_doc_nerf =
    "NeRF (Neural Radiance Field) is a method for synthesizing novel views "
    "of complex scenes by representing the scene as a continuous volumetric "
    "function parameterized by a neural network. The network accepts 5D input "
    "(spatial position x,y,z and viewing direction) and outputs volume density "
    "and view-dependent emitted radiance. NeRF training requires posed images "
    "and optimizes a rendering loss end-to-end.";

static const char *k_doc_diffusion =
    "Diffusion models are generative models that learn to reverse a gradual "
    "noising process. Starting from Gaussian noise, the model denoises step "
    "by step until a clean sample is produced. Stable Diffusion is a latent "
    "diffusion model that operates in a compressed latent space, enabling "
    "high-resolution image synthesis at lower computational cost.";

/* ── Setup helpers ───────────────────────────────────────────────────────── */

static void write_file(const char *path, const char *content)
{
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "FATAL: cannot create %s\n", path); exit(1); }
    fputs(content, fp);
    fclose(fp);
}

static void write_test_ini(const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "FATAL: cannot create %s\n", path); exit(1); }
    fprintf(fp,
        "[porool]\n"
        "tharavu_ini           = tharavu.ini\n"
        "data_dir              = ./data\n"
        "vocab_name            = test_p.vocab\n"
        "chunks_table          = test_p.chunks\n"
        "vectors_table         = test_p.chunks\n"
        "top_k_default         = 5\n"
        "max_context_chars     = 2000\n"
        "optimal_chunk_len     = 300.0\n"
        "length_penalty_sigma  = 150.0\n"
        "default_source_weight = 1.0\n"
        "[scoring]\n"
        "w1 = 0.60\n"
        "w2 = 0.20\n"
        "w3 = 0.20\n"
        "w4 = 0.15\n"
        "w5 = 0.15\n");
    fclose(fp);
}

static void cleanup(void)
{
#ifdef _WIN32
    system("rmdir /s /q .\\data\\test_p 2>nul");
#else
    system("rm -rf ./data/test_p");
#endif
    remove("test_porool.ini");
    remove("testdoc_nerf.txt");
    remove("testdoc_diffusion.txt");
}

/* ── Test sections ───────────────────────────────────────────────────────── */

static void test_pre_init(void)
{
    printf("\n[pre-init error cases]\n");
    CHECK("init(NULL) returns -1",   porool_init(NULL) == -1);
    CHECK("refresh before init returns -1", porool_refresh() == -1);
}

static void test_init(void)
{
    printf("\n[init]\n");
    int r = porool_init("test_porool.ini");
    CHECK("porool_init returns 0", r == 0);
}

static void test_ingest(void)
{
    printf("\n[ingest]\n");

    ChunkMeta m1 = {
        .concept          = "NeRF",
        .section          = "description",
        .type             = "definition",
        .tags             = "vision,3d,rendering",
        .importance       = "high",
        .related_concepts = "Diffusion",
        .language         = "english"
    };
    ChunkMeta m2 = {
        .concept          = "Diffusion",
        .section          = "description",
        .type             = "definition",
        .tags             = "generative,vision",
        .importance       = "high",
        .related_concepts = "NeRF"
        /* .language not set → falls back to first configured language */
    };

    int r1 = porool_ingest_with_meta("testdoc_nerf.txt",
                                     "test_p", "chunks", &m1);
    CHECK("ingest nerf doc returns 0", r1 == 0);

    int r2 = porool_ingest_with_meta("testdoc_diffusion.txt",
                                     "test_p", "chunks", &m2);
    CHECK("ingest diffusion doc returns 0", r2 == 0);

    /* Same file a second time must be skipped (returns 1) */
    int r3 = porool_ingest_with_meta("testdoc_nerf.txt",
                                     "test_p", "chunks", &m1);
    CHECK("re-ingest same file returns 1 (skipped)", r3 == 1);

    /* Tables were absent at init time so g_chunks/g_vectors are NULL.
     * porool_refresh() re-opens them now that the files exist — this is
     * the designed post-ingest pattern for a fresh table. */
    int rr = porool_refresh();
    CHECK("refresh after first ingest returns 0", rr == 0);
}

static void test_embed_query(void)
{
    printf("\n[embed_query]\n");

    float *emb = NULL;
    int    dim  = 0;
    int    r    = porool_embed_query("what is NeRF", &emb, &dim);

    CHECK("embed_query returns 0",    r == 0);
    CHECK("embedding non-NULL",       emb != NULL);
    CHECK("dimension > 0",            dim > 0);

    if (emb) {
        float sumsq = 0.0f;
        for (int i = 0; i < dim; i++) sumsq += emb[i] * emb[i];
        CHECK("embedding has nonzero L2 norm", sumsq > 0.0f);
        porool_free((char *)emb);
    }
}

static void test_retrieve_rerank(void)
{
    printf("\n[retrieve + rerank_query]\n");

    float *emb = NULL;
    int    dim  = 0;
    if (porool_embed_query("neural radiance field volume density",
                            &emb, &dim) != 0 || !emb) {
        printf("  SKIP  (embed_query failed)\n");
        return;
    }

    int           count   = 0;
    SearchResult *results = porool_retrieve(emb, 5, &count);
    porool_free((char *)emb);

    CHECK("retrieve returns non-NULL", results != NULL);
    CHECK("retrieve count > 0",        count > 0);

    if (!results || count == 0) return;

    porool_rerank_query(results, count, "what is neural radiance field");
    CHECK("top result has text",     results[0].text     != NULL);
    CHECK("top result has source",   results[0].source   != NULL);
    CHECK("top result has language", results[0].language != NULL);
    CHECK("top result score in [0,2]",
          results[0].score >= 0.0f && results[0].score <= 2.0f);

    /* NeRF doc should appear in the top-3 for a NeRF query */
    int found = 0;
    for (int i = 0; i < count && i < 3; i++) {
        if (results[i].text && strstr(results[i].text, "NeRF")) {
            found = 1;
            break;
        }
    }
    CHECK("NeRF doc in top-3 for NeRF query", found);

    /* Scores should be in descending order after rerank */
    int sorted = 1;
    for (int i = 1; i < count; i++) {
        if (results[i].score > results[i-1].score + 1e-6f) {
            sorted = 0;
            break;
        }
    }
    CHECK("results sorted descending by score", sorted);

    porool_free_results(results, count);
}

static void test_build_context(void)
{
    printf("\n[build_context]\n");

    float *emb = NULL;
    int    dim  = 0;
    if (porool_embed_query("generative model gaussian noise latent",
                            &emb, &dim) != 0 || !emb) {
        printf("  SKIP  (embed_query failed)\n");
        return;
    }

    int           count   = 0;
    SearchResult *results = porool_retrieve(emb, 5, &count);
    porool_free((char *)emb);

    if (!results || count == 0) {
        printf("  SKIP  (retrieve returned nothing)\n");
        return;
    }

    const int max_chars = 400;
    char *ctx = porool_build_context(results, count, max_chars);
    porool_free_results(results, count);

    CHECK("build_context returns non-NULL", ctx != NULL);
    if (ctx) {
        CHECK("context is non-empty",       ctx[0] != '\0');
        CHECK("context respects max_chars", (int)strlen(ctx) <= max_chars);
        porool_free(ctx);
    }
}

static void test_query(void)
{
    printf("\n[query]\n");

    char *ctx = porool_query("what is a diffusion model", 5, 800);
    CHECK("query returns non-NULL", ctx != NULL);
    if (ctx) {
        int has_content = strstr(ctx, "iffusion") != NULL ||
                          strstr(ctx, "noise")    != NULL ||
                          strstr(ctx, "latent")   != NULL;
        CHECK("context contains diffusion-related content", has_content);
        porool_free(ctx);
    }

    /* default top_k and max_chars (≤0 triggers INI defaults) */
    char *ctx2 = porool_query("NeRF rendering", 0, 0);
    CHECK("query with default top_k/chars returns non-NULL", ctx2 != NULL);
    if (ctx2) porool_free(ctx2);
}

static void test_query_from(void)
{
    printf("\n[query_from]\n");

    char *ctx = porool_query_from("NeRF novel view synthesis",
                                   "test_p", "chunks", 5, 800);
    CHECK("query_from specific table returns non-NULL", ctx != NULL);
    if (ctx) porool_free(ctx);

    /* NULL table_name → all registered tables for the db */
    char *ctx2 = porool_query_from("volumetric rendering neural",
                                    "test_p", NULL, 5, 800);
    CHECK("query_from NULL table returns non-NULL", ctx2 != NULL);
    if (ctx2) porool_free(ctx2);
}

static void test_query_target(void)
{
    printf("\n[query_target]\n");

    char *r1 = porool_query_target("diffusion model generative noise", "ALL", 5, 800);
    CHECK("query_target ALL returns non-NULL", r1 != NULL);
    if (r1) porool_free(r1);

    char *r2 = porool_query_target("radiance field view synthesis", "test_p", 5, 800);
    CHECK("query_target db returns non-NULL", r2 != NULL);
    if (r2) porool_free(r2);

    char *r3 = porool_query_target("volumetric density rendering",
                                    "test_p.chunks", 5, 800);
    CHECK("query_target db.table returns non-NULL", r3 != NULL);
    if (r3) porool_free(r3);
}

static void test_synonym(void)
{
    printf("\n[synonym_add]\n");

    /* After fresh init the custom synonym table is empty */
    int r1 = porool_synonym_add("nerf", "neural radiance field");
    CHECK("first synonym_add returns 0",        r1 == 0);

    int r2 = porool_synonym_add("nerf", "neural radiance field");
    CHECK("duplicate synonym_add returns -1",   r2 == -1);

    int r3 = porool_synonym_add(NULL, "something");
    CHECK("synonym_add NULL 'from' returns -2", r3 == -2);

    int r4 = porool_synonym_add("something", NULL);
    CHECK("synonym_add NULL 'to' returns -2",   r4 == -2);

    /* Query using the synonym should still return results */
    char *ctx = porool_query("nerf", 3, 400);
    CHECK("query with synonym term returns non-NULL", ctx != NULL);
    if (ctx) porool_free(ctx);
}

static void test_phrasing(void)
{
    printf("\n[phrasing_add + phrasing_reload]\n");

    /* "tell me about " is not a default prefix — first add should succeed */
    int r1 = porool_phrasing_add("tell me about ", 1);
    CHECK("phrasing_add new prefix 0 or -1 (dup on re-run)", r1 == 0 || r1 == -1);

    /* " is used for " is not a default marker */
    int r2 = porool_phrasing_add(" is used for ", 0);
    CHECK("phrasing_add new marker 0 or -1 (dup on re-run)", r2 == 0 || r2 == -1);

    porool_phrasing_reload();
    CHECK("phrasing_reload does not crash", 1);
}

static void test_refresh(void)
{
    printf("\n[refresh]\n");

    int r = porool_refresh();
    CHECK("refresh after init returns 0", r == 0);

    /* query must still work after refresh (handles re-opened) */
    char *ctx = porool_query("NeRF volume density neural network", 3, 400);
    CHECK("query after refresh returns non-NULL", ctx != NULL);
    if (ctx) porool_free(ctx);
}

/*
 * Gap-F pipeline: simulate the SORPAYIR export → Sorkuvai reload → Porool query
 * path without needing SORPAYIR.  We call ve_reload_vocab() directly (Sorkuvai
 * DLL) to exercise the hot-reload path, then porool_refresh() to re-open the
 * chunk/vector handles, then verify a full query still works.
 */
static void test_sorkuvai_reload_pipeline(void)
{
    printf("\n[Sorkuvai reload -> porool_refresh -> query  (Gap-F pipeline)]\n");

    int r1 = ve_reload_vocab();
    CHECK("ve_reload_vocab returns VE_OK (0)", r1 == 0);

    int r2 = porool_refresh();
    CHECK("porool_refresh after vocab reload returns 0", r2 == 0);

    char *ctx = porool_query("diffusion model latent stable", 3, 400);
    CHECK("query after reload+refresh returns non-NULL", ctx != NULL);
    if (ctx) {
        CHECK("query result is non-empty after reload", ctx[0] != '\0');
        porool_free(ctx);
    }
}

static void test_shutdown(void)
{
    printf("\n[shutdown]\n");

    porool_shutdown();
    CHECK("shutdown does not crash", 1);

    /* After shutdown the engine must refuse further operations */
    CHECK("refresh after shutdown returns -1", porool_refresh() == -1);
}

/* ── Entry point ─────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Porool integration test suite ===\n");

    /* Prepare isolated test files (runs from bin/) */
    cleanup();
    write_file("testdoc_nerf.txt",      k_doc_nerf);
    write_file("testdoc_diffusion.txt", k_doc_diffusion);
    write_test_ini("test_porool.ini");

    /* Execute in dependency order */
    test_pre_init();
    test_init();
    test_ingest();
    test_embed_query();
    test_retrieve_rerank();
    test_build_context();
    test_query();
    test_query_from();
    test_query_target();
    test_synonym();
    test_phrasing();
    test_refresh();
    test_sorkuvai_reload_pipeline();
    test_shutdown();

    /* Remove test data */
    cleanup();

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
