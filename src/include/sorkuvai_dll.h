/*
 * sorkuvai_dll.h  —  Public API: Tokenizer + Vocabulary + Embedding Engine
 *
 * "sorkuvai" (சொற்குவை) — Tamil: word store / vocabulary repository.
 *
 * Provides:
 *   Tokenization       — delimiter-driven, config loaded from {db}.tokenize.odat
 *   Vocabulary         — persistent word ↔ token_id mapping via tharavu.dll
 *   Embedding          — dense float vectors, Xavier-initialized, deterministic
 *   Similarity         — cosine similarity between any two token vectors
 *
 * Storage layout (all files created at runtime by sorkuvai):
 *   {db}.ovoc            vocabulary hash table: word → token_id  (O(1) lookup)
 *                        reverse index: token_id → word
 *                        record layout: [token_id u32][key_len u16][flags u16]
 *                                       [vector float*dim][word]
 *                        flags: domain enum — see [vocab_flags] in INI
 *   {db}.tokenize.odat   optional: col 0 = delimiter char/sequence per row
 *
 * Example: ve_init("general.english", ...) uses
 *   data/general/english.ovoc  (contains word + token_id + flag + dense vector)
 *
 * ── Memory ownership ──────────────────────────────────────────────────────
 *   Buffers returned by ve_tokenize / ve_process_text MUST be released with
 *   the matching ve_free_tokens / ve_free_ids.  Never call free() directly
 *   on DLL-allocated memory — CRT heap boundary rules apply.
 *
 * ── Threading ─────────────────────────────────────────────────────────────
 *   ve_last_error() is thread-local.
 *
 *   Lifecycle / configuration calls — call from ONE thread only, never
 *   concurrently with any other API call:
 *     ve_init, ve_cleanup, ve_set_tokenizer, sk_set_log_fn
 *
 *   All remaining API calls are fully thread-safe and may be issued
 *   concurrently from any number of threads:
 *     ve_tokenize, ve_get_token_id, ve_process_text, ve_get_word,
 *     ve_get_vector, ve_cosine_similarity, sk_get_dim, sk_vocab_size,
 *     ve_generate_vector, ve_free_tokens, ve_free_ids, ve_strerror,
 *     ve_last_error
 *
 *   Custom tokenizers installed via ve_set_tokenizer are called WITHOUT
 *   the internal lock, so they may safely call back into the sorkuvai API.
 */

#ifndef SORKUVAI_DLL_H
#define SORKUVAI_DLL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Export / calling-convention macros ──────────────────────────────────── */

#ifdef _WIN32
#  ifdef SORKUVAI_EXPORTS
#    define SK_API __declspec(dllexport)
#  else
#    define SK_API __declspec(dllimport)
#  endif
#  define SK_CALL __cdecl
#else
#  define SK_API  __attribute__((visibility("default")))
#  define SK_CALL
#endif

/* ── Library version ─────────────────────────────────────────────────────── */

#define SK_VERSION_MAJOR  1
#define SK_VERSION_MINOR  0
#define SK_VERSION_PATCH  0
#define SK_VERSION_STRING "1.0.0"

/* Minimum tharavu (tde) version required at runtime.
 * tde_vocab_reverse_lookup_ex and all file-builder APIs are present in the
 * DLL from v1.0.  The "v1.1/v1.2" labels in the Tharavu source refer to the
 * .ovoc / .ovec file-format version written into files, not to the DLL API
 * version reported by tde_version_major() / tde_version_minor().             */
#define SK_MIN_TDE_MAJOR  1
#define SK_MIN_TDE_MINOR  0

/* ── Embedding dimension ─────────────────────────────────────────────────── */

/* Default vector dimension.  Recompile sorkuvai.dll with -DSK_DEFAULT_DIM=N
 * to change.  Consumers call sk_get_dim() to discover the active value.     */
#ifndef SK_DEFAULT_DIM
#  define SK_DEFAULT_DIM 64
#endif

/* ── Error codes ─────────────────────────────────────────────────────────── */

#define VE_OK            0
#define VE_ERR_INIT     -1   /* init / tde_* setup failed              */
#define VE_ERR_MEM      -2   /* memory allocation failed               */
#define VE_ERR_NOTFOUND -3   /* word or token_id not in vocab          */
#define VE_ERR_INVAL    -4   /* NULL or invalid argument               */
#define VE_ERR_STATE    -5   /* called before ve_init                  */

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

/* Initialise the engine.
 *
 * vocab_logical_name : logical name of the vocabulary, e.g. "rag_db.vocab".
 *                      Resolves to {data_dir}/rag_db/vocab.odat (ODAT table)
 *                      and {data_dir}/rag_db/vocab_vec.ovec (vector store).
 *                      The files are created automatically on first run.
 * ini_path           : path to tharavu.ini.  Pass NULL if tde_config_load()
 *                      has already been called by the host application.
 *
 * Returns VE_OK or a negative VE_ERR_* code.
 * Must be called once before any other ve_* / sk_* function.               */
SK_API int  SK_CALL ve_init(const char* vocab_logical_name,
                             const char* ini_path);

/* Release all resources, flush vocab to disk, rebuild and save .ovec.       */
SK_API void SK_CALL ve_cleanup(void);

/* ── Tokenization ────────────────────────────────────────────────────────── */

/* Split text into tokens using delimiters loaded from {db}.tokenize.odat.
 * Falls back to a built-in default set if the table does not exist.
 *
 * On success *tokens is a DLL-allocated array of *count NUL-terminated
 * strings.  Release with ve_free_tokens(*tokens, *count).                   */
SK_API int  SK_CALL ve_tokenize(const char* text,
                                 char***     tokens,
                                 int*        count);

/* Release the array returned by ve_tokenize.  Always use this — never free().*/
SK_API void SK_CALL ve_free_tokens(char** tokens, int count);

/* ── Vocabulary: word → token_id ─────────────────────────────────────────── */

/* Look up word in the persistent vocabulary.
 *
 * If found     : returns the existing token_id (>= 0).
 * If not found : assigns a new incremental token_id, generates a
 *                deterministic Xavier embedding vector, inserts the row
 *                into vocab.odat, and persists it immediately.
 *
 * Returns token_id (>= 0) or a negative VE_ERR_* code.                     */
SK_API int SK_CALL ve_get_token_id(const char* word);

/* ── Vocabulary: token_id → word ─────────────────────────────────────────── */

/* Copy the word for token_id into buffer.
 *
 * Two-call pattern (identical to tde_get_string):
 *   1. buffer = NULL, buffer_size = 0  → returns required buffer size
 *      (including the NUL terminator).
 *   2. buffer != NULL with that size   → copies the word, returns VE_OK.
 *
 * Required for decoding transformer predictions back to human-readable text. */
SK_API int SK_CALL ve_get_word(uint32_t token_id,
                                char*    buffer,
                                int      buffer_size);

/* ── Embeddings ──────────────────────────────────────────────────────────── */

/* Copy the stored embedding vector for token_id into out_vec.
 * dim must equal sk_get_dim().  Returns VE_OK or VE_ERR_*.                 */
SK_API int SK_CALL ve_get_vector(uint32_t token_id,
                                  float*   out_vec,
                                  int      dim);

/* Generate the deterministic Xavier embedding for word into out_vec.
 * Uses FNV-1a hash of the word as the PRNG seed — same word always
 * produces the same vector regardless of vocab or session state.
 * dim must equal sk_get_dim().                                              */
SK_API void SK_CALL ve_generate_vector(const char* word,
                                        float*      out_vec,
                                        int         dim);

/* ── Main pipeline ───────────────────────────────────────────────────────── */

/* Tokenize text and map every token to its persistent token_id.
 * New words are inserted automatically (same rules as ve_get_token_id).
 *
 * On success *token_ids is a DLL-allocated uint32_t array of *count values.
 * Release with ve_free_ids(*token_ids).                                     */
SK_API int  SK_CALL ve_process_text(const char* text,
                                     uint32_t**  token_ids,
                                     int*        count);

/* Release the array returned by ve_process_text.                            */
SK_API void SK_CALL ve_free_ids(uint32_t* ids);

/* ── Similarity ──────────────────────────────────────────────────────────── */

/* Cosine similarity between the embedding vectors of two tokens.
 * Returns a value in [-1.0, 1.0], or -2.0f on error (check ve_last_error).*/
SK_API float SK_CALL ve_cosine_similarity(uint32_t token_a, uint32_t token_b);

/* ── Pluggable tokenizer ─────────────────────────────────────────────────── */

/* Signature for a custom tokenizer function.
 * Must behave like ve_tokenize: allocate *tokens as a DLL-heap array of
 * *count NUL-terminated strings released by the caller via ve_free_tokens.
 * Returns VE_OK or a negative VE_ERR_* code.                               */
typedef int (*sk_tokenize_fn)(const char *text,
                               char      ***tokens,
                               int         *count,
                               void        *userdata);

/* Install a custom tokenizer.  Pass fn=NULL to restore the built-in one.
 * userdata is forwarded unchanged to fn on every call.
 * Safe to call before ve_init() or between calls — takes effect immediately.*/
SK_API void SK_CALL ve_set_tokenizer(sk_tokenize_fn fn, void *userdata);

/* ── Utilities ───────────────────────────────────────────────────────────── */

/* Current embedding dimension (read-only after ve_init).                    */
SK_API int      SK_CALL sk_get_dim(void);

/* Number of words currently in the vocabulary.                              */
SK_API uint32_t SK_CALL sk_vocab_size(void);

/* ── Per-token domain flags ──────────────────────────────────────────────── */

/* Domain flag bitmask constants — each bit = one domain.
 * A token may belong to multiple domains: chemistry|physics = 0x0005.
 * These are compile-time defaults; runtime values come from [vocab_flags]
 * in the INI.  Use ve_flag_id() to resolve a name to its bitmask value.
 * Add new domains in the INI as the next unused power of two.               */
#define SK_FLAG_GENERAL    0x0000u   /* no domain (unclassified)             */
#define SK_FLAG_CHEMISTRY  0x0001u   /* bit 0                                */
#define SK_FLAG_LEGAL      0x0002u   /* bit 1                                */
#define SK_FLAG_PHYSICS    0x0004u   /* bit 2                                */
/* next free: 0x0008 (bit 3) */

/* Test whether a combined flag value includes a specific domain bit.        */
#define SK_FLAG_HAS(combined, flag)  (((combined) & (flag)) != 0)

/* Resolve a domain name → bitmask value as loaded from the INI.
 * Returns the uint16 bitmask, or -1 if the name is not registered.
 * Example: ve_flag_id("chemistry") → 0x0001                                */
SK_API int SK_CALL ve_flag_id(const char *name);

/* Set multiple domain flags on a token (OR'd into one uint16 bitmask).
 * Example: uint16_t f[] = {SK_FLAG_CHEMISTRY, SK_FLAG_PHYSICS};
 *          ve_set_word_flags(id, f, 2);  → stores 0x0005                   */
SK_API int SK_CALL ve_set_word_flags(uint32_t token_id, const uint16_t *flags, int count);

/* Expand the combined bitmask back into a list of domain flag values.
 * buf receives up to max flag values; *out_count is set to actual count.    */
SK_API int SK_CALL ve_get_word_flags(uint32_t token_id, uint16_t *buf, int max, int *out_count);

/* Returns 1 if the token has the given domain flag set, 0 if not.          */
SK_API int SK_CALL ve_word_has_flag(uint32_t token_id, uint16_t flag);

/* Set the raw combined bitmask directly.
 * Returns VE_OK or VE_ERR_NOTFOUND / VE_ERR_STATE.                         */
SK_API int SK_CALL ve_set_word_flag(uint32_t token_id, uint16_t flag);

/* Get the raw combined bitmask for a token.                                 */
SK_API int SK_CALL ve_get_word_flag(uint32_t token_id, uint16_t *out_flag);

/* ── Logging ─────────────────────────────────────────────────────────────── */

#define SK_LOG_ERROR  0   /* unrecoverable failure                           */
#define SK_LOG_WARN   1   /* degraded behaviour, operation continues         */
#define SK_LOG_INFO   2   /* lifecycle events (init, cleanup, re-init)       */

/* Callback invoked for each log message.
 * level : one of SK_LOG_*.
 * msg   : NUL-terminated string; valid only for the duration of the call.
 * ud    : the userdata pointer passed to sk_set_log_fn.                     */
typedef void (*sk_log_fn)(int level, const char *msg, void *ud);

/* Install or remove a logging callback.  Pass fn=NULL to silence all output
 * (the default).  Not thread-safe — call once before ve_init().            */
SK_API void SK_CALL sk_set_log_fn(sk_log_fn fn, void *ud);

/* ── Error handling ──────────────────────────────────────────────────────── */

/* Last VE_ERR_* code on the calling thread.  Reset at every API entry.     */
SK_API int         SK_CALL ve_last_error(void);

/* Human-readable description of a VE_ERR_* code.                           */
SK_API const char* SK_CALL ve_strerror(int code);

/* ── Synonym lookup ──────────────────────────────────────────────────────── */

/* Register a bidirectional synonym relationship between two tokens.
 * Both token IDs must already exist in the vocabulary.
 * Adding the same pair twice is a no-op.
 * Returns VE_OK, VE_ERR_STATE, VE_ERR_NOTFOUND, or VE_ERR_MEM.            */
SK_API int SK_CALL ve_add_synonym(uint32_t token_a, uint32_t token_b);

/* Retrieve synonyms for token_id.
 * out_ids: caller-allocated uint32_t[max] buffer.
 * *out_count: set to the number of synonyms found (may be 0).
 * Returns VE_OK, VE_ERR_STATE, or VE_ERR_INVAL.                            */
SK_API int SK_CALL ve_synonym_lookup(uint32_t  token_id,
                                      uint32_t *out_ids,
                                      int       max,
                                      int      *out_count);

/* ── Token relationships ─────────────────────────────────────────────────── */

/* Relationship type constants.  A token may have multiple relation types.  */
#define SK_REL_PARENT   1   /* token_a is a broader/parent concept of token_b */
#define SK_REL_CHILD    2   /* token_a is a narrower/child concept of token_b */
#define SK_REL_ANTONYM  3   /* token_a and token_b are opposites              */
#define SK_REL_RELATED  4   /* generic semantic association                   */
#define SK_REL_ANY      0   /* wildcard: match all relation types in queries  */

/* Add a directed relationship: token_a --[rel_type]--> token_b.
 * Both token IDs must already exist in the vocabulary.
 * rel_type must be one of SK_REL_PARENT, SK_REL_CHILD, SK_REL_ANTONYM,
 * or SK_REL_RELATED.
 * Returns VE_OK, VE_ERR_STATE, VE_ERR_NOTFOUND, VE_ERR_INVAL, or VE_ERR_MEM. */
SK_API int SK_CALL ve_add_relation(uint32_t token_a, int rel_type, uint32_t token_b);

/* Retrieve all outgoing relationships of rel_type from token_id.
 * Pass SK_REL_ANY to retrieve relationships of any type.
 * out_ids: caller-allocated uint32_t[max] — receives target token IDs.
 * *out_count: set to the number of results written.
 * Returns VE_OK, VE_ERR_STATE, or VE_ERR_INVAL.                            */
SK_API int SK_CALL ve_get_relations(uint32_t  token_id,
                                     int       rel_type,
                                     uint32_t *out_ids,
                                     int       max,
                                     int      *out_count);

/* ── Runtime vocabulary reload ───────────────────────────────────────────── */

/* Hot-reload the vocabulary from disk without shutting down the engine.
 *
 * Sequence: free in-memory structures → re-read .ovoc (+ synonym and
 * relation tables) → rebuild hash index.
 *
 * Any in-memory additions not yet flushed to disk are discarded.
 * Call ve_cleanup() before ve_reload_vocab() if unsaved state must be
 * preserved (ve_cleanup saves to disk; ve_reload_vocab does not).
 *
 * Use this when an external process (e.g. SORPAYIR) has written a new
 * vocabulary file and you want the running engine to pick it up without
 * overwriting it first.
 *
 * Thread-safe: acquires the write lock for the full duration.
 * Returns VE_OK or a negative VE_ERR_* code.                               */
SK_API int SK_CALL ve_reload_vocab(void);

#ifdef __cplusplus
}
#endif

#endif /* SORKUVAI_DLL_H */
