(module POROOL
  (version "NONE"
    (note "no version constant defined in source or sexp — see known-gaps"))
  (language C)
  (role "RAG retrieval engine")
  (description "embed -> retrieve -> rerank -> build context pipeline; multi-segment support; phrasing cache; query expansion synonyms")
  (process-model
    (porool_t "opaque instance handle; multiple instances may coexist in same process")
    (sorkuvai "process-global singleton shared by all instances; all instances MUST reference same vocab_name"))
  (dependencies
    (sorkuvai "embedding engine; vocab_name from root section of porool.ocfg")
    (tharavu  "chunk/vector tables in .odat/.ovec format"))
  (types
    (porool_t "opaque struct porool_s")
    (porool_config_t
      (segment      "const char*" "\"knowledge\"/\"context\"/\"session\"; NULL=default")
      (top_k        "int"         "chunks to retrieve; default 10")
      (min_score    "float"       "minimum chunk score; default 0.10")
      (max_chars    "int"         "context char budget; default 2000")
      (w_cosine     "float"       "composite weight w1; default 0.60")
      (w_length     "float"       "composite weight w2; default 0.20")
      (w_source     "float"       "composite weight w3; default 0.20")
      (w_def_bonus  "float"       "definitional bonus w4; default 0.15")
      (w_overlap    "float"       "term-overlap bonus w5; default 0.20")
      (note "THARAVU base path and SORKUVAI must be pre-configured by caller before porool_create()"))
    (ChunkMeta
      (concept          "const char*")
      (section          "const char*")
      (type             "const char*")
      (tags             "const char* comma-separated")
      (importance       "const char* 'high'|'medium'|'low'")
      (related_concepts "const char* comma-separated")
      (language         "const char* e.g. 'english'; NULL -> first language in [vocabularies] group"))
    (SearchResult
      (id               "uint32_t row index in .ovec vector store")
      (score            "float composite re-ranked score")
      (text             "char* heap-allocated chunk content")
      (source           "char* source document path")
      (chunk_id         "char* stable chunk identifier")
      (concept          "char* primary concept tag")
      (section          "char* document section")
      (type             "char* chunk type")
      (tags             "char* comma-separated")
      (importance       "char* high|medium|low")
      (related_concepts "char* comma-separated")
      (language         "char* language key chunk was indexed under")
      (note "all char* heap-allocated by POROOL; release entire array with porool_free_results; do not free individually"))))
  (public-api
    (header "src/include/porool.h")
    (functions
      (config
        (porool_config_defaults (cfg:porool_config_t*) -> void
          (note "populate cfg with built-in defaults; call before overriding individual fields")))
      (lifecycle
        (porool_create  (cfg:const-porool_config_t*) -> porool_t*
          (note "cfg may be NULL to use all built-in defaults; THARAVU must already be configured via tde_set_base_path; SORKUVAI must already be initialised via ve_init; SORKUVAI initialized on first call; second call with different vocab_name returns NULL"))
        (porool_destroy (p:porool_t*) -> void
          (note "SORKUVAI shut down when last instance destroyed"))
        (porool_refresh (p:porool_t*) -> int
          (note "re-open default chunk/vector handles; call after external vocab reload or ODAT/OVEC rebuild")))
      (core-pipeline
        (porool_embed_query (p:porool_t* query:const-char* embedding:float** dim:int*) -> int
          (note "mean-pooled token embeddings; heap-allocated; free with porool_free"))
        (porool_retrieve    (p:porool_t* query_vector:float* top_k:int result_count:int*) -> SearchResult*
          (note "retrieves from default chunk/vector tables; heap-allocated; free with porool_free_results"))
        (porool_rerank      (p:porool_t* results:SearchResult* count:int) -> void
          (note "in-place; combines cosine score + chunk-length score + source priority score; sorted descending"))
        (porool_rerank_query (p:porool_t* results:SearchResult* count:int query:const-char*) -> void
          (note "query-aware rerank; adds w4 definitional bonus + w5 term-overlap bonus"))
        (porool_build_context (p:porool_t* results:SearchResult* count:int
                                max_chars:int ctx_min_score:float) -> char*
          (note "deduplicated context string; max_chars limits budget; ctx_min_score=0.0 includes all; heap-allocated; free with porool_free"))
        (porool_query (p:porool_t* query:const-char* top_k:int max_chars:int) -> char*
          (note "all-in-one: embed->retrieve->rerank->build_context; top_k<=0 or max_chars<=0 uses porool.ocfg defaults; free with porool_free")))
      (targeted-retrieval
        (porool_retrieve_from (p:porool_t* query_vector:float* db:const-char*
                                table_name:const-char* top_k:int result_count:int*) -> SearchResult*
          (note "table_name=NULL retrieves from all registered tables in db; results merged and sorted"))
        (porool_query_from    (p:porool_t* query:const-char* db:const-char*
                                table_name:const-char* top_k:int max_chars:int) -> char*
          (note "all-in-one for specific table or all tables when table_name=NULL"))
        (porool_retrieve_target (p:porool_t* query_vector:float* target:const-char*
                                  top_k:int result_count:int*) -> SearchResult*
          (note "target: 'ALL' | 'db' | 'db.table'"))
        (porool_query_target    (p:porool_t* query:const-char* target:const-char*
                                  top_k:int max_chars:int) -> char*
          (note "target: 'ALL'|'db'|'db.table'; free with porool_free")))
      (ingestion
        (porool_ingest          (p:porool_t* file_path:const-char* db:const-char* table:const-char*) -> int
          (supported-types ".txt .pdf .jpg .jpeg .png .docx .xlsx")
          (external-tools "pdftotext tesseract docx2txt xlsx2csv"))
        (porool_ingest_with_meta (p:porool_t* file_path:const-char* db:const-char*
                                   table:const-char* meta:const-ChunkMeta*) -> int
          (returns "0=success; 1=already present (skipped); negative=error")))
      (phrasing
        (porool_phrasing_reload (p:porool_t*) -> void
          (note "reloads from phrasing.query_prefixes and phrasing.chunk_markers ODAT tables"))
        (porool_phrasing_add    (p:porool_t* pattern:const-char* is_prefix:int) -> int
          (note "is_prefix=1 -> query_prefixes; is_prefix=0 -> chunk_markers; returns 0=ok; -1=duplicate; -2=error")))
      (synonyms
        (porool_synonym_add (from:const-char* to:const-char*) -> int
          (note "process-global; linguistic not per-segment; appends 'to' to query before embedding; whole-word case-insensitive; returns 0=ok; -1=already registered; -2=bad args; -3=table full")))
      (utilities
        (porool_dim         (p:porool_t*) -> int
          (note "embedding dimension; SORKUVAI global; same for all instances; 0 if no instance created"))
        (porool_free        (ptr:char*) -> void)
        (porool_free_results (results:SearchResult* count:int) -> void))))
  (capabilities
    (multi-instance   "multiple porool_t instances per process sharing SORKUVAI singleton")
    (multi-segment    "named sub-sections in porool.ocfg: knowledge/context/session/interactions")
    (unified-target   "'ALL'|'db'|'db.table' query routing")
    (reranking        "cosine + chunk-length + source-priority + definitional + term-overlap scoring")
    (ingestion        "file-based ingest with optional ChunkMeta tagging; deduplication by file_path")
    (phrasing-cache   "query prefix and chunk marker tables for semantic phrasings")
    (query-expansion  "process-global synonym expansion before embedding"))
  (known-gaps
    (gap id "PG-01"
      severity HIGH
      description "vocab_first_lang field in PoroolConfig is never populated from sxp parser — read_config_node() has no parser for this key; all ingested chunks get empty language field unless caller provides meta->language explicitly"
      file "src/porool.c:97-100 (struct) :230-280 (reader) :1277 (usage)")
    (gap id "PG-02"
      severity MEDIUM
      description "No version constant defined anywhere (porool.h, porool.c, or this sexp)")
    (gap id "PG-03"
      severity MEDIUM
      description "porool.c uses manual extern declarations for tde/sk functions (lines 48-88) instead of including official headers; tgph_* graph API invisible to Porool internals"
      file "src/porool.c:48-88")
    (gap id "PG-04"
      severity MEDIUM
      description "porool_synonym_add() not thread-safe: reads/writes g_synonyms[] without lock while synonym_expand() reads concurrently"
      file "src/porool.c:1591-1601, porool.c:581-622")
    (gap id "PG-05"
      severity MEDIUM
      description "Legacy .ini sample files (docs/sample/porool.ini, tharavu.ini) are stale; .sxp renamed to .ocfg 2026-05-31; current config format is (porool ...) s-expression in .ocfg")
    (gap id "PG-06"
      severity LOW
      description "Requires host to pre-call tde_set_base_path() before porool_create(); failure causes logical name resolution to use ./data ignoring ocfg data_dir"
      file "src/porool.c:411-472"))
  (directive-divergences
    (divergence
      id "PR-01"
      severity MEDIUM
      description "No version field in sexp or source; added version NONE marker in this update"
      status resolved)
    (divergence
      id "PR-02"
      severity CRITICAL
      description "porool_create signature stale: sexp showed old two-arg form porool_create(config_path, segment_name); actual header v2026-05-30 has porool_create(const porool_config_t *cfg); segment is now a field in porool_config_t; tde_config_load dependency removed"
      file "src/include/porool.h:98"
      actual "POROOL_API porool_t *porool_create(const porool_config_t *cfg);"
      resolution "FIXED in this update: porool_config_t struct added; porool_create updated; porool_config_defaults added"))
  (meta
    (audit-verified "2026-05-31")
    (audit-status   PARTIAL)
    (audit-notes    "PR-02 FIXED: porool_create signature and porool_config_t corrected to DLL-values-only-v2 API; known-gap PG-06 (tde_config_load pre-call) superseded by caller MUST call tde_set_base_path pattern; build artifact libporool.dll present")))
