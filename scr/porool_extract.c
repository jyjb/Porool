/*
 * porool_extract.c — built-in document text extraction
 *
 * Native support (requires only zlib, no external tools):
 *   .txt  — plain text read-through
 *   .pdf  — content stream parser with FlateDecode decompression
 *   .docx — ZIP + word/document.xml  <w:t> element extraction
 *   .xlsx — ZIP + xl/sharedStrings.xml <t> element extraction
 *
 * Image files (.jpg .jpeg .png) are delegated to a registered OCR callback.
 * If none is registered, image files return NULL.
 *
 * Build: link with -lz (zlib — ships with MinGW/MSYS2 and every Linux distro)
 *
 * Pure C99.  No external process spawning.  Apache-2.0 licensed.
 */

#include "../include/porool_extract.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <zlib.h>

#ifdef _WIN32
#  define ICMP(a,b) (_stricmp((a),(b)) == 0)
#else
#  include <strings.h>
#  define ICMP(a,b) (strcasecmp((a),(b)) == 0)
#endif

/* ── OCR hook ────────────────────────────────────────────────────────────── */

static char *(*g_ocr_fn)(const char *path) = NULL;

void porool_register_ocr(char *(*fn)(const char *path)) { g_ocr_fn = fn; }

/* ── File I/O ─────────────────────────────────────────────────────────────── */

#define MAX_DOC_BYTES (32 * 1024 * 1024)

static uint8_t *read_binary(const char *path, size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    rewind(fp);
    if (sz <= 0 || sz > MAX_DOC_BYTES) { fclose(fp); return NULL; }
    uint8_t *buf = (uint8_t *)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[n] = '\0';
    *out_size = n;
    return buf;
}

/* ── Growing text buffer ─────────────────────────────────────────────────── */

typedef struct { char *buf; size_t len; size_t cap; } TB;

static void tb_init(TB *t) { t->buf = NULL; t->len = 0; t->cap = 0; }

static int tb_push(TB *t, const char *s, size_t n)
{
    if (!n) return 1;
    if (t->len + n + 1 > t->cap) {
        size_t nc = t->cap * 2 + n + 4096;
        char *tmp = (char *)realloc(t->buf, nc);
        if (!tmp) return 0;
        t->buf = tmp; t->cap = nc;
    }
    memcpy(t->buf + t->len, s, n);
    t->len += n;
    t->buf[t->len] = '\0';
    return 1;
}

static void tb_putc(TB *t, char c) { tb_push(t, &c, 1); }

static void tb_reset(TB *t) { t->len = 0; if (t->buf) t->buf[0] = '\0'; }

/* ── memmem (not in MSVC / older MinGW) ─────────────────────────────────── */

#ifdef _WIN32
static const void *p_memmem(const void *hay, size_t hlen,
                              const void *ndl, size_t nlen)
{
    if (!nlen) return hay;
    if (hlen < nlen) return NULL;
    const uint8_t *h = (const uint8_t *)hay;
    const uint8_t *n = (const uint8_t *)ndl;
    for (size_t i = 0; i <= hlen - nlen; i++)
        if (memcmp(h + i, n, nlen) == 0) return h + i;
    return NULL;
}
#else
#  define p_memmem memmem
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  ZIP reader — shared by DOCX and XLSX
 *
 *  DOCX and XLSX are both ZIP archives containing XML files.
 *  This minimal reader walks the Central Directory to find an entry by name,
 *  then decompresses it using raw DEFLATE (zlib inflateInit2 with wbits=-15).
 * ══════════════════════════════════════════════════════════════════════════ */

static uint16_t u16le(const uint8_t *p)
{ return (uint16_t)(p[0] | ((uint16_t)p[1] << 8)); }

static uint32_t u32le(const uint8_t *p)
{ return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24); }

#define ZIP_SIG_EOCD  0x06054b50u
#define ZIP_SIG_CD    0x02014b50u
#define ZIP_SIG_LFH   0x04034b50u

/*
 * Find an entry by name inside the ZIP archive [zip, zipsz].
 * Returns a heap-allocated decompressed buffer on success (caller must free),
 * with *out_size set to its length.  Returns NULL on any error.
 */
static uint8_t *zip_extract(const uint8_t *zip, size_t zipsz,
                              const char *name, size_t *out_size)
{
    if (zipsz < 22) return NULL;

    /* Locate End of Central Directory by scanning backwards for its signature */
    int64_t eocd_pos = -1;
    int64_t scan_start = (int64_t)zipsz - 22;
    int64_t scan_end   = scan_start > 65535 ? scan_start - 65535 : 0;
    for (int64_t i = scan_start; i >= scan_end; i--) {
        if (u32le(zip + i) == ZIP_SIG_EOCD) { eocd_pos = i; break; }
    }
    if (eocd_pos < 0) return NULL;

    const uint8_t *eocd   = zip + eocd_pos;
    uint16_t       nrecs  = u16le(eocd + 10);
    uint32_t       cd_off = u32le(eocd + 16);
    if (cd_off >= zipsz) return NULL;

    size_t  namelen = strlen(name);
    const uint8_t *cd = zip + cd_off;

    for (int r = 0; r < nrecs; r++) {
        if ((size_t)(cd - zip) + 46 > zipsz) break;
        if (u32le(cd) != ZIP_SIG_CD) break;

        uint16_t method    = u16le(cd + 10);
        uint32_t comp_sz   = u32le(cd + 20);
        uint32_t uncomp_sz = u32le(cd + 24);
        uint16_t fn_len    = u16le(cd + 28);
        uint16_t ex_len    = u16le(cd + 30);
        uint16_t cm_len    = u16le(cd + 32);
        uint32_t lhdr_off  = u32le(cd + 42);

        if (fn_len == (uint16_t)namelen &&
            memcmp(cd + 46, name, namelen) == 0) {
            /* Found the entry — jump to its Local File Header */
            if ((uint64_t)lhdr_off + 30 > zipsz) return NULL;
            const uint8_t *lhdr = zip + lhdr_off;
            if (u32le(lhdr) != ZIP_SIG_LFH) return NULL;
            uint16_t l_fn  = u16le(lhdr + 26);
            uint16_t l_ex  = u16le(lhdr + 28);
            const uint8_t *data = lhdr + 30 + l_fn + l_ex;
            if ((size_t)(data - zip) + comp_sz > zipsz) return NULL;

            if (method == 0) {
                /* Stored (no compression) */
                uint8_t *out = (uint8_t *)malloc(uncomp_sz + 1);
                if (!out) return NULL;
                memcpy(out, data, uncomp_sz);
                out[uncomp_sz] = '\0';
                *out_size = uncomp_sz;
                return out;
            }
            if (method == 8) {
                /* Deflated — raw DEFLATE (no zlib/gzip header) */
                uint8_t *out = (uint8_t *)malloc(uncomp_sz + 1);
                if (!out) return NULL;
                z_stream zs;
                memset(&zs, 0, sizeof(zs));
                zs.next_in   = (Bytef *)data;
                zs.avail_in  = comp_sz;
                zs.next_out  = out;
                zs.avail_out = uncomp_sz;
                if (inflateInit2(&zs, -15) != Z_OK) { free(out); return NULL; }
                int rc = inflate(&zs, Z_FINISH);
                inflateEnd(&zs);
                if (rc != Z_STREAM_END && rc != Z_OK) { free(out); return NULL; }
                out[zs.total_out] = '\0';
                *out_size = zs.total_out;
                return out;
            }
            return NULL; /* unsupported compression method */
        }
        cd += 46 + fn_len + ex_len + cm_len;
    }
    return NULL;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  XML text extractor
 *
 *  Scans XML for occurrences of <text_tag>…</text_tag> and collects their
 *  content.  Emits a newline at each </para_tag> (pass NULL to skip).
 *  Handles &lt; &gt; &amp; &apos; &quot; entities.
 * ══════════════════════════════════════════════════════════════════════════ */

static void xml_decode_entities(const char *s, size_t n, TB *tb)
{
    for (size_t i = 0; i < n; ) {
        if (s[i] == '&') {
            if (i+4 <= n && memcmp(s+i,"&lt;",4)==0)  { tb_putc(tb,'<'); i+=4; }
            else if (i+4 <= n && memcmp(s+i,"&gt;",4)==0)  { tb_putc(tb,'>'); i+=4; }
            else if (i+5 <= n && memcmp(s+i,"&amp;",5)==0) { tb_putc(tb,'&'); i+=5; }
            else if (i+6 <= n && memcmp(s+i,"&quot;",6)==0){ tb_putc(tb,'"'); i+=6; }
            else if (i+6 <= n && memcmp(s+i,"&apos;",6)==0){ tb_putc(tb,'\'');i+=6; }
            else { tb_putc(tb,s[i]); i++; }
        } else { tb_putc(tb,s[i]); i++; }
    }
}

static char *xml_extract_text(const uint8_t *xml, size_t xsz,
                               const char *text_tag, const char *para_tag)
{
    TB tb; tb_init(&tb);
    const char *p   = (const char *)xml;
    const char *end = p + xsz;

    /* Build close-tag strings once */
    char close_text[80], close_para[80];
    snprintf(close_text, sizeof(close_text), "</%s>", text_tag);
    if (para_tag)
        snprintf(close_para, sizeof(close_para), "</%s>", para_tag);

    size_t tt_len = strlen(text_tag);

    while (p < end) {
        const char *lt = (const char *)memchr(p, '<', (size_t)(end - p));
        if (!lt) break;
        const char *gt = (const char *)memchr(lt+1, '>', (size_t)(end - lt - 1));
        if (!gt) break;

        const char *tag   = lt + 1;
        size_t      tlen  = (size_t)(gt - tag);

        /* Closing paragraph → newline */
        if (para_tag && tlen >= 1 + strlen(para_tag) &&
            tag[0] == '/' && memcmp(tag+1, para_tag, strlen(para_tag)) == 0)
            tb_putc(&tb, '\n');

        /* Opening text tag */
        int is_open = tlen >= tt_len &&
                      memcmp(tag, text_tag, tt_len) == 0 &&
                      (tlen == tt_len || tag[tt_len]==' ' || tag[tt_len]=='/');

        p = gt + 1;

        if (is_open && (tlen == tt_len || tag[tt_len] != '/')) {
            const char *cl = (const char *)p_memmem(p, (size_t)(end - p),
                                                     close_text, strlen(close_text));
            if (cl) {
                xml_decode_entities(p, (size_t)(cl - p), &tb);
                p = cl + strlen(close_text);
            }
        }
    }

    if (!tb.buf) return (char *)calloc(1, 1);
    return tb.buf;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  DOCX  (.docx = ZIP containing word/document.xml)
 * ══════════════════════════════════════════════════════════════════════════ */

static char *extract_docx(const uint8_t *data, size_t size)
{
    size_t   xsz = 0;
    uint8_t *xml = zip_extract(data, size, "word/document.xml", &xsz);
    if (!xml) return NULL;
    char *text = xml_extract_text(xml, xsz, "w:t", "w:p");
    free(xml);
    return text;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  XLSX  (.xlsx = ZIP containing xl/sharedStrings.xml)
 *
 *  sharedStrings.xml holds all unique string values; cells reference them
 *  by index.  Extracting every shared string gives a good corpus for RAG.
 * ══════════════════════════════════════════════════════════════════════════ */

static char *extract_xlsx(const uint8_t *data, size_t size)
{
    size_t   xsz = 0;
    uint8_t *xml = zip_extract(data, size, "xl/sharedStrings.xml", &xsz);
    if (!xml) return NULL;
    char *text = xml_extract_text(xml, xsz, "t", "si");
    free(xml);
    return text;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  PDF content stream text extraction
 *
 *  Algorithm:
 *    1. Locate every "stream … endstream" block in the file.
 *    2. If /FlateDecode is present in the preceding dictionary, inflate
 *       the stream data with zlib (standard zlib header, wbits=15).
 *    3. Parse the (possibly decompressed) content stream for PDF text
 *       operators inside BT…ET blocks:
 *         (str) Tj   — show string
 *         [(…)] TJ   — show glyph array
 *         (str) '    — newline + show string
 *         Td TD T*   — text position moves → emit a space
 *
 *  Handles: FlateDecode (most modern PDFs), uncompressed streams, literal
 *  strings (…), hex strings <…>, basic escape sequences.
 *  Does NOT handle: encrypted PDFs, CIDFont ToUnicode maps, Type3 fonts.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Inflate a zlib-wrapped stream (PDF FlateDecode includes the 2-byte header) */
static uint8_t *pdf_inflate(const uint8_t *data, size_t comp_sz, size_t *out_sz)
{
    size_t cap = comp_sz * 4 + 4096;
    uint8_t *out = (uint8_t *)malloc(cap);
    if (!out) return NULL;

    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    zs.next_in  = (Bytef *)data;
    zs.avail_in = (uInt)comp_sz;

    /* wbits=15 → standard zlib format (PDF FlateDecode) */
    if (inflateInit2(&zs, 15) != Z_OK) { free(out); return NULL; }

    int rc;
    do {
        if (zs.total_out >= cap - 1) {
            cap *= 2;
            uint8_t *tmp = (uint8_t *)realloc(out, cap);
            if (!tmp) { inflateEnd(&zs); free(out); return NULL; }
            out = tmp;
        }
        zs.next_out  = out + zs.total_out;
        zs.avail_out = (uInt)(cap - zs.total_out - 1);
        rc = inflate(&zs, Z_NO_FLUSH);
    } while (rc == Z_OK || (rc == Z_BUF_ERROR && zs.avail_out == 0));

    size_t n = zs.total_out;
    inflateEnd(&zs);

    if (rc != Z_STREAM_END && rc != Z_OK && rc != Z_BUF_ERROR) {
        free(out); return NULL;
    }
    out[n] = '\0';
    *out_sz = n;
    return out;
}

/* Parse a literal PDF string (…) starting just after the opening '('.
 * Appends decoded bytes to tb.  Returns pointer to char after closing ')'. */
static const char *pdf_parse_literal(const char *p, const char *end, TB *tb)
{
    int depth = 1;
    while (p < end && depth > 0) {
        if (*p == '\\' && p+1 < end) {
            p++;
            switch (*p) {
                case 'n': tb_putc(tb, '\n'); break;
                case 'r': tb_putc(tb, ' ');  break;
                case 't': tb_putc(tb, '\t'); break;
                default:
                    if (*p >= '0' && *p <= '7') {
                        /* Octal escape (up to 3 digits) */
                        int val = *p - '0';
                        if (p+1 < end && p[1] >= '0' && p[1] <= '7')
                            { val = val*8 + (*++p - '0'); }
                        if (p+1 < end && p[1] >= '0' && p[1] <= '7')
                            { val = val*8 + (*++p - '0'); }
                        if ((unsigned char)val >= 0x20) tb_putc(tb, (char)val);
                    } else if ((unsigned char)*p >= 0x20) {
                        tb_putc(tb, *p);
                    }
                    break;
            }
            p++;
        } else if (*p == '(') { depth++; tb_putc(tb, '('); p++; }
        else if (*p == ')') {
            depth--;
            if (depth > 0) tb_putc(tb, ')');
            p++;
        } else {
            if ((unsigned char)*p >= 0x20) tb_putc(tb, *p);
            p++;
        }
    }
    return p;
}

/* Parse a hex string <AABB…> starting just after '<'.
 * Appends printable decoded bytes to tb.  Returns pointer after '>'. */
static const char *pdf_parse_hex(const char *p, const char *end, TB *tb)
{
    while (p < end && *p != '>') {
        while (p < end && (unsigned char)*p <= ' ') p++;
        if (p+1 < end && isxdigit((unsigned char)p[0]) &&
                          isxdigit((unsigned char)p[1])) {
            char hex[3] = { p[0], p[1], 0 };
            unsigned char c = (unsigned char)strtol(hex, NULL, 16);
            if (c >= 0x20) tb_putc(tb, (char)c);
            p += 2;
        } else if (p < end) p++;
    }
    if (p < end) p++; /* skip '>' */
    return p;
}

static void pdf_parse_stream(const char *s, size_t sz, TB *out)
{
    const char *p   = s;
    const char *end = s + sz;
    int in_text = 0;   /* inside BT…ET block */

    /* pending_str: last string operand seen; cleared after each operator */
    TB pending; tb_init(&pending);

    while (p < end) {
        /* Skip whitespace */
        while (p < end && (unsigned char)*p <= ' ') p++;
        if (p >= end) break;

        /* Comment */
        if (*p == '%') {
            while (p < end && *p != '\n') p++;
            continue;
        }

        /* Literal string */
        if (*p == '(') {
            tb_reset(&pending);
            p = pdf_parse_literal(p + 1, end, &pending);
            continue;
        }

        /* Hex string */
        if (*p == '<' && p+1 < end && p[1] != '<') {
            tb_reset(&pending);
            p = pdf_parse_hex(p + 1, end, &pending);
            continue;
        }

        /* Array [ … ] — collect all strings for TJ */
        if (*p == '[') {
            p++;
            tb_reset(&pending);
            while (p < end && *p != ']') {
                while (p < end && (unsigned char)*p <= ' ') p++;
                if (p >= end || *p == ']') break;
                if (*p == '(') {
                    p = pdf_parse_literal(p + 1, end, &pending);
                } else if (*p == '<' && p+1 < end && p[1] != '<') {
                    p = pdf_parse_hex(p + 1, end, &pending);
                } else {
                    /* number or other token — skip */
                    while (p < end && *p != '(' && *p != '<' &&
                           *p != ']' && (unsigned char)*p > ' ') p++;
                }
            }
            if (p < end) p++; /* skip ']' */
            continue;
        }

        /* Dictionary << … >> — skip entirely */
        if (*p == '<' && p+1 < end && p[1] == '<') {
            int depth = 1; p += 2;
            while (p < end && depth > 0) {
                if (*p == '<' && p+1 < end && p[1] == '<') { depth++; p += 2; }
                else if (*p == '>' && p+1 < end && p[1] == '>') { depth--; p += 2; }
                else p++;
            }
            continue;
        }

        /* Name /foo — skip */
        if (*p == '/') {
            p++;
            while (p < end && (unsigned char)*p > ' ' &&
                   *p != '/' && *p != '[' && *p != '(') p++;
            continue;
        }

        /* Read operator or number token */
        const char *tok = p;
        while (p < end && (unsigned char)*p > ' ' &&
               *p != '(' && *p != ')' && *p != '<' && *p != '>' &&
               *p != '[' && *p != ']' && *p != '{' && *p != '}' &&
               *p != '/') p++;
        size_t tlen = (size_t)(p - tok);
        if (tlen == 0) { p++; continue; }

        /* Classify operator */
        if (tlen == 2 && memcmp(tok, "BT", 2) == 0) {
            in_text = 1;
            tb_reset(&pending);
        } else if (tlen == 2 && memcmp(tok, "ET", 2) == 0) {
            in_text = 0;
            tb_putc(out, '\n');
            tb_reset(&pending);
        } else if (in_text) {
            /* Text-show operators */
            int is_show = (tlen == 2 &&
                           (memcmp(tok,"Tj",2)==0 || memcmp(tok,"TJ",2)==0)) ||
                          (tlen == 1 && (*tok == '\'' || *tok == '"'));

            if (is_show) {
                if (pending.buf && pending.len > 0)
                    tb_push(out, pending.buf, pending.len);
                tb_reset(&pending);
                if (*tok == '\'') tb_putc(out, '\n');
            } else if (tlen == 2 &&
                       (memcmp(tok,"Td",2)==0 || memcmp(tok,"TD",2)==0 ||
                        memcmp(tok,"T*",2)==0 || memcmp(tok,"Tm",2)==0)) {
                /* Position move — separate words */
                if (out->len > 0 &&
                    out->buf[out->len-1] != ' ' &&
                    out->buf[out->len-1] != '\n')
                    tb_putc(out, ' ');
                tb_reset(&pending);
            } else if (!isdigit((unsigned char)*tok) && *tok != '-' && *tok != '.') {
                /* Any other keyword clears the pending operand */
                tb_reset(&pending);
            }
        } else {
            tb_reset(&pending);
        }
    }
    free(pending.buf);
}

static char *extract_pdf(const uint8_t *data, size_t size)
{
    TB out; tb_init(&out);

    const char *s   = (const char *)data;
    const char *end = s + size;
    const char *p   = s;

    static const char SIG_STREAM[]    = "stream";
    static const char SIG_ENDSTREAM[] = "endstream";

    while (p < end) {
        /* Find next "stream" keyword */
        const char *sm = (const char *)p_memmem(p, (size_t)(end - p),
                                                 SIG_STREAM, 6);
        if (!sm) break;

        /* "endstream" also contains "stream" — skip if preceded by 'end' */
        if (sm >= s + 3 && memcmp(sm - 3, "end", 3) == 0) { p = sm + 6; continue; }

        /* Stream keyword must be immediately followed by \r\n, \n, or \r */
        const char *sd = sm + 6;
        if (sd >= end)                    { p = sd; continue; }
        if (*sd == '\r' && sd+1 < end && sd[1] == '\n') sd += 2;
        else if (*sd == '\r' || *sd == '\n') sd++;
        else { p = sm + 6; continue; } /* not a stream keyword */

        /* Check for /FlateDecode in the 512 bytes before "stream" */
        const char *dict_lo = sm > s + 512 ? sm - 512 : s;
        int flat = (p_memmem(dict_lo, (size_t)(sm - dict_lo),
                             "/FlateDecode", 12) != NULL) ||
                   (p_memmem(dict_lo, (size_t)(sm - dict_lo),
                             "/Fl\n", 4) != NULL) ||
                   (p_memmem(dict_lo, (size_t)(sm - dict_lo),
                             "/Fl\r", 4) != NULL) ||
                   (p_memmem(dict_lo, (size_t)(sm - dict_lo),
                             "/Fl ", 4) != NULL);

        /* Find "endstream" */
        const char *es = (const char *)p_memmem(sd, (size_t)(end - sd),
                                                 SIG_ENDSTREAM, 9);
        if (!es) break;
        size_t stream_sz = (size_t)(es - sd);

        if (flat && stream_sz > 0) {
            size_t   dec_sz = 0;
            uint8_t *dec    = pdf_inflate((const uint8_t *)sd, stream_sz, &dec_sz);
            if (dec) {
                pdf_parse_stream((const char *)dec, dec_sz, &out);
                free(dec);
            }
        } else if (!flat) {
            /* Uncompressed: parse as-is (older PDFs) */
            pdf_parse_stream(sd, stream_sz, &out);
        }

        p = es + 9;
    }

    if (!out.buf) return (char *)calloc(1, 1);
    return out.buf;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Plain text
 * ══════════════════════════════════════════════════════════════════════════ */

static char *extract_txt(const char *path)
{
    size_t sz = 0;
    uint8_t *raw = read_binary(path, &sz);
    if (!raw) return NULL;
    /* raw is already NUL-terminated by read_binary */
    return (char *)raw;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Public API
 * ══════════════════════════════════════════════════════════════════════════ */

char *porool_extract(const char *path)
{
    if (!path) return NULL;
    const char *ext = strrchr(path, '.');
    if (!ext) return NULL;

    if (ICMP(ext, ".txt")) return extract_txt(path);

    /* Image files: delegate to registered OCR callback */
    if (ICMP(ext, ".jpg") || ICMP(ext, ".jpeg") || ICMP(ext, ".png"))
        return g_ocr_fn ? g_ocr_fn(path) : NULL;

    /* Binary formats: read the whole file first */
    size_t   sz   = 0;
    uint8_t *data = read_binary(path, &sz);
    if (!data) return NULL;

    char *result = NULL;
    if      (ICMP(ext, ".pdf"))  result = extract_pdf (data, sz);
    else if (ICMP(ext, ".docx")) result = extract_docx(data, sz);
    else if (ICMP(ext, ".xlsx")) result = extract_xlsx(data, sz);

    free(data);
    return result;
}
