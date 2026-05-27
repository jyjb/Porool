/*
 * porool_extract.h — internal document text extraction API
 *
 * NOT part of the public porool.h API surface.
 * Included by porool.c (DLL) and porool_cli.c (EXE).
 */

#ifndef POROOL_EXTRACT_H
#define POROOL_EXTRACT_H

/*
 * Extract plain text from a document file.
 * Supported natively (no external tools): .txt  .pdf  .docx  .xlsx
 * Images (.jpg .jpeg .png) require a registered OCR callback — see below.
 *
 * Returns a heap-allocated NUL-terminated string, or NULL on failure.
 * Caller must free() the result.
 */
char *porool_extract(const char *path);

/*
 * Register a custom OCR function for image files (.jpg .jpeg .png).
 * fn(path) must return a heap-allocated string, or NULL if it cannot process
 * the file.  Pass NULL to disable image support (default).
 */
void porool_register_ocr(char *(*fn)(const char *path));

#endif /* POROOL_EXTRACT_H */
