# Makefile for Porool
# Windows (MinGW/MSYS2) and Linux/macOS
#
# Layout:
#   src/          porool.c  porool_extract.c
#   src/include/  all headers
#   src/tools/    porool_cli.c
#   test/         test_porool.c (and any future *.c test files)
#   build/        all compiled outputs  (gitignored — never committed)
#   docs/         documentation and sample config templates

CC     = gcc
CFLAGS = -std=c99 -Wall -O2 -Isrc/include

# Sorkuvai repo supplies sorkuvai_dll.h and its import library
SORKUVAI_DIR = ../Sorkuvai
SK_INC       = $(SORKUVAI_DIR)/src/include
SK_LIB       = $(SORKUVAI_DIR)/build
SK_BIN       = $(SORKUVAI_DIR)/build

# Tharavu repo supplies tharavu_dll.h (via SK_INC copy) and the tharavu import library
THARAVU_DIR  = ../Tharavu
TH_LIB       = $(THARAVU_DIR)/build

# SLispManager for .ocfg loading
SLM_DIR      = ../sLispManager
SLM_INC      = $(SLM_DIR)/src/include
SLM_LIB      = $(SLM_DIR)/build

CFLAGS += -I$(SK_INC) -I$(THARAVU_DIR)/src/include -I$(SLM_INC)

ifeq ($(OS),Windows_NT)
    DLL         = build/libporool.dll
    IMPLIB      = build/libporool.dll.a
    DLL_LDFLAGS = -shared -Wl,--out-implib,$(IMPLIB)
    EXE         = .exe
    RM          = rm -f
    MKDIR       = mkdir -p
    THREAD_LIBS =
    RPATH       =
else
    DLL         = build/libporool.so
    IMPLIB      =
    DLL_LDFLAGS = -shared -fPIC -fvisibility=hidden
    EXE         =
    RM          = rm -f
    MKDIR       = mkdir -p
    THREAD_LIBS = -lpthread
    RPATH       = -Wl,-rpath,'$$ORIGIN'
endif

OBJ_DIR = build/obj
CLI     = build/porool$(EXE)
LDSK    = -L$(SK_LIB) -L$(TH_LIB) -L$(SLM_LIB) -lsorkuvai_dll -ltharavu_dll -lslispmanager

# ── Core object files ─────────────────────────────────────────────────────────
# porool_extract is compiled twice: once with POROOL_EXPORTS (for the DLL) and
# once without (for the CLI static link path, kept for future static builds).
EXTRACT_OBJ_DLL = $(OBJ_DIR)/porool_extract_dll.o
EXTRACT_OBJ_CLI = $(OBJ_DIR)/porool_extract_cli.o

# OBJS = the set of object files that the shared library bundles
OBJS = $(EXTRACT_OBJ_DLL)

# ── Dynamic test discovery ────────────────────────────────────────────────────
TEST_SRCS = $(wildcard test/*.c)
TEST_BINS = $(patsubst test/%.c,build/%$(EXE),$(TEST_SRCS))

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all test clean runtime-dlls

# ── Default target ────────────────────────────────────────────────────────────
all: $(DLL) $(CLI) runtime-dlls

# ── Runtime DLL staging ───────────────────────────────────────────────────────
# Copy dependency DLLs next to the binaries so they run directly from build/
runtime-dlls:
	-cp -f $(SK_BIN)/sorkuvai.dll build/
	-cp -f $(TH_LIB)/tharavu.dll  build/

# ── Object compilation ────────────────────────────────────────────────────────
$(EXTRACT_OBJ_DLL): src/porool_extract.c src/include/porool_extract.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -DPOROOL_EXPORTS -c -o $@ src/porool_extract.c

$(EXTRACT_OBJ_CLI): src/porool_extract.c src/include/porool_extract.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ src/porool_extract.c

# ── Shared library: porool.dll / libporool.so ─────────────────────────────────
$(DLL): src/porool.c src/include/porool.h src/include/porool_extract.h \
        $(OBJS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -DPOROOL_EXPORTS $(DLL_LDFLAGS) \
	    -o $@ src/porool.c $(OBJS) \
	    $(LDSK) -lz -lm

# ── CLI executable ────────────────────────────────────────────────────────────
$(CLI): src/tools/porool_cli.c src/include/porool.h src/include/tharavu_dll.h \
        $(DLL) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $@ src/tools/porool_cli.c \
	    -Lbuild -lporool -L$(SK_LIB) -L$(TH_LIB) -L$(SLM_LIB) \
	    -lsorkuvai_dll -ltharavu_dll -lslispmanager -lm

# ── Pattern rule: build each test binary from test/*.c ────────────────────────
build/%$(EXE): test/%.c src/include/porool.h $(DLL) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $@ $< \
	    -Lbuild -lporool $(LDSK) $(RPATH) -lm $(THREAD_LIBS)

# ── Test target ───────────────────────────────────────────────────────────────
# Stages tharavu.ini, compiles all test/*.c, runs each binary sequentially,
# tracks failures, exits non-zero if any test fails.
test: $(DLL) runtime-dlls $(TEST_BINS)
	@cp -f docs/sample/tharavu.ini build/ 2>/dev/null || true
	@FAIL=0; \
	for t in $(TEST_BINS); do \
		printf "  Running $$t ... "; \
		(cd build && ./`basename $$t`) \
		    && echo "[PASS]" \
		    || { echo "[FAIL]"; FAIL=1; }; \
	done; \
	if [ $$FAIL -ne 0 ]; then \
		echo ""; echo "--- One or more tests FAILED ---"; exit 1; \
	fi; \
	echo "--- All tests passed ---"

# ── Directory creation ────────────────────────────────────────────────────────
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	-$(RM) -r $(OBJ_DIR)
	-$(RM) $(DLL) $(IMPLIB) $(CLI) $(TEST_BINS)
	-$(RM) build/sorkuvai.dll build/tharavu.dll build/zlib1.dll
	-$(RM) build/libgcc_s_seh-1.dll build/libwinpthread-1.dll
	-$(RM) build/tharavu.ini
