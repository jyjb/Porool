# Makefile for Porool
# Windows (MinGW/MSYS2) and Linux/macOS
#
# Layout:
#   scr/        porool.c
#   include/    porool.h
#   tests/      test_porool.c
#   bin/        porool.dll  test_porool.exe  (sorkuvai.dll  tharavu.dll copied here)
#   lib/        libporool.a

CC     = gcc
CFLAGS = -std=c99 -Wall -O2 -Iinclude

# Sorkuvai repo supplies the tharavu_dll.h header and both import libraries
SORKUVAI_DIR = ../Sorkuvai
SK_INC       = $(SORKUVAI_DIR)/include
SK_LIB       = $(SORKUVAI_DIR)/lib
SK_BIN       = $(SORKUVAI_DIR)/bin

CFLAGS += -I$(SK_INC)

ifeq ($(OS),Windows_NT)
    DLL         = bin/porool.dll
    IMPLIB      = lib/libporool.a
    DLL_LDFLAGS = -shared -Wl,--out-implib,$(IMPLIB)
    EXE         = .exe
    RM          = rm -f
    MKDIR       = mkdir -p
    THREAD_LIBS =
    RPATH       =
else
    DLL         = bin/libporool.so
    IMPLIB      =
    DLL_LDFLAGS = -shared -fPIC -fvisibility=hidden
    EXE         =
    RM          = rm -f
    MKDIR       = mkdir -p
    THREAD_LIBS = -lpthread
    # Executables find DLLs next to themselves at runtime
    RPATH       = -Wl,-rpath,'$$ORIGIN'
endif

TEST    = bin/test_porool$(EXE)
CLI     = bin/porool$(EXE)
LDSK    = -L$(SK_LIB) -lsorkuvai_dll -ltharavu_dll

# ── Targets ──────────────────────────────────────────────────────────────────

all: bin lib $(DLL) $(CLI) runtime-dlls

# Copy runtime DLLs next to the binaries so the exe runs from bin/
runtime-dlls:
	@cp -f $(SK_BIN)/sorkuvai.dll bin/ 2>/dev/null || true
	@cp -f $(SK_BIN)/tharavu.dll  bin/ 2>/dev/null || true
ifeq ($(OS),Windows_NT)
	@cp -f /mingw64/bin/zlib1.dll            bin/ 2>/dev/null || true
	@cp -f /mingw64/bin/libgcc_s_seh-1.dll   bin/ 2>/dev/null || true
	@cp -f /mingw64/bin/libwinpthread-1.dll   bin/ 2>/dev/null || true
endif

# Built-in document extraction (shared by DLL and CLI, requires zlib)
EXTRACT_SRC = scr/porool_extract.c
EXTRACT_OBJ_DLL = bin/porool_extract_dll.o
EXTRACT_OBJ_CLI = bin/porool_extract_cli.o

$(EXTRACT_OBJ_DLL): $(EXTRACT_SRC) include/porool_extract.h | bin
	$(CC) $(CFLAGS) -DPOROOL_EXPORTS -c -o $@ $(EXTRACT_SRC)

$(EXTRACT_OBJ_CLI): $(EXTRACT_SRC) include/porool_extract.h | bin
	$(CC) $(CFLAGS) -c -o $@ $(EXTRACT_SRC)

# Build porool.dll (+ import lib on Windows)
$(DLL): scr/porool.c include/porool.h include/porool_extract.h \
        $(EXTRACT_OBJ_DLL) | bin lib
	$(CC) $(CFLAGS) -DPOROOL_EXPORTS $(DLL_LDFLAGS) \
	    -o $@ scr/porool.c $(EXTRACT_OBJ_DLL) \
	    $(LDSK) -lz -lm

# Build the CLI (thin wrapper — links against porool.dll + tharavu_dll for stats/peek)
$(CLI): scr/porool_cli.c include/porool.h include/tharavu_dll.h $(DLL) | bin
	$(CC) $(CFLAGS) -o $@ scr/porool_cli.c \
	    -Llib -lporool -L$(SK_LIB) -ltharavu_dll -lm

# Build and run the full test suite
test: $(TEST)
	@cp -f $(SK_BIN)/sorkuvai.dll bin/ 2>/dev/null || true
	@cp -f $(SK_BIN)/tharavu.dll  bin/ 2>/dev/null || true
ifeq ($(OS),Windows_NT)
	@cp -f /mingw64/bin/libgcc_s_seh-1.dll   bin/ 2>/dev/null || true
	@cp -f /mingw64/bin/libwinpthread-1.dll   bin/ 2>/dev/null || true
endif
	cd bin && ./test_porool$(EXE)

$(TEST): tests/test_porool.c include/porool.h $(DLL) | bin
	$(CC) $(CFLAGS) -o $@ tests/test_porool.c \
	    -Llib $(LDSK) $(RPATH) -lporool -lm $(THREAD_LIBS)

# Create output directories
bin lib:
	$(MKDIR) $@

clean:
	-$(RM) $(DLL) $(IMPLIB) $(TEST) $(CLI)
	-$(RM) $(EXTRACT_OBJ_DLL) $(EXTRACT_OBJ_CLI)
	-$(RM) bin/sorkuvai.dll bin/tharavu.dll bin/zlib1.dll
	-$(RM) bin/libgcc_s_seh-1.dll bin/libwinpthread-1.dll

.PHONY: all test clean
