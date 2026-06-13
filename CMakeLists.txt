cmake_minimum_required(VERSION 3.16)
project(Porool C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── All outputs land in build/ when invoked as: cmake -B build -S . ──────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ── External Sorkuvai dependency ──────────────────────────────────────────────
set(SORKUVAI_DIR "${CMAKE_SOURCE_DIR}/../Sorkuvai"
    CACHE PATH "Path to the Sorkuvai repository root")
set(SK_INC "${SORKUVAI_DIR}/src/include")
set(SK_LIB "${SORKUVAI_DIR}/build")

# ── sLispManager dependency ───────────────────────────────────────────────────
set(SLM_DIR "${CMAKE_SOURCE_DIR}/../sLispManager"
    CACHE PATH "Path to sLispManager repository root")
set(SLM_INC "${SLM_DIR}/src/include")
set(SLM_LIB "${SLM_DIR}/build/libslispmanager.dll.a")

# ── Include paths ─────────────────────────────────────────────────────────────
include_directories(src/include ${SK_INC} ${SLM_INC})

# ── THARAVU dependency ────────────────────────────────────────────────────────
set(THARAVU_DIR "${CMAKE_SOURCE_DIR}/../Tharavu"
    CACHE PATH "Path to the Tharavu repository root")
set(TH_LIB "${THARAVU_DIR}/build")

# ── Imported libraries ────────────────────────────────────────────────────────
add_library(sorkuvai_dll STATIC IMPORTED)
set_target_properties(sorkuvai_dll PROPERTIES
    IMPORTED_LOCATION "${SK_LIB}/libsorkuvai.dll.a")

add_library(tharavu_dll STATIC IMPORTED)
set_target_properties(tharavu_dll PROPERTIES
    IMPORTED_LOCATION "${TH_LIB}/libtharavu.dll.a")

# ── porool_extract object (DLL-export variant) ────────────────────────────────
add_library(porool_extract_obj OBJECT src/porool_extract.c)
target_compile_definitions(porool_extract_obj PRIVATE POROOL_EXPORTS)

# ── Core shared library: porool.dll / libporool.so ────────────────────────────
add_library(porool SHARED
    src/porool.c
    $<TARGET_OBJECTS:porool_extract_obj>
)
target_compile_definitions(porool PRIVATE POROOL_EXPORTS)
target_link_libraries(porool PRIVATE sorkuvai_dll tharavu_dll z m ${SLM_LIB})
if(UNIX)
    target_link_libraries(porool PRIVATE pthread)
endif()
# Normalize DLL name: strip MinGW's default "lib" prefix -> porool.dll
set_target_properties(porool PROPERTIES PREFIX "")

# ── CLI executable ────────────────────────────────────────────────────────────
add_executable(porool_cli src/tools/porool_cli.c)
set_target_properties(porool_cli PROPERTIES OUTPUT_NAME porool_CLI)
target_include_directories(porool_cli PRIVATE ${SLM_INC})
target_link_libraries(porool_cli PRIVATE porool sorkuvai_dll tharavu_dll ${SLM_LIB} m)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()

file(GLOB TEST_SOURCES "test/*.c")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    # Skip tests whose filename ends in "_win" when building on non-Windows hosts
    if(WIN32 OR NOT test_name MATCHES "_win$")
        add_executable(${test_name} ${test_src})
        set_target_properties(${test_name} PROPERTIES OUTPUT_NAME ${test_name}_CLI)
        target_include_directories(${test_name} PRIVATE src/include ${SK_INC})
        target_link_libraries(${test_name} PRIVATE
            porool sorkuvai_dll tharavu_dll m)
        if(UNIX)
            target_link_libraries(${test_name} PRIVATE pthread)
        endif()

        # Stage tharavu.sxp next to the test binary so relative config paths resolve
        add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/docs/sample/tharavu.sxp"
                "${CMAKE_BINARY_DIR}/tharavu.sxp"
            COMMENT "Staging tharavu.sxp for ${test_name}")

        add_test(
            NAME    ${test_name}
            COMMAND ${test_name}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endforeach()

# ── Sample code in docs/sample/ (any .c files placed there) ──────────────────
file(GLOB SAMPLE_SOURCES "docs/sample/*.c")
foreach(sample_src ${SAMPLE_SOURCES})
    get_filename_component(sample_name ${sample_src} NAME_WE)
    add_executable(sample_${sample_name} ${sample_src})
    set_target_properties(sample_${sample_name} PROPERTIES OUTPUT_NAME sample_${sample_name}_CLI)
    target_include_directories(sample_${sample_name}
        PRIVATE src/include ${SK_INC})
    target_link_libraries(sample_${sample_name} PRIVATE porool m)
endforeach()
