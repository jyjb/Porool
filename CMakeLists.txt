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

# ── Include paths ─────────────────────────────────────────────────────────────
include_directories(src/include ${SK_INC})

# ── Imported Sorkuvai libraries ───────────────────────────────────────────────
add_library(sorkuvai_dll STATIC IMPORTED)
set_target_properties(sorkuvai_dll PROPERTIES
    IMPORTED_LOCATION "${SK_LIB}/libsorkuvai_dll.a")

add_library(tharavu_dll STATIC IMPORTED)
set_target_properties(tharavu_dll PROPERTIES
    IMPORTED_LOCATION "${SK_LIB}/libtharavu_dll.a")

# ── porool_extract object (DLL-export variant) ────────────────────────────────
add_library(porool_extract_obj OBJECT src/porool_extract.c)
target_compile_definitions(porool_extract_obj PRIVATE POROOL_EXPORTS)

# ── Core shared library: porool.dll / libporool.so ────────────────────────────
add_library(porool SHARED
    src/porool.c
    $<TARGET_OBJECTS:porool_extract_obj>
)
target_compile_definitions(porool PRIVATE POROOL_EXPORTS)
target_link_libraries(porool PRIVATE sorkuvai_dll tharavu_dll z m)
if(UNIX)
    target_link_libraries(porool PRIVATE pthread)
endif()

# ── CLI executable ────────────────────────────────────────────────────────────
add_executable(porool_cli src/tools/porool_cli.c)
set_target_properties(porool_cli PROPERTIES OUTPUT_NAME porool)
target_link_libraries(porool_cli PRIVATE porool tharavu_dll m)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()

file(GLOB TEST_SOURCES "test/*.c")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    # Skip tests whose filename ends in "_win" when building on non-Windows hosts
    if(WIN32 OR NOT test_name MATCHES "_win$")
        add_executable(${test_name} ${test_src})
        target_include_directories(${test_name} PRIVATE src/include ${SK_INC})
        target_link_libraries(${test_name} PRIVATE
            porool sorkuvai_dll tharavu_dll m)
        if(UNIX)
            target_link_libraries(${test_name} PRIVATE pthread)
        endif()

        # Stage tharavu.ini next to the test binary so relative ini paths resolve
        add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/docs/sample/tharavu.ini"
                "${CMAKE_BINARY_DIR}/tharavu.ini"
            COMMENT "Staging tharavu.ini for ${test_name}")

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
    target_include_directories(sample_${sample_name}
        PRIVATE src/include ${SK_INC})
    target_link_libraries(sample_${sample_name} PRIVATE porool m)
endforeach()
