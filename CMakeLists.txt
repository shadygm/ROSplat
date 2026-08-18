cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(rosplat_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SS_BACKEND vulkan CACHE STRING "Spirula compute backend" FORCE)
set(SS_BUILD_GUI OFF CACHE BOOL "ROSplat owns the ImGui application" FORCE)
set(SS_BUILD_SFM OFF CACHE BOOL "ROSplat only needs Gaussian rendering" FORCE)
set(SS_BUILD_SAM OFF CACHE BOOL "ROSplat only needs Gaussian rendering" FORCE)
set(SS_BUILD_BACKEND_TESTS OFF CACHE BOOL "Do not build upstream backend tests" FORCE)
set(SS_SEPARATE_TOOLS OFF CACHE BOOL "Do not build Spirula tools" FORCE)
set(SS_CHECK_COMMENTS OFF CACHE BOOL "Do not lint external source from ROSplat" FORCE)

add_subdirectory(external/spirula-studio EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)

add_library(rosplat_spirula SHARED
    native/spirula_bridge.cpp
)
target_include_directories(rosplat_spirula
    PUBLIC native
    PRIVATE external/spirula-studio/src
)
target_link_libraries(rosplat_spirula
    PRIVATE csrc_portable ss_backend_vulkan Threads::Threads
)
set_target_properties(rosplat_spirula PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

install(TARGETS rosplat_spirula
    LIBRARY DESTINATION lib/rosplat
    RUNTIME DESTINATION bin
)
