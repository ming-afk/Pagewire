# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Trimmed for DiskANN_reproduce: only the control_diskann target used by the
# three PageWire experiments is built here. The upstream apps/CMakeLists.txt
# also builds build_memory_index, search_disk_index, and several per-branch
# duplicate targets (control_diskann_twocenter/_layoutasync/_phase/_layoutstop)
# that are not needed to run these experiments -- all are the exact same
# control_diskann.cpp source built under a different binary name.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Q-learning static-workload rewire on a disk-resident DiskANN index.
# Linux-only: it drives LinuxAlignedFileReader (O_DIRECT + libaio) directly.
# zlib is for the .npz query/GT cache reader in qlrewire_support.
if (UNIX AND NOT APPLE)
    find_package(ZLIB REQUIRED)
    add_executable(control_diskann control_diskann.cpp)
    target_include_directories(control_diskann PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(control_diskann ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} ZLIB::ZLIB)
endif()
