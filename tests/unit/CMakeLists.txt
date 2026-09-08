cmake_minimum_required(VERSION 3.10)
project(libriscv_unittests CXX)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O1 -ggdb3 -fno-omit-frame-pointer -fsanitize=address,undefined")


option(RISCV_MULTIPROCESS "" ON)
add_subdirectory(../../lib lib)
target_compile_definitions(riscv PUBLIC
	FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION=1
	RISCV_SYSCALLS_MAX=600
	RISCV_ARENA_DEBUG=1
)

add_subdirectory(../Catch2 Catch2)
target_compile_features(Catch2 PUBLIC cxx_std_17)

enable_testing()

function(add_unit_test NAME)
	add_executable(${NAME}
		${ARGN}
		codebuilder.cpp
	)
	target_compile_definitions(${NAME} PUBLIC SRCDIR="${CMAKE_CURRENT_SOURCE_DIR}")
	target_link_libraries(${NAME} riscv Catch2WithMain)
	add_test(
		NAME test_${NAME}
		COMMAND ${NAME}
	)
endfunction()

add_unit_test(basic    basic.cpp)
if (RISCV_BINARY_TRANSLATION OR RISCV_ASMJIT)
add_unit_test(bgcompile bgcompile.cpp) # background JIT vs. execute segment lifetime
endif()
if (RISCV_BINARY_TRANSLATION AND UNIX)
add_unit_test(bintr_source binary_translation_source.cpp)
endif()
add_unit_test(basic32  basic32.cpp)
if (RISCV_VIRTUAL_PAGING)
add_unit_test(brutal   brutal.cpp)  # uses non-arena machines (paging)
endif()
add_unit_test(checksum checksum.cpp)
add_unit_test(crc32c   crc32c.cpp)
add_unit_test(custom   custom.cpp)
add_unit_test(dynamic  dynamic.cpp)
add_unit_test(dyncall  dyncall.cpp)
add_unit_test(examples examples.cpp)
add_unit_test(exceptions exceptions.cpp)
add_unit_test(heap     heaptest.cpp)
add_unit_test(heap_invariants heap_invariants.cpp)
add_unit_test(libcfast libc_fastpath.cpp)
add_unit_test(fptest   fp_testsuite.cpp)
add_unit_test(micro    micro.cpp)
if (RISCV_VIRTUAL_PAGING)
add_unit_test(memtrap  memory_trap.cpp)
endif()
add_unit_test(native   native.cpp)
add_unit_test(native_rust native_rust.cpp rustbuilder.cpp)
add_unit_test(pcrel    pcrel.cpp)
add_unit_test(png      png.cpp)
if (RISCV_VIRTUAL_PAGING)
add_unit_test(protect  protections.cpp)
endif()
add_unit_test(rvbuffer rvbuffer.cpp)
if (RISCV_BINARY_TRANSLATION OR RISCV_ASMJIT)
add_unit_test(unchecked unchecked.cpp)  # translate_unsafe_remove_checks, both back-ends
add_unit_test(nbitarena nbit_arena.cpp)
endif()
if (RISCV_EXT_V)
add_unit_test(rvv     rvv.cpp)
endif()
add_unit_test(rva23   rva23.cpp)
add_unit_test(zcb     zcb.cpp)
if (RISCV_VIRTUAL_PAGING)
add_unit_test(serialize serialize.cpp) # serializes the page table (non-arena machines)
endif()
add_unit_test(vmcall   vmcall.cpp)
if (RISCV_VIRTUAL_PAGING)
add_unit_test(va_exec  va_execute.cpp) # execute permissions need page attributes
endif()
add_unit_test(elftest  verify_elf.cpp)
add_unit_test(security security.cpp)
if (NOT RISCV_VIRTUAL_PAGING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
add_unit_test(shared_rodata shared_rodata.cpp) # linear arena, memfd sealing
endif()
