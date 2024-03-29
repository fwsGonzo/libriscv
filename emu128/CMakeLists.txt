cmake_minimum_required(VERSION 3.9)
project(emulator CXX)

option(GPROF        "Enable profiling with gprof" OFF)
option(LINKER_GC    "Enable linker section GC" OFF)
option(LTO          "Enable interprocedural optimizations" ON)
option(NATIVE       "Enable native instructions" OFF)
option(SANITIZE     "Enable sanitizers" OFF)
option(STATIC_BUILD "Build a static executable" OFF)

option(RISCV_32I   "" OFF)
option(RISCV_64I   "" OFF)
option(RISCV_128I  "" ON)
option(RISCV_EXT_C "" OFF)
add_subdirectory(../lib lib)

set(SOURCES
	src/main.cpp
)

if (LTO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT supported OUTPUT error)
	if (supported)
		message(STATUS "IPO / LTO enabled")
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
	else()
		message(STATUS "IPO / LTO not supported: <${error}>")
	endif()
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fuse-ld=lld")
endif()
if (SANITIZE)
	target_compile_options(riscv PUBLIC "-fsanitize=address,undefined")
endif()
if (NATIVE)
	target_compile_options(riscv PUBLIC "-march=native")
endif()
if (GPROF)
	target_compile_options(riscv PUBLIC -pg -g -fno-inline)
endif()
if (LINKER_GC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-gc-sections -Wl,-s")
endif()

function (add_emulator NAME)
	add_executable(${NAME} ${SOURCES})
	target_link_libraries(${NAME} riscv)

	if (LTO)
		set_property(TARGET ${NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
	endif()
	if (SANITIZE)
		target_link_libraries(${NAME} "-fsanitize=address,undefined")
	endif()
	if (GPROF)
		target_link_libraries(${NAME} "-pg")
	endif()
	if (STATIC_BUILD)
		target_link_libraries(${NAME} "-static -static-libgcc")
	endif()

endfunction()

add_emulator(emulator)
