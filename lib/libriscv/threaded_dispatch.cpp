#include "common.hpp"
#define DISPATCH_MODE_THREADED
#define DISPATCH_FUNC simulate_threaded

#define EXECUTE_INSTR()      \
	if constexpr (FUZZING) { \
		if (UNLIKELY(decoder->get_bytecode() >= BYTECODES_MAX)) \
			abort();         \
	}                        \
	goto *computed_opcode[decoder->get_bytecode()];

#include "cpu_dispatch.cpp"

namespace riscv
{
	INSTANTIATE_32_IF_ENABLED(CPU);
	INSTANTIATE_64_IF_ENABLED(CPU);
	INSTANTIATE_128_IF_ENABLED(CPU);
} // riscv
