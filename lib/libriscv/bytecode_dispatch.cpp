#include "common.hpp"
#define DISPATCH_MODE_SWITCH_BASED
#define DISPATCH_FUNC simulate_bytecode

#define EXECUTE_INSTR() \
	break;

#include "cpu_dispatch.cpp"

namespace riscv
{
	INSTANTIATE_32_IF_ENABLED(CPU);
	INSTANTIATE_64_IF_ENABLED(CPU);
	INSTANTIATE_128_IF_ENABLED(CPU);
} // riscv
