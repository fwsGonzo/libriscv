#include "types.hpp"
#include "rv32i_instr.hpp"
#include <unordered_set>

namespace riscv
{
	template <int W>
	struct TransInstr;

	template <int W>
	struct TransInfo
	{
		const rv32i_instruction* instr;
		address_type<W> basepc;
		address_type<W> endpc;
		address_type<W> gp;
		int len;
		bool forward_jumps;
		std::unordered_set<address_type<W>> jump_locations;
		// Pointer to all the other blocks (including current)
		std::vector<TransInfo<W>>* blocks = nullptr;

		std::unordered_set<address_type<W>>& global_jump_locations;
	};
}
