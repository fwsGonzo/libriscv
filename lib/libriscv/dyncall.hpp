#pragma once
#include <cstdint>

namespace riscv {
// Custom-2 dynamic-call ABI
//
//   31          20 19         15 14   12 11        7 6    0
//  +--------------+-------------+-------+-----------+------+
//  |    index     | 0b11 |F|out | 0b111 | 0b1 | in  | 0x5B |  counted
//  +--------------+-------------+-------+-----------+------+
//  |    index     |    x0       | 0b000 |    x0     | 0x5B |  legacy
//  +--------------+-------------+-------+-----------+------+
//
// Legacy is the encoding documented for `.insn i 0b1011011, 0, x0, x0, N`:
// it carries no counts and means 8 inputs and 2 outputs.
struct Dyncall {
	static constexpr uint32_t opcode = 0x5b;
	/// @brief funct3 of the counted form. The legacy form uses 0.
	static constexpr unsigned counted_funct3 = 0b111;
	/// @brief rd  = rd_tag << 4 | inputs, so bit 11 separates it from a small rd.
	static constexpr unsigned rd_tag  = 0b1;
	/// @brief Bits 19:18 tag rs1; bit 17 enables floating-point synchronization.
	static constexpr unsigned rs1_tag = 0b111;
	static constexpr uint32_t float_mask = 1u << 17;
	static constexpr unsigned max_inputs  = 8;
	static constexpr unsigned max_outputs = 2;

	/// @brief True for the tagged form
	static constexpr bool is_counted(uint32_t word) noexcept {
		return ((word >> 12) & 7) == counted_funct3;
	}
	static constexpr bool valid(uint32_t word) noexcept {
		if ((word & 127) != opcode)
			return false;
		const unsigned rd  = (word >> 7) & 31;
		const unsigned rs1 = (word >> 15) & 31;
		if (is_counted(word))
			return (rd >> 4) == rd_tag   && (rd  & 15) <= max_inputs
				&& ((rs1 | 4) >> 2) == rs1_tag && (rs1 & 3) <= max_outputs;
		// Legacy (no counts)
		return ((word >> 12) & 7) == 0 && rd == 0 && rs1 == 0;
	}
	/// @brief Integer inputs a0..a(N-1). Only meaningful for a valid() word.
	static constexpr unsigned inputs(uint32_t word) noexcept {
		return is_counted(word) ? ((word >> 7) & 15) : max_inputs;
	}
	/// @brief Integer outputs a0..a(N-1). Only meaningful for a valid() word.
	static constexpr unsigned outputs(uint32_t word) noexcept {
		return is_counted(word) ? ((word >> 15) & 3) : max_outputs;
	}
	/// @brief Synchronize fa0-fa7 inputs and fa0-fa1 outputs. Clearing the bit
	/// promises no guest FP register access by the handler. The embedder must
	/// validate that promise; legacy and existing counted words retain FP.
	static constexpr bool floats(uint32_t word) noexcept {
		return !is_counted(word) || (word & float_mask) != 0;
	}
	static constexpr uint32_t arg_mask(unsigned count) noexcept {
		return ((1u << count) - 1) << 10;
	}
	// Caller: validate index < 4096, inputs <= 8 and outputs <= 2.
	static constexpr uint32_t encode(unsigned index, unsigned inputs, unsigned outputs,
		bool floats = true) noexcept {
		return (index << 20)
			| ((((rs1_tag << 2) & ~4u) | (floats ? 4u : 0u) | outputs) << 15)
			| (counted_funct3 << 12)
			| (((rd_tag << 4) | inputs) << 7)
			| opcode;
	}
};
static_assert(Dyncall::valid(Dyncall::encode(4095, 8, 2)), "The counted form round-trips");
static_assert(Dyncall::inputs(Dyncall::encode(0, 5, 1)) == 5, "rd has the input count");
static_assert(Dyncall::outputs(Dyncall::encode(0, 5, 1)) == 1, "rs1 has the output count");
static_assert(Dyncall::valid(Dyncall::opcode), "The legacy form needs no counts");
static_assert(Dyncall::floats(Dyncall::opcode) && Dyncall::floats(Dyncall::encode(0, 0, 0)),
	"Existing callers retain floating-point synchronization");
static_assert(Dyncall::valid(Dyncall::encode(4095, 8, 2, false)) &&
	!Dyncall::floats(Dyncall::encode(4095, 8, 2, false)), "Integer-only counted form");
static_assert(Dyncall::inputs(Dyncall::opcode) == 8 && Dyncall::outputs(Dyncall::opcode) == 2,
	"The legacy form is the full argument range");
// A custom instruction that merely reuses custom-2 must not read as a dyncall.
static_assert(!Dyncall::valid(0b1000011011011), "funct3=1 is not the counted form");
static_assert(!Dyncall::valid(0b0000111011011), "a non-zero rd is not the legacy form");
}
