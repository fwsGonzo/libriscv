#include <catch2/catch_test_macros.hpp>
#include <libriscv/machine.hpp>
#include <libriscv/dyncall.hpp>
#include <libriscv/rv32i_instr.hpp>
#include <libriscv/decoded_exec_segment.hpp>
#include <array>
using namespace riscv;
extern std::vector<uint8_t> build_and_load(const std::string&, const std::string&, bool);

TEST_CASE("Counted dyncall FP flag encoding", "[Dyncall]") {
	for (unsigned funct3 = 0; funct3 < 8; funct3++)
	for (unsigned rd = 0; rd < 32; rd++)
	for (unsigned rs1 = 0; rs1 < 32; rs1++) {
		const uint32_t word = (4095u << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x5b;
		const bool legacy = funct3 == 0 && rd == 0 && rs1 == 0;
		const bool counted = funct3 == 7 && rd >= 16 && rd <= 24 &&
			((rs1 >= 24 && rs1 <= 26) || (rs1 >= 28 && rs1 <= 30));
		REQUIRE(Dyncall::valid(word) == (legacy || counted));
		if (counted) {
			REQUIRE(Dyncall::inputs(word) == (rd & 15));
			REQUIRE(Dyncall::outputs(word) == (rs1 & 3));
			REQUIRE(Dyncall::floats(word) == bool(rs1 & 4));
			REQUIRE(Dyncall::encode(4095, rd & 15, rs1 & 3, rs1 & 4) == word);
		}
	}
	REQUIRE(Dyncall::floats(Dyncall::opcode));
	REQUIRE(Dyncall::floats(Dyncall::encode(0, 1, 1)));
}

namespace {
struct CallState { unsigned calls = 0, scenario = 0; };
const Instruction<RISCV64> call_handler {
	[](CPU<RISCV64>& cpu, rv32i_instruction i) {
		auto& machine = cpu.machine();
		auto& state = *machine.get_userdata<CallState>();
		state.calls++;
		cpu.reg(REG_ARG0) += 7;
		if (i.Itype.imm == 501) {
			double sum = 0;
			for (unsigned r = REG_FA0; r < REG_FA0 + 8; r++)
				sum += cpu.registers().getfl(r).f64;
			cpu.registers().getfl(REG_FA0).f64 = sum / 16;
			cpu.registers().getfl(REG_FA0 + 1).f64 = sum / 32;
		}
		if (state.scenario == 1) machine.penalize(100000);
		if (state.scenario == 2) machine.stop();
		if (state.scenario == 3) cpu.trigger_exception(ILLEGAL_OPERATION);
	},
	[](char* buffer, size_t len, const CPU<RISCV64>&, rv32i_instruction) {
		return snprintf(buffer, len, "Test dyncall");
	}
};
}

TEST_CASE("Counted integer calls preserve live FP values", "[Dyncall]") {
	const auto previous = CPU<RISCV64>::on_unimplemented_instruction;
	struct Restore {
		decltype(previous) callback;
		~Restore() { CPU<RISCV64>::on_unimplemented_instruction = callback; }
	} restore{previous};
	CPU<RISCV64>::on_unimplemented_instruction = [](rv32i_instruction i) -> const Instruction<RISCV64>& {
		if (Dyncall::valid(i.whole) && (i.Itype.imm == 500 || i.Itype.imm == 501) &&
			(!Dyncall::is_counted(i.whole) || (Dyncall::inputs(i.whole) == 1 && Dyncall::outputs(i.whole) == 1)) &&
			(i.Itype.imm != 501 || Dyncall::floats(i.whole))) return call_handler;
		return CPU<RISCV64>::get_unimplemented_instruction();
	};
	for (const uint32_t word : {Dyncall::encode(500, 1, 1, false),
		Dyncall::encode(500, 1, 1), (500u << 20) | Dyncall::opcode,
		Dyncall::encode(501, 1, 1, false)}) {
		// Dirty every FP argument and a non-argument. Each loop also mixes in
		// an FP-enabled handler, and a branch skips an integer call.
		std::string body = "li t0, 7\n1:\naddi a0, a0, 1\n";
		for (unsigned r = 10; r < 18; r++)
			body += "fadd.d f" + std::to_string(r) + ", f" + std::to_string(r) + ", f31\n";
		body += "fadd.d f2, f2, f31\n.word " + std::to_string(word) + "\n";
		body += ".word " + std::to_string(Dyncall::encode(501, 1, 1)) + "\n";
		body += "andi t1, t0, 1\nbeqz t1, 2f\n.word " + std::to_string(word) + "\n2:\n";
		body += "addi t0, t0, -1\nbnez t0, 1b\n.word 0x7ff00073\n";
		std::string source = "__attribute__((naked)) void _start(void) { __asm__ volatile(\"";
		for (char c : body) source += c == '\n' ? "\\n" : std::string(1, c);
		source += "\"); }";
		const auto binary = build_and_load(source, "-nostdlib -static -march=rv64imafd_zicsr_zifencei -mabi=lp64d -Wl,-e,_start", false);
		for (unsigned scenario = 0; scenario < 4; scenario++) {
			std::array<uint64_t, 32> expected_gpr{};
			std::array<int64_t, 32> expected_fp{};
			uint64_t expected_pc = 0, expected_counter = 0;
			unsigned expected_calls = 0;
			int expected_exception = -1;
			for (bool translated : {false, true}) {
#if !defined(RISCV_ASMJIT) && !defined(RISCV_BINARY_TRANSLATION)
				if (translated) continue;
#endif
				if (!translated && (scenario == 1 || scenario == 2)) continue;
				MachineOptions<RISCV64> options;
				options.memory_max = 8u << 20;
				options.use_shared_execute_segments = false;
#ifdef RISCV_BINARY_TRANSLATION
				options.translate_enabled = translated;
				options.translation_cache = false;
				options.translate_enable_embedded = false;
#endif
#ifdef RISCV_ASMJIT
				options.asmjit_enabled = translated;
#endif
				Machine<RISCV64> machine(binary, options);
#if defined(RISCV_ASMJIT) || defined(RISCV_BINARY_TRANSLATION)
				if (translated) REQUIRE((machine.cpu.current_execute_segment().is_binary_translated() ||
					machine.cpu.current_execute_segment().is_asmjit_translated()));
#endif
				CallState state{0, scenario};
				machine.set_userdata(&state);
				for (unsigned r = 0; r < 32; r++) machine.cpu.registers().getfl(r).f64 = r + 0.25;
				int exception = -1;
				try { machine.simulate(10000); }
				catch (const MachineException& e) { exception = e.type(); }
				CAPTURE(word, scenario, translated, state.calls, exception);
				if (scenario == 1 || scenario == 2) {
					const bool invalid = (word >> 20) == 501;
					REQUIRE(state.calls == (invalid ? 0 : 1));
					REQUIRE(exception == (invalid ? UNIMPLEMENTED_INSTRUCTION : scenario == 1 ? MAX_INSTRUCTIONS_REACHED : -1));
					REQUIRE(machine.cpu.reg(REG_ARG0) == (invalid ? 1 : 8));
					for (unsigned r = 0; r < 32; r++)
						REQUIRE(machine.cpu.registers().getfl(r).f64 == r + 0.25 +
							((r == 2 || (r >= 10 && r < 18)) ? 31.25 : 0.0));
					continue;
				}
				for (unsigned r = 0; r < 32; r++) {
					CAPTURE(r);
					if (!translated) {
						expected_gpr[r] = machine.cpu.reg(r);
						expected_fp[r] = machine.cpu.registers().getfl(r).i64;
					} else {
						REQUIRE(machine.cpu.reg(r) == expected_gpr[r]);
						REQUIRE(machine.cpu.registers().getfl(r).i64 == expected_fp[r]);
					}
				}
				if (!translated) {
					expected_pc = machine.cpu.pc();
					expected_counter = machine.instruction_counter();
					expected_calls = state.calls;
					expected_exception = exception;
				} else {
					if (exception == -1) {
						REQUIRE(machine.cpu.pc() == expected_pc);
						REQUIRE(machine.instruction_counter() == expected_counter);
					} else {
						// Interpreter handlers report the block PC; native calls
						// publish the precise faulting instruction instead.
						REQUIRE(machine.cpu.pc() == machine.address_of("_start") + 44);
					}
					REQUIRE(state.calls == expected_calls);
					REQUIRE(exception == expected_exception);
				}
			}
		}
	}
}
