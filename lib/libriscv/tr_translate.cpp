#include <cmath>
#include <dlfcn.h>
#include <unistd.h>
#include "machine.hpp"
#include "instruction_list.hpp"
#include "rv32i_instr.hpp"
#include "tr_api.hpp"
#include "util/crc32.hpp"
#include <stdexcept>
#include <unordered_set>
//#define BINTR_TIMING

namespace riscv
{
	static constexpr int  LOOP_OFFSET_MAX = 160;
	static constexpr bool SCAN_FOR_GP = true;

	inline timespec time_now();
	inline long nanodiff(timespec, timespec);
#ifdef BINTR_TIMING
	#define TIME_POINT(x) \
		asm("" : : : "memory"); \
		auto x = time_now();    \
		asm("" : : : "memory");
#else
	#define TIME_POINT(x)  /* */
#endif

static std::unordered_set<uint32_t> good_insn
{
	RV32I_LOAD,
	RV32I_STORE,
	RV32I_BRANCH,
	RV32I_JAL,
	RV32I_OP_IMM,
	RV32I_OP,
	RV32I_LUI,
	RV32I_AUIPC,
	RV32I_SYSTEM,
	RV32I_FENCE,
	RV64I_OP_IMM32,
	RV64I_OP32,
	RV32F_LOAD,
	RV32F_STORE,
	RV32F_FMADD,
	RV32F_FMSUB,
	RV32F_FNMADD,
	RV32F_FNMSUB,
	RV32F_FPFUNC,
};

template <int W>
inline uint32_t opcode(const typename CPU<W>::instr_pair& ip) {
	return ip.second.opcode();
}
template <int W>
inline bool gucci(const typename CPU<W>::instr_pair& ip) {
	return good_insn.count(opcode<W>(ip)) > 0;
}

template <int W>
struct NamedIPair {
	using instr_pair = typename CPU<W>::instr_pair;
	instr_pair& ipair;
	std::string symbol;
};

template <int W>
void CPU<W>::try_translate(const MachineOptions<W>& options,
	address_t basepc, std::vector<instr_pair>& ipairs) const
{
	// Disable translator with NO_TRANSLATE=1
	// or by setting max blocks to zero.
	if (getenv("NO_TRANSLATE") || 0 == options.translate_blocks_max) {
		machine().memory.set_binary_translated(nullptr);
		return;
	}
	// Run with VERBOSE=1 to see command and output
	const bool verbose = (getenv("VERBOSE") != nullptr);
	address_t gp = 0;
	TIME_POINT(t0);
if constexpr (SCAN_FOR_GP) {
	// We assume that GP is initialized with AUIPC,
	// followed by OP_IMM (and maybe OP_IMM32)
	for (auto it = ipairs.begin(); it != ipairs.end(); ++it)
	if (it->second.opcode() == RV32I_AUIPC) {
		const auto auipc = it->second;
		if (auipc.Utype.rd == 3) { // GP
			// calculate current PC for AUIPC
			const address_t pc = basepc + 4 * (it - ipairs.begin());
			const auto addi = (it+1)->second;
			if (addi.opcode() == RV32I_OP_IMM && addi.Itype.funct3 == 0x0) {
				//printf("Found OP_IMM: ADDI  rd=%d, rs1=%d\n", addi.Itype.rd, addi.Itype.rs1);
				if (addi.Itype.rd == 3 && addi.Itype.rs1 == 3) { // GP
					gp = pc + auipc.Utype.upper_imm() + addi.Itype.signed_imm();
					break;
				}
			}
		}
	}
#ifdef BINTR_TIMING
	TIME_POINT(t1);
	printf(">> GP scan took %ld ns\n", nanodiff(t0, t1));
#endif
} // SCAN_FOR_GP

	// Code block and loop detection
	TIME_POINT(t2);
	size_t icounter = 0;
	auto it = ipairs.begin();
	std::vector<std::pair<decltype(it), address_t>> loops;
	std::unordered_set<address_t> already_generated;
	std::unordered_set<address_t> already_looped;
	struct CodeBlock {
		instr_pair& instr;
		size_t      length;
		address_t   addr;
		bool        has_branch;
	};
	std::vector<CodeBlock> blocks;

	while (it != ipairs.end() && icounter < options.translate_instr_max)
	{
		if (!loops.empty()) {
			it = loops.back().first;
			basepc = loops.back().second;
			loops.pop_back();
		}
		if (gucci<W>(*it))
		{
			auto block = it;
			bool has_branch = false;
			bool has_loop = false;
			// measure block length
			while (++it != ipairs.end()) {
				// we can include this but not continue after
				if (it->second.opcode() == RV32I_JALR ||
					(it->second.opcode() == RV32I_SYSTEM && it->second.Itype.funct3 == 0x0 && it->second.Itype.imm != 0))
				{
					++it; break;
				}
if constexpr (LOOP_OFFSET_MAX > 0) {
				// loop detection (negative branch offsets)
				if (it->second.opcode() == RV32I_BRANCH && it->second.Btype.sign()) {
					has_branch = true;
					// detect jump location
					const auto& instr = it->second;
					const size_t length = it - block;
					const auto offset = instr.Btype.signed_imm();
					const auto dst = basepc + (4 * length) + offset;
					if (offset > -LOOP_OFFSET_MAX && already_looped.count(dst) == 0) {
						loops.push_back({it + offset / 4, dst});
						has_loop = true;
						already_looped.insert(dst);
					}
				}
}
				// we can accelerate these and continue
				if (gucci<W>(*it) == false) {
					// must exit native, ending block
					break;
				}
			}
			const size_t length = it - block;
			if (length >= options.block_size_treshold
				&& icounter + length < options.translate_instr_max
				&& already_generated.count(basepc) == 0)
			{
				already_generated.insert(basepc);
				//printf("Block found at %#lX. Length: %zu\n", (long) basepc, length);
				blocks.push_back({*block, length, basepc, has_branch});
				icounter += length;
				// we can't translate beyond this estimate, otherwise
				// the compiler will never finish code generation
				if (blocks.size() >= options.translate_blocks_max)
					break;
			}
			basepc += 4 * length;
		}
		else {
			basepc += 4;
			++it;
		}
	}
#ifdef BINTR_TIMING
	TIME_POINT(t3);
	printf(">> Code block detection %ld ns\n", nanodiff(t2, t3));
#endif

	// Code generation
	std::vector<NamedIPair<W>> dlmappings;
	extern std::string bintr_code;
	std::string code = bintr_code;

	for (const auto& block : blocks)
	{
		std::string func =
			"f" + std::to_string(block.addr);
		emit(code, func, &block.instr, {
			block.addr, gp, block.length,
			block.has_branch,
			options.forward_jumps
		});
		dlmappings.push_back({block.instr, std::move(func)});
	}
#ifdef BINTR_TIMING
	TIME_POINT(t4);
	printf(">> Code generation took %ld ns\n", nanodiff(t3, t4));
#endif

	if (verbose) {
		printf("Emitted %zu accelerated instructions and %zu functions. GP=0x%lX  FWJ=%d\n",
			icounter, dlmappings.size(), (long) gp, options.forward_jumps);
	}
	// nothing to compile without mappings
	if (dlmappings.empty()) {
		return;
	}
	if (machine().memory.is_binary_translated()) {
		throw std::runtime_error("Machine already reports binary translation");
	}
	TIME_POINT(t5);
	const size_t code_size = code.size();
	// add compiler arguments to make it part of checksum
	extern std::string compile_command(int arch);
	code.append(compile_command(W));
	// create cacheable filename
	const uint32_t checksum = crc32c(code.c_str(), code.size());
	char filename[256];
	int len = snprintf(filename, sizeof(filename),
		"/tmp/rvbintr-%08X", checksum);
	if (len <= 0) {
		return;
	}
	void* dylib = nullptr;
	code.resize(code_size); // remove compiler arguments
#ifdef BINTR_TIMING
	TIME_POINT(t6);
	printf(">> Code hashing took %ld ns\n", nanodiff(t5, t6));
#endif

#ifdef RISCV_TRANSLATION_CACHE
	if (access(filename, R_OK) == 0) {
		TIME_POINT(t7);
		dylib = dlopen(filename, RTLD_LAZY);
	#ifdef BINTR_TIMING
		TIME_POINT(t8);
		printf(">> dlopen took %ld ns\n", nanodiff(t7, t8));
	#endif
	}
#endif
	// compile ourselves
	if (dylib == nullptr) {
		TIME_POINT(t9);
		extern void* compile(const std::string& code, int arch, const char*);
		dylib = compile(code, W, filename);
	#ifdef BINTR_TIMING
		TIME_POINT(t10);
		printf(">> Code compilation took %.2f ms\n", nanodiff(t9, t10) / 1e6);
	#endif
		// check compilation result
		if (dylib == nullptr) {
			return;
		}
	}

	TIME_POINT(t11);
	// map the API callback table
	auto* ptr = dlsym(dylib, "init");
	if (ptr == nullptr) {
		fprintf(stderr, "libriscv: Could not find dylib init function\n");
		dlclose(dylib);
		return;
	}
	auto func = (void (*)(const CallbackTable<W>&)) ptr;
	func(CallbackTable<W>{
		.mem_read8 = [] (CPU<W>& cpu, address_type<W> addr) -> uint8_t {
			return cpu.machine().memory.template read<uint8_t> (addr);
		},
		.mem_read16 = [] (CPU<W>& cpu, address_type<W> addr) -> uint16_t {
			return cpu.machine().memory.template read<uint16_t> (addr);
		},
		.mem_read32 = [] (CPU<W>& cpu, address_type<W> addr) -> uint32_t {
			return cpu.machine().memory.template read<uint32_t> (addr);
		},
		.mem_read64 = [] (CPU<W>& cpu, address_type<W> addr) -> uint64_t {
			return cpu.machine().memory.template read<uint64_t> (addr);
		},
		.mem_write8 = [] (CPU<W>& cpu, address_type<W> addr, uint8_t val) {
			cpu.machine().memory.template write<uint8_t> (addr, val);
		},
		.mem_write16 = [] (CPU<W>& cpu, address_type<W> addr, uint16_t val) {
			cpu.machine().memory.template write<uint16_t> (addr, val);
		},
		.mem_write32 = [] (CPU<W>& cpu, address_type<W> addr, uint32_t val) {
			cpu.machine().memory.template write<uint32_t> (addr, val);
		},
		.mem_write64 = [] (CPU<W>& cpu, address_type<W> addr, uint64_t val) {
			cpu.machine().memory.template write<uint64_t> (addr, val);
		},
		.mem_getrptr = [] (CPU<W>& cpu, address_type<W> addr) -> void* {
			auto& page = cpu.machine().memory.get_readable_page(addr);
			return &page.data() + (addr & (Page::size()-1));
		},
		.jump = [] (CPU<W>& cpu, address_type<W> addr, uint64_t val) {
			cpu.jump(addr);
			cpu.machine().increment_counter(val);
		},
		.syscall = [] (CPU<W>& cpu, address_type<W> n, uint64_t val) -> int {
			auto old_pc = cpu.pc();
			cpu.registers().pc += val * 4;
			cpu.machine().system_call(n);
			// if the system did not modify PC, return to bintr
			if (cpu.pc() - val * 4 == old_pc && !cpu.machine().stopped()) {
				cpu.registers().pc = old_pc;
				return 0;
			}
			// otherwise, update instruction counter and exit
			cpu.machine().increment_counter(val);
			return 1;
		},
		.stop = [] (CPU<W>& cpu, uint64_t val) {
			cpu.registers().pc += val * 4;
			cpu.machine().increment_counter(val);
			cpu.machine().stop();
		},
		.ebreak = [] (CPU<W>& cpu, uint64_t val) {
			cpu.registers().pc += val * 4;
			cpu.machine().increment_counter(val);
			cpu.machine().ebreak();
		},
		.system = [] (CPU<W>& cpu, uint32_t instr) {
			cpu.machine().system(rv32i_instruction{instr});
		},
		.trigger_exception = [] (CPU<W>& cpu, int e) {
			cpu.trigger_exception(e);
		},
		.sqrtf32 = [] (float f) -> float {
			return std::sqrt(f);
		},
		.sqrtf64 = [] (double d) -> double {
			return std::sqrt(d);
		},
	});

	// map all the functions
	for (auto& mapping : dlmappings) {
		auto* func = dlsym(dylib, mapping.symbol.c_str());
		if (func != nullptr) {
			mapping.ipair.first = (instruction_handler<W>) func;
		}
	}

#ifndef RISCV_TRANSLATION_CACHE
	// delete program
	unlink(filename);
#endif
	// close dylib when machine is destructed
	machine().memory.set_binary_translated(dylib);
#ifdef BINTR_TIMING
	TIME_POINT(t12);
	printf(">> Activating binary translation took %ld ns\n", nanodiff(t11, t12));
	printf(">> Bintr totals %.2f ms\n", nanodiff(t0, t12) / 1e6);
#endif
}

	template void CPU<4>::try_translate(const MachineOptions<4>&, address_t, std::vector<instr_pair>&) const;
	template void CPU<8>::try_translate(const MachineOptions<8>&, address_t, std::vector<instr_pair>&) const;
	static_assert(!compressed_enabled,
		"C-extension incompatible with binary translation");

	timespec time_now()
	{
		timespec t;
		clock_gettime(CLOCK_MONOTONIC, &t);
		return t;
	}
	long nanodiff(timespec start_time, timespec end_time)
	{
		return (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
	}
}
