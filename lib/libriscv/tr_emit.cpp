#include "machine.hpp"
#include "decoder_cache.hpp"
#include "instruction_list.hpp"
#include "dyncall.hpp"
#include <algorithm>
#include <array>
#include <inttypes.h>
#include <optional>
#include <set>
#include "rv32i_instr.hpp"
#include "rvfd.hpp"
#include "tr_types.hpp"
#ifdef RISCV_EXT_C
#include "rvc.hpp"
#endif
#ifdef RISCV_EXT_VECTOR
#include "rvv.hpp"
#endif

#define PCRELA(x) ((address_t) (this->pc() + (x)))
#define PCRELS(x) hex_address(PCRELA(x)) + "LL"
#define STRADDR(x) (hex_address(x) + "LL")
// libtcc: resolve handler at translate time; shared builds dispatch at runtime
#define UNKNOWN_INSTRUCTION() { \
  this->invalidate_all_bounds_checks(); \
  if (tinfo.is_libtcc) { \
	if (!instr.is_illegal()) { \
		this->store_loaded_registers(); \
		const uintptr_t handler = (uintptr_t)CPU<W>::decode(instr).handler; \
		code += "if (api.execute_handler(cpu, " + std::to_string(instr.whole) + ", " + std::to_string(handler) + "))\n" \
			"  RETURN_VALUES(0, 0);\n"; \
		this->reload_all_registers(); \
	} else if (m_zero_insn_counter <= 1) { \
		code += "api.exception(cpu, " + STRADDR(this->pc()) + ", ILLEGAL_OPCODE);\n"; \
		code += "RETURN_VALUES(0, 0);\n"; \
	} \
  } else { \
	if (!instr.is_illegal()) { \
		this->store_loaded_registers(); \
		code += "#ifdef __wasm__\n"; \
		code += "api.execute(cpu, " + std::to_string(instr.whole) + ");\n"; \
		code += "#else\n"; \
		code += "{ static int handler_idx = 0;\n"; \
		code += "if (handler_idx) api.handlers[handler_idx](cpu, " + std::to_string(instr.whole) + ");\n"; \
		code += "else handler_idx = api.execute(cpu, " + std::to_string(instr.whole) + "); }\n"; \
		code += "#endif\n"; \
		this->reload_all_registers(); \
	} else if (m_zero_insn_counter <= 1) \
		code += "api.exception(cpu, " + hex_address(this->pc()) + ", ILLEGAL_OPCODE);\n"; \
  } \
}
// Handler may clobber any GPR; invalidate all bounds-check windows.
#define WELL_KNOWN_INSTRUCTION() { \
  this->invalidate_all_bounds_checks(); \
  if (tinfo.is_libtcc) { \
	const uintptr_t handler = (uintptr_t)CPU<W>::decode(instr).handler; \
	code += "if (api.execute_handler(cpu, " + std::to_string(instr.whole) + ", " + std::to_string(handler) + "))\n" \
		"  RETURN_VALUES(0, 0);\n"; \
  } else { \
	code += "#ifdef __wasm__\n"; \
	code += "api.execute(cpu, " + std::to_string(instr.whole) + ");\n"; \
	code += "#else\n"; \
    code += "{ static int handler_idx = 0;\n"; \
    code += "if (handler_idx) api.handlers[handler_idx](cpu, " + std::to_string(instr.whole) + ");\n"; \
    code += "else handler_idx = api.execute(cpu, " + std::to_string(instr.whole) + "); }\n"; \
	code += "#endif\n"; \
  } \
}

namespace riscv {
static const std::string LOOP_EXPRESSION = "LIKELY(ic < max_ic)";
static const std::string SIGNEXTW = "(int32_t)";
static constexpr int ALIGN_MASK = (compressed_enabled) ? 0x1 : 0x3;

static std::string hex_address(uint64_t addr) {
	char buf[64];
	if (const int len = snprintf(buf, sizeof(buf), "0x%" PRIx64, uint64_t(addr)); len > 0)
		return std::string(buf, len);
	throw MachineException(INVALID_PROGRAM, "Failed to format address");
}

template <int W>
static std::string funclabel(const std::string& func, uint64_t addr) {
	char buf[64];
	if (const int len = snprintf(buf, sizeof(buf), "%s_%" PRIx64, func.c_str(), addr); len > 0)
		return std::string(buf, len);
	throw MachineException(INVALID_PROGRAM, "Failed to format function label");
}
#define FUNCLABEL(addr) funclabel<W>(func, addr)

struct BranchInfo {
	bool sign;
	bool ignore_instruction_limit;
	uint64_t jump_pc;
	uint64_t call_pc;
};

template <int W>
struct Emitter
{
	static constexpr unsigned XLEN = W * 8u;
	// An optimizing compiler allocates locals for free, so every register is a
	// local: a register left in cpu->r[] costs a memory access on every use.
	static constexpr int LIBTCC_CACHED_REGISTERS = 32;
	static constexpr int SYSCC_CACHED_REGISTERS  = 32;
	// A return is resolved by comparing against the known call sites before
	// falling back to indirect dispatch; beyond this many sites the compares
	// are more likely to miss than the branch predictor.
	static constexpr size_t MAX_PREDICTED_RETURN_SITES = 4;
	const int CACHED_REGISTERS;
	using address_t = address_type<W>;
	using saddr_t = signed_address_type<W>;

	bool uses_register_caching() const noexcept { return tinfo.use_register_caching; }

	Emitter(const TransInfo<W>& ptinfo)
		: CACHED_REGISTERS(ptinfo.is_libtcc ? LIBTCC_CACHED_REGISTERS : SYSCC_CACHED_REGISTERS),
		  m_pc(ptinfo.basepc), tinfo(ptinfo)
	{
		this->func = funclabel<W>("f", this->pc());
		this->m_arena_hex_address = hex_address(tinfo.arena_ptr) + "L";

		if (ptinfo.use_automatic_nbit_address_space) {
			// Calculate the encompassing arena bits, which is the highest bit set in the arena size
			int encompassing_arena_bits = 0;
			for (uint64_t i = 1; i < ptinfo.arena_size; i <<= 1)
				encompassing_arena_bits++;
			this->m_encompassing_arena_mask = (1ULL << encompassing_arena_bits) - 1;
		}
	}

	template <typename ... Args>
	void add_code(Args&& ... addendum) {
		([&] {
			this->code += std::string(addendum) + "\n";
		}(), ...);
	}
	const std::string& get_code() const noexcept { return this->code; }

	std::string loaded_regname(int reg) {
		return "reg" + std::to_string(reg);
	}
	void load_register(int reg) {
		if (uses_register_caching()) {
			if (LIKELY(reg != 0 && reg < CACHED_REGISTERS))
				gpr_exists[reg] = true;
		}
	}
	void potentially_reload_register(int reg) {
		// Reload from memory; may invalidate bounds-check.
		this->invalidate_bounds_checks(reg);
		if (uses_register_caching()) {
			if (reg != 0 && reg < CACHED_REGISTERS) {
				add_code(loaded_regname(reg) + " = cpu->r[" + std::to_string(reg) + "];");
			}
		}
	}
	void potentially_realize_register(int reg) {
		if (uses_register_caching()) {
			if (reg != 0 && reg < CACHED_REGISTERS) {
				add_code("cpu->r[" + std::to_string(reg) + "] = " + loaded_regname(reg) + ";");
			}
		}
	}
	void potentially_realize_registers(int x0, int x1) {
		if (uses_register_caching()) {
			for (int reg = x0; reg < x1; reg++) {
				if (reg != 0 && reg < CACHED_REGISTERS) {
					add_code("cpu->r[" + std::to_string(reg) + "] = " + loaded_regname(reg) + ";");
				}
			}
		}
	}

	void reload_all_registers() {
		this->invalidate_all_bounds_checks();
		if (uses_register_caching())
			add_code("LOAD_REGS_" + this->func + "();");
	}
	uint32_t dirty_registers() const noexcept {
		return (index() < m_dirty_at.size()) ? m_dirty_at[index()] : ~0u;
	}
	void store_registers(uint32_t mask) {
		if (uses_register_caching()) {
			this->m_store_masks.insert(mask);
			add_code("STORE_REGS_" + this->func + "_" + hex_address(mask) + "();");
		}
	}
	void store_loaded_registers() {
		this->store_registers(this->dirty_registers());
	}
	const auto& get_store_masks() const noexcept { return this->m_store_masks; }
	void reload_syscall_registers() {
		this->invalidate_bounds_checks(10);
		this->invalidate_bounds_checks(11);
		if (uses_register_caching())
			add_code("LOAD_SYS_REGS_" + this->func + "();");
	}

	void exit_function(const std::string& new_pc, bool add_bracket = false)
	{
		this->store_loaded_registers();
		const char* return_code = (tinfo.ignore_instruction_limit) ? "RETURN_VALUES(0, max_ic);" : "RETURN_VALUES(ic, max_ic);";
		add_code(
			(new_pc != "cpu->pc") ? "cpu->pc = " + new_pc + ";" : "",
			return_code, (add_bracket) ? " }" : "");
	}

	std::string from_reg(int reg) {
		if (reg == 3 && tinfo.gp != 0)
			return hex_address(tinfo.gp) + "LL";
		else if (reg != 0) {
			if (auto tracked_value = get_tracked_register(reg); tinfo.is_libtcc && tracked_value) {
				return "(" + hex_address(*tracked_value) + "LL)";
			}
			else if (uses_register_caching() && reg < CACHED_REGISTERS) {
				load_register(reg);
				return loaded_regname(reg);
			} else {
				return "cpu->r[" + std::to_string(reg) + "]";
			}
		}
		return "(addr_t)0";
	}
	std::string from_untracked_reg(int reg) {
		if (reg == 3 && tinfo.gp != 0)
			return hex_address(tinfo.gp) + "L";
		else if (reg != 0) {
			if (uses_register_caching() && reg < CACHED_REGISTERS) {
				load_register(reg);
				return loaded_regname(reg);
			} else {
				return "cpu->r[" + std::to_string(reg) + "]";
			}
		}
		return "(addr_t)0";
	}
	// Every GPR write passes through to_reg(); invalidating here kills stale bounds-check windows.
	std::string to_reg(int reg) {
		if (reg != 0) {
			this->invalidate_bounds_checks(reg);
			if (uses_register_caching() && reg < CACHED_REGISTERS) {
				load_register(reg);
				this->gpr_written[reg] = true;
				return loaded_regname(reg);
			} else {
				return "cpu->r[" + std::to_string(reg) + "]";
			}
		}
		return "(addr_t)0";
	}
	std::string from_fpreg(int reg) {
		return "cpu->fr[" + std::to_string(reg) + "]";
	}
#ifdef RISCV_EXT_VECTOR
	std::string from_rvvreg(int reg) {
		return "cpu->rvv.lane[" + std::to_string(reg) + "]";
	}
	// Elements in one full register at this SEW, ie. VLMAX at LMUL=1.
	static constexpr unsigned rvv_full_vl(unsigned vsew) noexcept {
		return VectorLane::size() >> vsew;
	}
	// Full m1 lane at this SEW: all elements active, no tail.
	// Matches the fast path in unit_stride_load/store.
	std::string rvv_full_lane_condition(unsigned vsew) const {
		// An inlined vsetvli leaves vl in a C local, sparing us a reload.
		std::string cond = (m_vl_local.empty() ? "cpu->rvv.vl" : m_vl_local)
			+ " == " + std::to_string(rvv_full_vl(vsew));
		if (m_vtype.known)
			return cond;
		return cond + " && cpu->rvv.vsew == " + std::to_string(vsew)
			+ " && cpu->rvv.lmul == 0 && !cpu->rvv.vill";
	}
	// Per-SEW guard as a C local, shared by all vector instructions in the block.
	const std::string& rvv_guard(unsigned vsew)
	{
		auto& guard = m_rvv_guard.at(vsew);
		if (guard.empty()) {
			// With vl known too the guard folds to a constant, and the C
			// compiler drops either the body or the fallback entirely.
			if (m_vtype.known && m_vl_known)
				guard = (m_vl == rvv_full_vl(vsew)) ? "1" : "0";
			else {
				guard = "vok" + PCRELS(0) + "_" + std::to_string(vsew);
				// Declared and assigned separately: an embedded translation
				// is compiled as C++, where a goto may not jump over an
				// initialization, and labels land between vector blocks.
				add_code("int " + guard + "; " + guard + " = "
					+ rvv_full_lane_condition(vsew) + ";");
			}
		}
		return guard;
	}
	// vl is about to change, or may have changed behind our back. SEW, LMUL
	// and vill are untouched, and so is anything proven from them.
	void reset_vector_vl() {
		this->m_vl_known = false;
		this->m_vl_local.clear();
		for (auto& guard : this->m_rvv_guard) guard.clear();
	}
	void reset_vector_config() {
		this->m_vtype = {};
		this->reset_vector_vl();
	}
	// Call the interpreter's instruction handler (non-libtcc slow path)
	void emit_vector_handler_call(const rv32i_instruction& instr) {
		code += "#ifdef __wasm__\n";
		code += "api.execute(cpu, " + std::to_string(instr.whole) + ");\n";
		code += "#else\n";
		code += "{ static int handler_idx = 0;\n";
		code += "if (handler_idx) api.handlers[handler_idx](cpu, " + std::to_string(instr.whole) + ");\n";
		code += "else handler_idx = api.execute(cpu, " + std::to_string(instr.whole) + "); }\n";
		code += "#endif\n";
	}

	// The integer registers potentially used by a vector instruction handler
	struct VectorScalarUse {
		int reads[2] { -1, -1 };
		int writes { -1 };
	};
	static VectorScalarUse vector_scalar_use(const rv32i_instruction& vinstr)
	{
		const rv32v_instruction vi { vinstr };
		VectorScalarUse use;
		switch (vinstr.opcode()) {
		case RV32V_OP:
			switch (vinstr.vwidth()) {
			case 0x4: // OPI.VX: scalar operand from x[rs1]
				use.reads[0] = vi.OPVI.imm;
				break;
			case 0x6: // OPM.VX: scalar operand from x[rs1] (VMV.S.X, VSLIDE1*.VX, ...)
				use.reads[0] = vi.OPVV.vs1;
				break;
			case 0x2: // OPM.VV: VWXUNARY0 (VMV.X.S, VCPOP.M, VFIRST.M) writes x[rd]
				if (vi.OPVV.funct6 == 0b010000)
					use.writes = vi.OPVV.vd;
				break;
			case 0x7: // Vector configuration: AVL from x[rs1], resulting vl into x[rd]
				switch (vinstr.vsetfunc()) {
				case 0x0:
				case 0x1: // VSETVLI
					use.reads[0] = vi.VLI.rs1;
					break;
				case 0x2: // VSETVL: new vtype comes from x[rs2] as well
					use.reads[0] = vi.VSETVL.rs1;
					use.reads[1] = vi.VSETVL.rs2;
					break;
				default: // VSETIVLI: the AVL is a 5-bit immediate
					break;
				}
				use.writes = vi.VLI.rd;
				break;
			default: // OPI.VV, OPI.VI, OPF.VV, OPF.VF: vector and fp registers only
				break;
			}
			break;
		case RV32F_LOAD:  // Vector loads address memory through x[rs1]
			use.reads[0] = vi.VL.rs1;
			// A strided access takes its byte stride from x[rs2] too.
			if (vi.VL.mop == 0b10)
				use.reads[1] = vi.VLS.rs2;
			break;
		case RV32F_STORE: // x[rs1] likewise
			use.reads[0] = vi.VS.rs1;
			if (vi.VS.mop == 0b10)
				use.reads[1] = vi.VSS.rs2;
			break;
		default:
			break;
		}
		return use;
	}
	// Realize the integer registers the handler reads.
	void realize_vector_scalar_reads(const VectorScalarUse& use) {
		for (const int reg : use.reads) {
			if (reg > 0) {
				this->load_register(reg);
				this->potentially_realize_register(reg);
			}
		}
	}
	// Reload the one integer register the handler may have written.
	void reload_vector_scalar_writes(const VectorScalarUse& use) {
		if (use.writes > 0) {
			this->reset_tracked_register(use.writes);
			this->potentially_reload_register(use.writes);
		}
	}
	// Group alignment: must be aligned to its size and fit in v0-v31. Fractional LMUL = single register.
	bool vector_group_is_aligned(unsigned vreg) const noexcept
	{
		const unsigned regs = (m_vtype.lmul >= 0) ? (1u << m_vtype.lmul) : 0u;
		return regs <= 1 || (vreg % regs == 0 && vreg + regs <= 32);
	}
	// True if this encoding's handler cannot trap. Three preconditions (valid vtype,
	// supported SEW, aligned register group) are proven by an earlier inlined vsetvli;
	// without a known vtype the answer is always yes.
	bool vector_handler_can_trap(const rv32i_instruction& vinstr) const
	{
		// Loads and stores fault on an address that is not known here.
		if (vinstr.opcode() != RV32V_OP)
			return true;
		if (!m_vtype.known || m_vtype.vill)
			return true;
		const rv32v_instruction vi { vinstr };
		const unsigned f6 = vi.OPVV.funct6;
		// The floating-point families are defined for SEW 32 and 64 only.
		const bool fp_sew = (m_vtype.vsew == 2 || m_vtype.vsew == 3);
		// vs1 shares its field with the immediate, and is only a register
		// group in the .vv forms.
		const bool vd_ok  = vector_group_is_aligned(vi.OPVV.vd);
		const bool vs1_ok = vector_group_is_aligned(vi.OPVV.vs1);
		const bool vs2_ok = vector_group_is_aligned(vi.OPVV.vs2);

		switch (vinstr.vwidth()) {
		case 0x0: // OPI.VV: elements out of three SEW-sized groups
			if (!vd_ok || !vs1_ok || !vs2_ok)
				return true;
			switch (f6) {
			case 0b000000: // VADD.VV
			case 0b000001: // VANDN.VV
			case 0b000010: // VSUB.VV
			case 0b000100: // VMINU.VV
			case 0b000101: // VMIN.VV
			case 0b000110: // VMAXU.VV
			case 0b000111: // VMAX.VV
			case 0b001001: // VAND.VV
			case 0b001010: // VOR.VV
			case 0b001011: // VXOR.VV
			case 0b001100: // VRGATHER.VV
			case 0b010001: // VMADC.VV(M)
			case 0b010011: // VMSBC.VV(M)
			case 0b010111: // VMERGE.VVM / VMV.V.V
			case 0b011000: // VMSEQ.VV
			case 0b011001: // VMSNE.VV
			case 0b011010: // VMSLTU.VV
			case 0b011011: // VMSLT.VV
			case 0b011100: // VMSLEU.VV
			case 0b011101: // VMSLE.VV
			case 0b100000: // VSADDU.VV
			case 0b100001: // VSADD.VV
			case 0b100010: // VSSUBU.VV
			case 0b100011: // VSSUB.VV
			case 0b100101: // VSLL.VV
			case 0b101000: // VSRL.VV
			case 0b101001: // VSRA.VV
				return false;
			case 0b010000: // VADC.VVM
			case 0b010010: // VSBC.VVM
				// vm=1 is reserved for these two: v0 is the carry, not a mask.
				return vi.OPVV.vm != 0;
			default:
				return true;
			}
		case 0x3: // OPI.VI
		case 0x4: // OPI.VX: the second source is an immediate or x[rs1]
			if (!vd_ok || !vs2_ok)
				return true;
			switch (f6) {
			case 0b000000: // VADD
			case 0b000011: // VRSUB
			case 0b001001: // VAND
			case 0b001010: // VOR
			case 0b001011: // VXOR
			case 0b001100: // VRGATHER
			case 0b001110: // VSLIDEUP
			case 0b001111: // VSLIDEDOWN
			case 0b010111: // VMERGE / VMV.V.I|X
			case 0b011000: // VMSEQ
			case 0b011001: // VMSNE
			case 0b011010: // VMSLTU
			case 0b011011: // VMSLT
			case 0b011100: // VMSLEU
			case 0b011101: // VMSLE
			case 0b011110: // VMSGTU
			case 0b011111: // VMSGT
			case 0b100000: // VSADDU
			case 0b100001: // VSADD
			case 0b100010: // VSSUBU
			case 0b100011: // VSSUB
			case 0b100101: // VSLL
			case 0b101000: // VSRL
			case 0b101001: // VSRA
				return false;
			case 0b000001: // VANDN: there is no .vi form
				return vinstr.vwidth() != 0x4;
			default:
				return true;
			}
		case 0x2: // OPM.VV
		case 0x6: // OPM.VX
			switch (f6) {
			case 0b011000: // VMANDN.MM
			case 0b011001: // VMAND.MM
			case 0b011010: // VMOR.MM
			case 0b011011: // VMXOR.MM
			case 0b011100: // VMORN.MM
			case 0b011101: // VMNAND.MM
			case 0b011110: // VMNOR.MM
			case 0b011111: // VMXNOR.MM
				// Whole single mask registers: no group to align. The .vx
				// forms are reserved and fall through to the handler's trap.
				return vinstr.vwidth() != 0x2;
			case 0b100000: // VDIVU
			case 0b100001: // VDIV
			case 0b100010: // VREMU
			case 0b100011: // VREM
			case 0b100100: // VMULHU
			case 0b100101: // VMUL
			case 0b100110: // VMULHSU
			case 0b100111: // VMULH
			case 0b101001: // VMADD
			case 0b101011: // VNMSUB
			case 0b101101: // VMACC
			case 0b101111: // VNMSAC
				return !vd_ok || !vs2_ok
					|| (vinstr.vwidth() == 0x2 && !vs1_ok);
			default:
				return true;
			}
		case 0x1: // OPF.VV
			// Only element-wise OPF.VV arithmetic; widening/reduction/conversion handlers have their own checks.
			if (!RVV_IS_OPFVV_ARITH(f6))
				return true;
			return !fp_sew || !vd_ok || !vs1_ok || !vs2_ok;
		case 0x5: // OPF.VF
			if (!fp_sew || !vd_ok || !vs2_ok)
				return true;
			switch (f6) {
			case 0b000000: // VFADD.VF
			case 0b000010: // VFSUB.VF
			case 0b000100: // VFMIN.VF
			case 0b000110: // VFMAX.VF
			case 0b001000: // VFSGNJ.VF
			case 0b001001: // VFSGNJN.VF
			case 0b001010: // VFSGNJX.VF
			case 0b001110: // VFSLIDE1UP.VF
			case 0b001111: // VFSLIDE1DOWN.VF
			case 0b010000: // VFMV.S.F
			case 0b010111: // VFMERGE.VFM / VFMV.V.F
			case 0b011000: // VMFEQ.VF
			case 0b011001: // VMFLE.VF
			case 0b011011: // VMFLT.VF
			case 0b011100: // VMFNE.VF
			case 0b011101: // VMFGT.VF
			case 0b011111: // VMFGE.VF
			case 0b100000: // VFDIV.VF
			case 0b100001: // VFRDIV.VF
			case 0b100100: // VFMUL.VF
			case 0b100111: // VFRSUB.VF
			case 0b101000: case 0b101001: case 0b101010: case 0b101011:
			case 0b101100: case 0b101101: case 0b101110: case 0b101111:
				return false; // ... and the eight fused multiply-adds
			default:
				return true;
			}
		default: // 0x7: vector configuration
			return true;
		}
	}
	// True if the handler may modify vl or vtype. Only vsetvl* and fault-only-first loads do.
	static bool vector_handler_can_reconfigure(const rv32i_instruction& vinstr)
	{
		if (vinstr.opcode() == RV32V_OP)
			return vinstr.vwidth() == 0x7;
		if (vinstr.opcode() != RV32F_LOAD)
			return false; // vector stores never write vl
		const rv32v_instruction vi { vinstr };
		return vi.VL.mop == 0b00 && vi.VL.lumop == 0b10000;
	}
	// Leave the block when a vector handler has raised.
	void emit_vector_trap_branch(const std::string& condition)
	{
		code += "if (UNLIKELY(" + condition + "))";
		if (this->uses_register_caching()) {
			this->m_used_vector_trap = true;
			code += " goto " + this->vector_trap_label() + ";\n";
		} else {
			// Nothing to store, so the exit is already as small as the jump.
			code += " {\nRETURN_VALUES(0, 0);\n}\n";
		}
	}
	void emit_vector_handler_invoke()
	{
		if (!tinfo.is_libtcc) {
			this->emit_vector_handler_call(instr);
			return;
		}
		const uintptr_t handler = (uintptr_t)CPU<W>::decode(instr).handler;
		const std::string call = "api.execute_handler(cpu, "
			+ std::to_string(instr.whole) + ", " + std::to_string(handler) + ")";
		// Nothing follows a call that has been proven unable to raise.
		if (!this->m_vtrap_needed)
			add_code(call + ";");
		else
			this->emit_vector_trap_branch(call);
	}
	std::string vector_trap_label() const {
		return this->func + "_vtrap";
	}
	void emit_vector_trap_epilogue()
	{
		if (!this->m_used_vector_trap)
			return;
		add_code(this->vector_trap_label() + ":;");
		this->store_registers(~0u); // Reached from anywhere in the block
		add_code("RETURN_VALUES(0, 0);");
	}
	// Interpreter fallback: realize/reload only the named scalar registers.
	void emit_vector_slowpath()
	{
		// Fault-only-first loads shorten vl; vsetvl* rewrites the whole vtype.
		if (vector_handler_can_reconfigure(instr)) {
			if (instr.opcode() == RV32V_OP)
				this->reset_vector_config();
			else
				this->reset_vector_vl();
		}
		const auto use = vector_scalar_use(this->instr);
		this->realize_vector_scalar_reads(use);
		this->emit_vector_handler_invoke();
		this->reload_vector_scalar_writes(use);
	}
	// Whether an arena bounds check is needed at runtime.
	bool vector_memory_needs_bounds_check() noexcept {
		return !tinfo.unsafe_remove_checks && !uses_Nbit_encompassing_arena();
	}
	// Element-wise float operator, or nullptr when not inlined.
	static const char* vector_float_operator(const rv32i_instruction& vinstr)
	{
		if (vinstr.vwidth() != 0x1 && vinstr.vwidth() != 0x5) // OPF.VV / OPF.VF
			return nullptr;
		if (!rv32v_instruction{vinstr}.OPVV.vm)
			return nullptr; // Masked: element activity depends on v0 at runtime
		switch (rv32v_instruction{vinstr}.OPVV.funct6) {
		case 0b000000: return " + "; // VFADD
		case 0b000010: return " - "; // VFSUB
		case 0b100000: return " / "; // VFDIV
		case 0b100100: return " * "; // VFMUL
		default:       return nullptr;
		}
	}
	// True for the eight fused multiply-add encodings shared by OPFVV and
	// OPFVF. They read the destination as their third operand.
	static bool vector_is_float_fma(const rv32i_instruction& vinstr)
	{
		if (vinstr.vwidth() != 0x1 && vinstr.vwidth() != 0x5) // OPF.VV / OPF.VF
			return false;
		if (!rv32v_instruction{vinstr}.OPVV.vm)
			return false; // Masked: element activity depends on v0 at runtime
		return (rv32v_instruction{vinstr}.OPVV.funct6 & 0b111000) == 0b101000;
	}
	// EEW log2 for unit-stride widths, or -1 for non-standard widths.
	static int vector_unit_stride_sew(uint32_t width) noexcept
	{
		switch (width) {
		case 0b000: return 0; // VLE8  / VSE8
		case 0b101: return 1; // VLE16 / VSE16
		case 0b110: return 2; // VLE32 / VSE32
		case 0b111: return 3; // VLE64 / VSE64
		default:    return -1;
		}
	}
	// Helpers to inline vector memory forms that are simple enough to be expressed in C
	enum class VectorMemForm {
		None,             // left to the interpreter handler
		UnitStride,       // vle<eew>.v / vse<eew>.v: vl*EEW bytes
		MaskedUnitStride, // ... the same, with v0 choosing the elements
		WholeRegister,    // vl<n>re<eew>.v / vs<n>r.v: n registers, no vtype
		Mask,             // vlm.v / vsm.v: ceil(vl/8) bytes
	};
	struct VectorMemInfo {
		VectorMemForm form = VectorMemForm::None;
		bool     is_store = false;
		unsigned eew_log2 = 0; // log2(EEW / 8), the unit-stride forms
		unsigned nregs    = 1; // registers moved by a whole-register transfer
		unsigned vreg     = 0; // vd, or vs3 in a store
		unsigned rs1      = 0; // the base address
	};
	// Max inlined transfer: 8 registers (LMUL=8 / widest whole-register form).
	static constexpr uint64_t vector_max_transfer() noexcept {
		return 8ull * VectorLane::size();
	}
	// Static classification of vector memory ops. Only vtype-dependent checks remain at runtime.
	VectorMemInfo vector_memory_form(const rv32i_instruction& vinstr)
	{
		const rv32v_instruction vi { vinstr };
		VectorMemInfo info;
		if (vinstr.opcode() == RV32F_STORE)
			info.is_store = true;
		else if (vinstr.opcode() != RV32F_LOAD)
			return {};
		// Every body below reaches guest memory as a host pointer.
		if (!uses_flat_memory_arena())
			return {};
		// MEW asks for an element wider than 64 bits, and a mop other than 00
		// is a strided or indexed access: both belong to the handler.
		if (vi.VL.mew || vi.VL.mop != 0b00)
			return {};
		info.vreg = info.is_store ? vi.VS.vs3 : vi.VL.vd;
		info.rs1  = info.is_store ? vi.VS.rs1 : vi.VL.rs1;
		const unsigned nf = vi.VL.nf + 1; // fields per segment
		switch (info.is_store ? vi.VS.sumop : vi.VL.lumop) {
		case 0b00000: { // The plain unit-stride access
			if (nf != 1)
				return {}; // Segmented: several fields share one address
			const int eew = vector_unit_stride_sew(vi.VL.width);
			if (eew < 0)
				return {};
			info.eew_log2 = unsigned(eew);
			info.form = vi.VL.vm ? VectorMemForm::UnitStride
			                     : VectorMemForm::MaskedUnitStride;
			return info; }
		case 0b01000: // vl<n>re<eew>.v / vs<n>r.v
			// An illegal register count, or a group that is not aligned to
			// its own size, has to trap: only the handler knows how.
			if (!vi.VL.vm || (nf != 1 && nf != 2 && nf != 4 && nf != 8))
				return {};
			if ((info.vreg % nf) != 0 || info.vreg + nf > 32)
				return {};
			info.nregs = nf;
			info.form = VectorMemForm::WholeRegister;
			return info;
		case 0b01011: // vlm.v / vsm.v
			if (!vi.VL.vm || nf != 1 || vi.VL.width != 0)
				return {};
			info.form = VectorMemForm::Mask;
			return info;
		default: // Fault-only-first, and the encodings that trap
			return {};
		}
	}
	// Inlinable SEWs for float arithmetic. Empty = use interpreter.
	std::vector<unsigned> vector_inlinable_sews(const rv32i_instruction& vinstr)
	{
		if (vinstr.opcode() != RV32V_OP)
			return {};
		if (vector_float_operator(vinstr) == nullptr && !vector_is_float_fma(vinstr))
			return {};
		// Both float widths are inlinable: a known vtype selects one,
		// otherwise both bodies are emitted behind their own guard.
		if (!m_vtype.known)
			return { 2, 3 };
		// An inline body is only reachable at the SEW and LMUL it was written
		// for, so a known vtype decides it here, not at runtime.
		if (m_vtype.vill || m_vtype.lmul != 0 || m_vtype.vsew < 2)
			return {}; // Fractional/multi-register groups, and no 8/16-bit floats
		return { m_vtype.vsew };
	}
	// The live vl, as a C expression: the local an inlined vsetvli left
	// behind, or the field itself.
	std::string vector_vl_expr() const {
		return m_vl_local.empty() ? "cpu->rvv.vl" : m_vl_local;
	}
	// Runtime checks for a unit-stride transfer, minus what the proven vtype already settled.
	struct VectorStridePlan {
		std::vector<std::string> locals; // declared before the guard is tested
		std::string guard;       // empty when nothing is left to test
		std::string bytes;       // C expression for the length of the run
		bool     bytes_known = false;
		uint64_t known_bytes = 0;
		unsigned dregs = 0;      // registers in the group, 0 when runtime-only
	};
	// Returns false when this encoding cannot be inlined here at all, in
	// which case the caller hands the instruction over whole.
	bool vector_plan_unit_stride(const VectorMemInfo& info, VectorStridePlan& plan)
	{
		constexpr uint64_t lane_size = VectorLane::size();
		const unsigned eew = info.eew_log2;
		if (m_vtype.known) {
			// EMUL = LMUL * EEW / SEW, as a count of registers.
			const int emul = m_vtype.lmul + int(eew) - int(m_vtype.vsew);
			if (m_vtype.vill || emul > 3 || emul < -3)
				return false;
			// A group of one needs no alignment, which covers every EEW at
			// LMUL <= 1; anything wider has to be aligned to its own size.
			plan.dregs = (emul > 0) ? (1u << emul) : 1u;
			if (plan.dregs > 1 && ((info.vreg % plan.dregs) != 0
				|| info.vreg + plan.dregs > 32))
				return false;
			const uint64_t max_bytes = uint64_t(plan.dregs) * lane_size;
			if (m_vl_known) {
				plan.known_bytes = m_vl << eew;
				plan.bytes_known = true;
				// vl outliving the vtype that sized it is rare enough to
				// leave to the handler.
				if (plan.known_bytes > max_bytes)
					return false;
				plan.bytes = std::to_string(plan.known_bytes);
			} else {
				plan.bytes = "vnb" + PCRELS(0);
				plan.locals.push_back("const addr_t " + plan.bytes + " = (addr_t)"
					+ vector_vl_expr() + " << " + std::to_string(eew) + ";");
				plan.guard = plan.bytes + " <= " + std::to_string(max_bytes);
			}
			return true;
		}
		// vtype is live, so the same arithmetic goes into the guard. The
		// destination register is a constant, so the alignment test folds
		// down to a compare once the C compiler knows EMUL.
		const std::string emul = "vem" + PCRELS(0);
		const std::string vd = std::to_string(info.vreg);
		plan.bytes = "vnb" + PCRELS(0);
		plan.locals.push_back("const int " + emul + " = (int)cpu->rvv.lmul + "
			+ std::to_string(eew) + " - (int)cpu->rvv.vsew;");
		plan.locals.push_back("const addr_t " + plan.bytes + " = (addr_t)"
			+ vector_vl_expr() + " << " + std::to_string(eew) + ";");
		plan.guard = "!cpu->rvv.vill && " + emul + " <= 3 && " + emul + " >= -3"
			" && (" + emul + " <= 0 || ((" + vd + " & ((1 << " + emul + ") - 1)) == 0"
			" && " + vd + " + (1 << " + emul + ") <= 32))"
			" && " + plan.bytes + " <= ((addr_t)" + std::to_string(lane_size)
			+ " << (" + emul + " > 0 ? " + emul + " : 0))";
		return true;
	}
	// Emit prologue: base address, combined vtype+arena guard. Returns true if a fallback else-branch exists.
	bool emit_vector_memory_prologue(const VectorMemInfo& info,
		const std::string& addr, const std::vector<std::string>& locals,
		const std::string& guard)
	{
		this->load_register(int(info.rs1));
		add_code("{ const addr_t " + addr + " = " + from_untracked_reg(int(info.rs1)) + ";");
		for (const auto& local : locals)
			add_code("  " + local);

		std::string cond = guard;
		if (this->vector_memory_needs_bounds_check()) {
			// The longest run still fits within the arena over-allocation, so
			// a readable/writable start address covers all of it.
			static_assert(vector_max_transfer() <= Memory<W>::OVERALLOCATE,
				"A vector register group must fit within the arena over-allocation");
			const std::string bounds =
				std::string(info.is_store ? "ARENA_WRITABLE(" : "ARENA_READABLE(") + addr + ")";
			cond = cond.empty() ? bounds : "(" + cond + ") && " + bounds;
		}
		if (!cond.empty())
			add_code("if (LIKELY(" + cond + ")) {");
		else
			add_code("{");
		return !cond.empty();
	}
	void emit_vector_memory_epilogue(const VectorMemInfo& info, bool has_fallback)
	{
		if (has_fallback) {
			add_code("} else {");
			// rs1 is the handler's only integer input, and it writes none.
			this->potentially_realize_register(int(info.rs1));
			this->emit_vector_handler_invoke();
		}
		add_code("}", "}");
	}
	// One VLEN-sized register, moved as a single struct assignment.
	void emit_vector_lane_move(const VectorMemInfo& info,
		const std::string& addr, unsigned index)
	{
		const uint64_t offset = uint64_t(index) * VectorLane::size();
		const std::string at = offset
			? "(" + addr + " + " + std::to_string(offset) + ")" : addr;
		const std::string lane = "*(VectorLaneBytes *)&" + from_rvvreg(int(info.vreg + index));
		const std::string mem  = "*(VectorLaneBytes *)" + arena_at(at);
		add_code(info.is_store ? "  " + mem + " = " + lane + ";"
		                       : "  " + lane + " = " + mem + ";");
	}
	// Byte-loop fallback for partial-register runs (strip-mine tail, mask loads).
	void emit_vector_byte_move(const VectorMemInfo& info,
		const std::string& addr, const std::string& count)
	{
		const std::string i   = "vbi" + PCRELS(0);
		const std::string dst = "vbd" + PCRELS(0);
		const std::string src = "vbs" + PCRELS(0);
		const std::string reg = "(uint8_t *)&" + from_rvvreg(int(info.vreg));
		const std::string mem = "(uint8_t *)" + arena_at(addr);
		add_code("  { uint8_t *" + dst + " = " + (info.is_store ? mem : reg) + ";",
			"    const uint8_t *" + src + " = " + (info.is_store ? reg : mem) + ";",
			"    addr_t " + i + ";",
			"    for (" + i + " = 0; " + i + " < " + count + "; " + i + "++)",
			"      " + dst + "[" + i + "] = " + src + "[" + i + "]; }");
	}
	// Contiguous block copy. Constant-sized lanes compile to wide moves.
	void emit_vector_contiguous_move(const VectorMemInfo& info,
		const std::string& addr, const VectorStridePlan& plan)
	{
		constexpr uint64_t lane_size = VectorLane::size();
		if (plan.bytes_known) {
			if (plan.known_bytes % lane_size == 0) {
				for (unsigned r = 0; r < plan.known_bytes / lane_size; r++)
					this->emit_vector_lane_move(info, addr, r);
			} else {
				this->emit_vector_byte_move(info, addr, plan.bytes);
			}
			return;
		}
		add_code("  if (LIKELY(" + plan.bytes + " == " + std::to_string(lane_size) + ")) {");
		this->emit_vector_lane_move(info, addr, 0);
		if (plan.dregs > 1) {
			// The whole group, for the iterations of a strip-mined loop that
			// are not the last one at LMUL > 1.
			add_code("  } else if (LIKELY(" + plan.bytes + " == "
				+ std::to_string(uint64_t(plan.dregs) * lane_size) + ")) {");
			for (unsigned r = 0; r < plan.dregs; r++)
				this->emit_vector_lane_move(info, addr, r);
		}
		add_code("  } else {");
		this->emit_vector_byte_move(info, addr, plan.bytes);
		add_code("  }");
	}
	// Unit-stride: vl*EEW contiguous bytes. Adjacent group registers make wider EMUL a longer run.
	void emit_vector_unit_stride(const VectorMemInfo& info)
	{
		VectorStridePlan plan;
		if (!this->vector_plan_unit_stride(info, plan)) {
			this->emit_vector_slowpath();
			return;
		}
		// An empty transfer touches no memory, and so cannot fault.
		if (plan.bytes_known && plan.known_bytes == 0)
			return;

		const std::string addr = "vaddr" + PCRELS(0);
		const bool fallback = this->emit_vector_memory_prologue(info, addr, plan.locals, plan.guard);
		this->emit_vector_contiguous_move(info, addr, plan);
		this->emit_vector_memory_epilogue(info, fallback);
	}
	// Masked unit-stride: per-element v0 test over contiguous data.
	void emit_vector_masked_unit_stride(const VectorMemInfo& info)
	{
		// May-alias types: the elements are guest memory on one side.
		static const char* const element_types[4] =
			{ "ga_uint8_t", "ga_uint16_t", "ga_uint32_t", "ga_uint64_t" };
		VectorStridePlan plan;
		if (!this->vector_plan_unit_stride(info, plan)) {
			this->emit_vector_slowpath();
			return;
		}
		if (plan.bytes_known && plan.known_bytes == 0)
			return;
		const std::string elems = m_vl_known
			? std::to_string(m_vl) : this->vector_vl_expr();

		const std::string addr = "vaddr" + PCRELS(0);
		const bool fallback = this->emit_vector_memory_prologue(info, addr, plan.locals, plan.guard);

		const char* const type = element_types[info.eew_log2];
		const std::string i    = "vmi" + PCRELS(0);
		const std::string dst  = "vmd" + PCRELS(0);
		const std::string src  = "vms" + PCRELS(0);
		const std::string mask = "vmm" + PCRELS(0);
		const std::string reg = "(" + std::string(type) + " *)&" + from_rvvreg(int(info.vreg));
		const std::string mem = "(" + std::string(type) + " *)" + arena_at(addr);
		add_code("  { " + std::string(type) + " *" + dst + " = " + (info.is_store ? mem : reg) + ";",
			"    const " + std::string(type) + " *" + src + " = " + (info.is_store ? reg : mem) + ";",
			// v0 always holds the mask, one bit per element, and is read as
			// the loop runs so that a destination overlapping it behaves as
			// the handler's element loop does.
			"    const uint8_t *" + mask + " = (const uint8_t *)&" + from_rvvreg(0) + ";",
			"    addr_t " + i + ";",
			"    for (" + i + " = 0; " + i + " < " + elems + "; " + i + "++)",
			"      if ((" + mask + "[" + i + " >> 3] >> (" + i + " & 7)) & 1)",
			"        " + dst + "[" + i + "] = " + src + "[" + i + "]; }");

		this->emit_vector_memory_epilogue(info, fallback);
	}
	// vl<n>re<eew>.v / vs<n>r.v: a raw copy of n whole registers that reads
	// neither vl nor vtype, leaving nothing to guard but the address.
	void emit_vector_whole_register(const VectorMemInfo& info)
	{
		const std::string addr = "vaddr" + PCRELS(0);
		const bool fallback = this->emit_vector_memory_prologue(info, addr, {}, "");
		for (unsigned r = 0; r < info.nregs; r++)
			this->emit_vector_lane_move(info, addr, r);
		this->emit_vector_memory_epilogue(info, fallback);
	}
	// vlm.v / vsm.v: ceil(vl/8) bytes of mask, always EEW=8 with EMUL=1 and
	// never masked. Only vill can stop it, and vl can never ask for more than
	// the one register a mask lives in.
	void emit_vector_mask_move(const VectorMemInfo& info)
	{
		constexpr uint64_t lane_size = VectorLane::size();
		if (m_vtype.known && m_vtype.vill) {
			this->emit_vector_slowpath();
			return;
		}
		VectorStridePlan plan;
		plan.guard = m_vtype.known ? "" : "!cpu->rvv.vill";
		if (m_vl_known) {
			plan.known_bytes = (m_vl + 7) / 8;
			plan.bytes_known = true;
			if (plan.known_bytes > lane_size) {
				this->emit_vector_slowpath();
				return;
			}
			if (plan.known_bytes == 0)
				return;
			plan.bytes = std::to_string(plan.known_bytes);
		} else {
			plan.bytes = "vnb" + PCRELS(0);
			plan.locals.push_back("const addr_t " + plan.bytes + " = ((addr_t)"
				+ vector_vl_expr() + " + 7) / 8;");
			plan.guard += (plan.guard.empty() ? "" : " && ")
				+ plan.bytes + " <= " + std::to_string(lane_size);
		}
		const std::string addr = "vaddr" + PCRELS(0);
		const bool fallback = this->emit_vector_memory_prologue(info, addr, plan.locals, plan.guard);
		this->emit_vector_contiguous_move(info, addr, plan);
		this->emit_vector_memory_epilogue(info, fallback);
	}
	// Emit one vector load or store that does not need the interpreter.
	void emit_vector_memory(const VectorMemInfo& info)
	{
		switch (info.form) {
		case VectorMemForm::UnitStride:       this->emit_vector_unit_stride(info); break;
		case VectorMemForm::MaskedUnitStride: this->emit_vector_masked_unit_stride(info); break;
		case VectorMemForm::WholeRegister:    this->emit_vector_whole_register(info); break;
		case VectorMemForm::Mask:             this->emit_vector_mask_move(info); break;
		default:                              this->emit_vector_slowpath(); break;
		}
	}
	// Per-element FMA expression matching the interpreter's OPFVV/OPFVF handlers.
	static std::string vector_fma_expression(unsigned funct6, const std::string& fma,
		const std::string& a, const std::string& b, const std::string& c)
	{
		const std::string call = fma + "(";
		switch (funct6) {
		case 0b101000: return       call +       a + ", " + c + ", " + b + ")"; // VFMADD
		case 0b101001: return "-" + call +       a + ", " + c + ", " + b + ")"; // VFNMADD
		case 0b101010: return       call +       a + ", " + c + ", -" + b + ")"; // VFMSUB
		case 0b101011: return       call + "-" + a + ", " + c + ", " + b + ")"; // VFNMSUB
		case 0b101100: return       call +       a + ", " + b + ", " + c + ")"; // VFMACC
		case 0b101101: return "-" + call +       a + ", " + b + ", " + c + ")"; // VFNMACC
		case 0b101110: return       call +       a + ", " + b + ", -" + c + ")"; // VFMSAC
		default:       return       call + "-" + a + ", " + b + ", " + c + ")"; // VFNMSAC
		}
	}
	// Inline float arithmetic: guard ensures full m1 register, all elements active.
	void emit_vector_float_arith_body(unsigned vsew)
	{
		const rv32v_instruction vi { instr };
		const bool dbl = (vsew == 3);
		const bool is_scalar = instr.vwidth() == 0x5; // OPF.VF takes f[rs1]
		const char* const elem = dbl ? ".f64[" : ".f32[";
		std::string rhs;
		if (is_scalar) {
			// The guard pins SEW, so the scalar is the low element.
			rhs = "vscalar" + PCRELS(0) + "_" + std::to_string(vsew);
			add_code(std::string("  const ") + (dbl ? "double " : "float ") + rhs + " = "
				+ from_fpreg(vi.OPVV.vs1) + (dbl ? ".f64;" : ".f32[0];"));
		}
		const char* const op = vector_float_operator(instr);
		const std::string fma = dbl ? "api.vfmaf64" : "api.vfmaf32";
		const unsigned elements = VectorLane::size() >> vsew;
		for (unsigned i = 0; i < elements; i++) {
			const std::string e = elem + std::to_string(i) + "]";
			const std::string a = is_scalar ? rhs : from_rvvreg(vi.OPVV.vs1) + e;
			const std::string b = from_rvvreg(vi.OPVV.vs2) + e;
			const std::string d = from_rvvreg(vi.OPVV.vd) + e;
			add_code("  " + d + " = " + (op != nullptr
				? b + op + a
				: vector_fma_expression(vi.OPVV.funct6, fma, a, b, d)) + ";");
		}
	}
	// True for the two configuration forms with an immediate vtype, which is
	// what compilers emit. VSETVL takes it from a register instead, so its
	// vtype is only known at runtime.
	bool vector_is_inlinable_setvl(const rv32i_instruction& vinstr) const noexcept
	{
		if (vinstr.opcode() != RV32V_OP || vinstr.vwidth() != 0x7)
			return false;
		const auto func = vinstr.vsetfunc();
		return func == 0x0 || func == 0x1 || func == 0x3;
	}
	// Inline vsetvli: avoids per-iteration register spill/reload in strip-mined loops.
	void emit_vector_setvl()
	{
		const rv32v_instruction vi { instr };
		const bool imm_avl = instr.vsetfunc() == 0x3; // VSETIVLI
		// vsetivli's vtype is ten bits. The two above it mark the form.
		const uint32_t vtypei = imm_avl ? (vi.IVLI.zimm & 0x3FF) : vi.VLI.zimm;
		const int rd  = imm_avl ? vi.IVLI.rd : vi.VLI.rd;
		const int rs1 = imm_avl ? 0 : int(vi.VLI.rs1);

		// Decode with the interpreter's own code, so the two can never
		// disagree about what is legal or how large VLMAX is.
		VectorRegisters<W> newtype;
		const bool legal = newtype.set_vtype(vtypei);
		const uint64_t vlmax = newtype.vlmax();

		this->reset_vector_config();
		if (!legal) {
			// Unsupported vtype: vl = 0 and vtype reads back as vill alone.
			// vsew/lmul keep their old values, as set_vtype leaves them.
			add_code("cpu->rvv.vta = " + std::to_string(newtype.vta()) + ";",
				"cpu->rvv.vma = " + std::to_string(newtype.vma()) + ";",
				"cpu->rvv.vill = 1;", "cpu->rvv.vl = 0;");
			if (rd != 0)
				add_code(to_reg(rd) + " = 0;");
			this->m_vtype = { true, true, 0, 0 };
			this->m_vl_known = true;
			this->m_vl = 0;
			return;
		}
		// A strip-mined loop sets the same vtype every iteration, and only
		// vl changes. The raw encoding decides every other field, so when it
		// is already in place, and not disowned by vill, the configuration
		// is a no-op worth branching over.
		add_code("if (UNLIKELY(cpu->rvv.vtype != " + std::to_string(vtypei)
			+ " || cpu->rvv.vill)) {",
			"  cpu->rvv.vill = 0;",
			"  cpu->rvv.vta = " + std::to_string(newtype.vta()) + ";",
			"  cpu->rvv.vma = " + std::to_string(newtype.vma()) + ";",
			"  cpu->rvv.vsew = " + std::to_string(newtype.encoded_sew()) + ";",
			"  cpu->rvv.lmul = " + std::to_string(newtype.lmul_shift()) + ";",
			"  cpu->rvv.vtype = " + std::to_string(vtypei) + ";",
			"}");

		// AVL: the immediate, x[rs1], VLMAX when rs1 is x0 but rd is not,
		// and otherwise the current vl clamped to the new VLMAX.
		const std::string vl = "cpu->rvv.vl";
		const std::string newvl = "vnewvl" + PCRELS(0);
		if (imm_avl || rs1 == 0) {
			const uint64_t avl = imm_avl ? vi.IVLI.uimm : (rd != 0 ? vlmax : 0);
			if (imm_avl || rd != 0) {
				add_code(vl + " = " + std::to_string(std::min(avl, vlmax)) + ";");
				this->m_vl_known = true;
				this->m_vl = std::min(avl, vlmax);
			} else {
				add_code("if (" + vl + " > " + std::to_string(vlmax) + ") "
					+ vl + " = " + std::to_string(vlmax) + ";");
			}
		} else {
			// Downstream reads vl from this local, not from the field.
			const std::string avl = "vavl" + PCRELS(0);
			this->load_register(rs1);
			add_code("addr_t " + avl + "; " + avl + " = " + from_untracked_reg(rs1) + ";",
				"addr_t " + newvl + "; " + newvl + " = (" + avl + " < "
				+ std::to_string(vlmax) + ") ? " + avl + " : "
				+ std::to_string(vlmax) + ";",
				vl + " = " + newvl + ";");
			this->m_vl_local = newvl;
		}
		this->m_vtype = { true, false, newtype.encoded_sew(), newtype.lmul_shift() };
		if (rd != 0) {
			this->reset_tracked_register(rd);
			if (this->m_vl_known)
				this->track_register_value(rd, this->m_vl);
			add_code(to_reg(rd) + " = " + (m_vl_local.empty() ? vl : m_vl_local) + ";");
		}
	}
	// Emit one vector instruction: inlined when the encoding allows it,
	// otherwise straight to the handler.
	void emit_vector_instruction()
	{
		// Settle this before emitting anything: it reads the vtype the block
		// has proven, which the emission below can change.
		this->m_vtrap_needed = this->vector_handler_can_trap(instr);

		if (this->vector_is_inlinable_setvl(instr)) {
			this->emit_vector_setvl();
			return;
		}
		// A load or store carries its own EEW and brings its own guard, so
		// it does not go through the SEW machinery the arithmetic needs.
		const auto meminfo = this->vector_memory_form(instr);
		if (meminfo.form != VectorMemForm::None) {
			this->emit_vector_memory(meminfo);
			return;
		}
		const auto sews = this->vector_inlinable_sews(instr);
		if (sews.empty()) {
			this->emit_vector_slowpath();
			return;
		}
		// Materialize every guard first: they are C locals, and the bodies
		// below open scopes a later declaration could not escape.
		for (const unsigned vsew : sews)
			(void)this->rvv_guard(vsew);

		for (const unsigned vsew : sews) {
			add_code("if (LIKELY(" + rvv_guard(vsew) + ")) {");
			this->emit_vector_float_arith_body(vsew);
			add_code("} else {");
		}
		this->emit_vector_slowpath();
		for (size_t i = 0; i < sews.size(); i++)
			add_code("}");
	}
#endif
	std::string from_imm(int64_t imm) {
		return std::to_string(imm);
	}
	void emit_op(const std::string& op, const std::string& sop,
		uint32_t rd, uint32_t rs1, const std::string& rs2)
	{
		if (rd == 0) {
			/* must be a NOP */
		} else if (rd == rs1) {
			add_code(to_reg(rd) + sop + rs2 + ";");
		} else {
		add_code(
			to_reg(rd) + " = " + from_reg(rs1) + op + rs2 + ";");
		}
	}

	void emit_branch(const BranchInfo& binfo, const std::string& op);

	void emit_system_call(std::string syscall_reg);

	// Returns true if the function call has exited/returned from the block
	bool emit_function_call(address_t target, address_t dest_pc);

	bool gpr_exists_at(int reg) const { return this->gpr_exists.at(reg); }
	bool gpr_written_at(int reg) const { return this->gpr_written.at(reg); }
	auto& get_gpr_exists() const noexcept { return this->gpr_exists; }
	bool gpr_needs_store(size_t reg) const {
		return this->gpr_exists_at(reg) && this->gpr_written_at(reg);
	}

	bool uses_flat_memory_arena() noexcept {
		return riscv::flat_readwrite_arena && tinfo.arena_ptr != 0;
	}
	bool uses_Nbit_encompassing_arena() noexcept {
		if (riscv::encompassing_Nbit_arena != 0 && tinfo.arena_ptr != 0)
			return true;
		if (tinfo.use_automatic_nbit_address_space && tinfo.arena_ptr != 0)
			return true;
		return false;
	}
	constexpr address_t get_Nbit_encompassing_arena_mask() noexcept {
		if constexpr (riscv::encompassing_Nbit_arena != 0)
			return riscv::encompassing_arena_mask;
		else if (tinfo.use_automatic_nbit_address_space)
			return this->m_encompassing_arena_mask;
		else
			return 0;
	}

	// The C type used to access guest memory: the may_alias twin of `type`,
	// see the ga_* typedefs in tr_api.cpp.
	static std::string guest_access_type(const std::string& type) {
		const std::string volatile_prefix = "volatile ";
		if (type.rfind(volatile_prefix, 0) == 0)
			return volatile_prefix + "ga_" + type.substr(volatile_prefix.size());
		return "ga_" + type;
	}

	std::string arena_at(const std::string& address) {
		// Direct arena pointer for libtcc (non-shared segments only).
		if (tinfo.is_libtcc && !tinfo.use_shared_execute_segments) {
			if (uses_Nbit_encompassing_arena()) {
				if (riscv::encompassing_Nbit_arena == 32)
					return "(" + m_arena_hex_address + " + (uint32_t)(" + address + "))";
				else
					return "(" + m_arena_hex_address + " + ((" + address + ") & " + hex_address(address_t(get_Nbit_encompassing_arena_mask())) + "))";
			} else {
				return "(" + m_arena_hex_address + " + " + speculation_safe(address) + ")";
			}
		} else if (uses_Nbit_encompassing_arena()) {
			if constexpr (riscv::encompassing_Nbit_arena == 32)
				return "ARENA_AT(cpu, (uint32_t)(" + address + "))";
			else
				return "ARENA_AT(cpu, (" + address + ") & " + hex_address(address_t(get_Nbit_encompassing_arena_mask())) + ")";
		} else {
			return "ARENA_AT(cpu, " + speculation_safe(address) + ")";
		}
	}

	std::string arena_at_fixed(const std::string& access_type, address_t address) {
		const std::string type = guest_access_type(access_type);
		if (tinfo.is_libtcc && !tinfo.use_shared_execute_segments) {
			if (uses_Nbit_encompassing_arena()) {
				return "*(" + type + "*)" + hex_address(tinfo.arena_ptr + (address & address_t(get_Nbit_encompassing_arena_mask()))) + "";
			} else {
				return "*(" + type + "*)" + hex_address(tinfo.arena_ptr + address) + "";
			}
		} else if (uses_Nbit_encompassing_arena()) {
			return "*(" + type + "*)ARENA_AT(cpu, " + hex_address(address & address_t(get_Nbit_encompassing_arena_mask())) + ")";
		} else {
			return "*(" + type + "*)ARENA_AT(cpu, " + speculation_safe(address) + ")";
		}
	}

	// A bounds check on `base + anchor` proves that address lies inside the arena.
	// The arena is over-allocated by OVERALLOCATE bytes on both ends, so any access
	// through the same base register at an offset within that window of the anchor
	// is also guaranteed to land in mapped memory, and needs no check of its own.
	static bool offset_is_within_overallocation(int64_t anchor, int64_t offset, size_t size) {
		if (UNLIKELY(size > Memory<W>::OVERALLOCATE))
			return false;
		return std::abs(offset - anchor) <= int64_t(Memory<W>::OVERALLOCATE - size);
	}

	// Window survives not-taken branches but dies at labels, calls, or base-register writes.
	void invalidate_bounds_checks(int reg) {
		if (reg > 0 && reg < 32) {
			this->m_read_checked[reg].valid = false;
			this->m_write_checked[reg].valid = false;
		}
	}
	void invalidate_all_bounds_checks() {
		for (auto& entry : this->m_read_checked) entry.valid = false;
		for (auto& entry : this->m_write_checked) entry.valid = false;
#ifdef RISCV_EXT_VECTOR
		this->reset_vector_config();
#endif
	}

	bool skip_load_bounds_check(int reg, int64_t offset, size_t size) {
		if (tinfo.unsafe_remove_checks
			|| uses_Nbit_encompassing_arena()) return true; // No bounds check
		if (tinfo.use_virtual_paging_fallback) return false; // Always check

		if (m_read_checked[reg].valid
			&& offset_is_within_overallocation(m_read_checked[reg].anchor, offset, size))
			return true;
		// Writable implies readable; inherit the write-check.
		if (m_write_checked[reg].valid
			&& offset_is_within_overallocation(m_write_checked[reg].anchor, offset, size))
			return true;

		m_read_checked[reg] = { true, offset };
		return false;
	}
	bool skip_store_bounds_check(int reg, int64_t offset, size_t size) {
		if (tinfo.unsafe_remove_checks
			|| uses_Nbit_encompassing_arena()) return true; // No bounds check
		if (tinfo.use_virtual_paging_fallback) return false; // Always check

		// NOTE: A live read check does *not* cover a write: the readable region
		// includes read-only data, which the writable region excludes.
		if (m_write_checked[reg].valid
			&& offset_is_within_overallocation(m_write_checked[reg].anchor, offset, size))
			return true;

		m_write_checked[reg] = { true, offset };
		return false;
	}

	template <typename T>
	std::string memory_load_type(const std::string& address)
	{
		if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
			return "rd8(cpu, " + address + ");";
		} else if (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) {
			return "rd16(cpu, " + address + ");";
		} else if (std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value) {
			return "rd32(cpu, " + address + ");";
		} else if (std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value) {
			return "rd64(cpu, " + address + ");";
		} else {
			throw MachineException(INVALID_PROGRAM, "Unsupported memory load type");
		}
	}
	template <typename T>
	void memory_load(std::string dst, std::string type, int reg, int32_t imm)
	{
		if (uses_flat_memory_arena()) {
			address_t absolute_vaddr = 0;
			if (reg == REG_GP && tinfo.gp != 0x0) {
				absolute_vaddr = tinfo.gp + imm;
			}
			constexpr bool good = riscv::encompassing_Nbit_arena != 0;
			if (absolute_vaddr != 0 && absolute_vaddr >= 0x1000 && (good || absolute_vaddr + sizeof(T) <= tinfo.arena_size)) {
				add_code(
					dst + " = " + arena_at_fixed(type, absolute_vaddr) + ";"
				);
				return;
			}
			if (auto tracked_value = get_tracked_register(reg)) {
				const address_t vaddr = *tracked_value + imm;
				if (vaddr >= 0x1000 && vaddr + sizeof(T) <= tinfo.arena_size) {
					add_code(dst + " = " + arena_at_fixed(type, vaddr) + ";");
					return;
				}
			}
		}

		const std::string address = from_untracked_reg(reg) + " + " + from_imm(imm);
		const std::string access = "*(" + guest_access_type(type) + "*)";
		if (skip_load_bounds_check(reg, imm, sizeof(T)))
		{
			add_code(dst + " = " + access + arena_at(address) + ";");
		}
		else if (uses_flat_memory_arena()) {
			add_code(
				"if (LIKELY(ARENA_READABLE(" + address + ")))",
					dst + " = " + access + arena_at(address) + ";",
				"else {");
			if (!tinfo.use_virtual_paging_fallback) {
				add_code("  cpu->pc = " + hex_address(this->pc()) + "LL; goto exception;",
						"}");
				return;
			}
			else if ((W == 8 && (type == "int64_t" || type == "uint64_t"))
				|| (W == 4 && (type == "int32_t" || type == "uint32_t"))) {
				add_code(
					dst + " = " + memory_load_type<T>(address) + ";",
				"}");
			} else {
				add_code(
					dst + " = (" + type + ")" + memory_load_type<T>(address) + ";",
				"}");
			}
		} else {
			add_code(
				dst + " = (" + type + ")" + memory_load_type<T>(address) + ";"
			);
		}
	}
	std::string memory_store_type(const std::string& type, const std::string& address, const std::string& value)
	{
		if (type == "int8_t" || type == "uint8_t") {
			return "wr8(cpu, " + address + ", " + value + ");";
		} else if (type == "int16_t" || type == "uint16_t") {
			return "wr16(cpu, " + address + ", " + value + ");";
		} else if (type == "int32_t" || type == "uint32_t") {
			return "wr32(cpu, " + address + ", " + value + ");";
		} else if (type == "int64_t" || type == "uint64_t") {
			return "wr64(cpu, " + address + ", " + value + ");";
		} else {
			throw MachineException(INVALID_PROGRAM, "Unsupported memory store type");
		}
	}
	// TCC truncates 64-bit addresses; materialize pointer in a function-scope scratch.
	void fixed_store(const std::string& type, address_t address, const std::string& value)
	{
		this->m_used_fixed_store = true;
		add_code("mstore = (char*)&" + arena_at_fixed(type, address) + ";",
			"*(" + guest_access_type(type) + "*)mstore = " + value + ";");
	}
	void memory_store(std::string type, size_t size, int reg, int32_t imm, std::string value)
	{
		if (uses_flat_memory_arena()) {
			address_t absolute_vaddr = 0;
			if (reg == REG_GP && tinfo.gp != 0x0) {
				absolute_vaddr = tinfo.gp + imm;
			}
			constexpr bool good = riscv::encompassing_Nbit_arena != 0;
			if (absolute_vaddr != 0 && absolute_vaddr >= tinfo.arena_roend && (good || absolute_vaddr < tinfo.arena_size)) {
				fixed_store(type, absolute_vaddr, value);
				return;
			}
			if (auto tracked_value = get_tracked_register(reg)) {
				const address_t vaddr = *tracked_value + imm;
				if (vaddr >= tinfo.arena_roend && vaddr <= tinfo.arena_size - 32) {
					fixed_store(type, vaddr, value);
					return;
				}
			}
		}

		const std::string address = from_untracked_reg(reg) + " + " + from_imm(imm);
		const std::string access = "*(" + guest_access_type(type) + "*)";
		if (skip_store_bounds_check(reg, imm, size))
		{
			add_code(access + arena_at(address) + " = " + value + ";");
		}
		else if (uses_flat_memory_arena()) {
			add_code(
				"if (LIKELY(ARENA_WRITABLE(" + address + ")))",
				"  " + access + arena_at(address) + " = " + value + ";",
				"else {");
			if (!tinfo.use_virtual_paging_fallback) {
				add_code("  cpu->pc = " + hex_address(this->pc()) + "LL; goto exception;",
						"}");
			} else {
				add_code("  " + memory_store_type(type, address, value) + ";",
						"}");
			}
		} else {
			add_code(
				memory_store_type(type, address, value)
			);
		}
	}

	bool no_labels_after_this() const noexcept {
		for (auto addr : labels)
			if (addr > this->pc())
				return false;
		for (auto addr : tinfo.jump_locations)
			if (addr > this->pc())
				return false;
		return true;
	}

	void add_mapping(address_t addr, std::string symbol) { this->mappings.push_back({addr, std::move(symbol)}); }
	auto& get_mappings() { return this->mappings; }

	bool add_reentry_next() {
		if (this->pc() + this->m_instr_length >= end_pc())
			return false;
		this->mapping_labels.insert(index() + 1);
		//code.append(FUNCLABEL(this->pc() + 4) + ":;\n");
		return true;
	}

	uint64_t reset_and_get_icounter() {
		auto result = this->m_instr_counter;
		this->m_instr_counter = 0;
		return result;
	}
	void increment_counter_so_far() {
		auto icount = this->reset_and_get_icounter();
		if (icount > 0 && !tinfo.ignore_instruction_limit)
			code.append("ic += " + std::to_string(icount) + ";\n");
	}
	void penalty(uint64_t cycles) {
		this->m_instr_counter += cycles;
	}

	bool block_exists(address_t pc) const noexcept {
		for (auto& blk : *tinfo.blocks) {
			if (blk.basepc == pc) return true;
		}
		return false;
	}
	uint64_t find_block_base(address_t pc) const noexcept {
		for (auto& blk : *tinfo.blocks) {
			if (pc >= blk.basepc && pc < blk.endpc) return blk.basepc;
		}
		return 0;
	}

	void add_forward(const std::string& target_func) {
		this->m_forward_declared.push_back(target_func);
	}
	const auto& get_forward_declared() const noexcept { return this->m_forward_declared; }

	size_t index() const noexcept { return this->m_idx; }
	address_t pc() const noexcept { return this->m_pc; }
	address_t begin_pc() const noexcept { return tinfo.basepc; }
	address_t end_pc() const noexcept { return tinfo.endpc; }

	bool within_segment(address_t addr) const noexcept {
		return addr >= this->tinfo.segment_basepc && addr < this->tinfo.segment_endpc;
	}
	bool used_fixed_store() const noexcept { return this->m_used_fixed_store; }

	const std::string get_func() const noexcept { return this->func; }
	void analyze_dirty_registers();
	void emit();
	rv32i_instruction emit_rvc();

private:
	static std::string speculation_safe(const std::string& address) {
		return "SPECSAFE(" + address + ")";
	}
	static std::string speculation_safe(const address_t address) {
		return "SPECSAFE(" + hex_address(address) + ")";
	}
	std::optional<address_t> get_tracked_register(int idx) const {
		if (idx < 0 || idx >= 32) {
			throw MachineException(INVALID_PROGRAM, "Invalid register index for tracking", idx);
		}
		if (this->m_is_tracked_register[idx]) {
			return this->m_tracked_registers[idx];
		}
		return std::nullopt;
	}
	void track_register_value(int idx, address_t value) {
		if (idx < 0 || idx >= 32) {
			throw MachineException(INVALID_PROGRAM, "Invalid register index for tracking", idx);
		}
		else if (idx > 0) {
			this->m_tracked_registers[idx] = value;
			this->m_is_tracked_register[idx] = true;
		}
	}
	void reset_tracked_register(int idx) {
		if (idx < 0 || idx >= 32) {
			throw MachineException(INVALID_PROGRAM, "Invalid register index for tracking", idx);
		}
		this->m_is_tracked_register[idx] = false;
		this->invalidate_bounds_checks(idx);
	}
	// Reset tracked constants only; bounds-check windows are handled separately.
	void reset_all_tracked_constants() {
		this->m_is_tracked_register.fill(false);
	}
	void reset_all_tracked_registers() {
		this->reset_all_tracked_constants();
		this->invalidate_all_bounds_checks();
	}

	std::string code;
	size_t m_idx = 0;
	address_t m_pc = 0x0;
	address_t m_last_pc = 0x0;
	rv32i_instruction instr;
	unsigned m_instr_length = 0;
	uint64_t m_instr_counter = 0;
	uint32_t m_zero_insn_counter = 0;
#ifdef RISCV_EXT_VECTOR
	// vtype as proven by an inlined vsetvli earlier in this block. Dropped
	// where control flow joins, or where a handler may reconfigure it.
	struct KnownVtype {
		bool     known = false;
		bool     vill  = false;
		unsigned vsew  = 0;  // log2(SEW / 8)
		int      lmul  = 0;  // log2(LMUL)
	};
	KnownVtype m_vtype;
	// vl when the vsetvli that produced it had a compile-time AVL
	bool     m_vl_known = false;
	uint64_t m_vl = 0;
	// C local holding the live vl, when an inlined vsetvli computed one
	std::string m_vl_local;
	// Live C local holding the vl/vtype fast-path test per SEW, see rvv_guard()
	std::array<std::string, 4> m_rvv_guard;
	// Whether the current vector instruction's handler can raise at all
	bool m_vtrap_needed = true;
#endif
	address_t m_encompassing_arena_mask = 0;
	bool m_used_fixed_store = false;
	bool m_used_vector_trap = false;

	// Per-register live bounds-check window (see offset_is_within_overallocation)
	struct BoundsCheck {
		bool    valid  = false;
		int64_t anchor = 0;
	};
	std::array<BoundsCheck, 32> m_read_checked {};
	std::array<BoundsCheck, 32> m_write_checked {};

	std::array<bool, 32> gpr_exists {};
	std::array<bool, 32> gpr_written {};
	std::vector<uint32_t> m_dirty_at;  // dirty when the instruction executes (what a sync here stores)
	std::vector<uint32_t> m_dirty_out; // dirty after it, for the flow
	std::set<uint32_t> m_store_masks;
	std::array<bool, 32> m_is_tracked_register {};
	std::array<address_t, 32> m_tracked_registers {};

	std::string func;
	const TransInfo<W>& tinfo;
	std::string m_arena_hex_address;

	std::vector<TransMapping<W>> mappings;
	std::unordered_set<unsigned> labels;
	std::unordered_set<unsigned> mapping_labels;
	std::unordered_set<address_t> pagedata;

	std::vector<std::string> m_forward_declared;
};

template <int W>
inline void Emitter<W>::emit_branch(const BranchInfo& binfo, const std::string& op)
{
	using address_t = address_type<W>;
	if (binfo.sign == false)
		code += "if (" + from_reg(instr.Btype.rs1) + op + from_reg(instr.Btype.rs2) + ")";
	else
		code += "if ((saddr_t)" + from_reg(instr.Btype.rs1) + op + " (saddr_t)" + from_reg(instr.Btype.rs2) + ")";

	if (UNLIKELY(PCRELA(instr.Btype.signed_imm()) & ALIGN_MASK))
	{
		// TODO: Make exception a helper function, as return values are implementation detail
		code += "\n  { api.exception(cpu, " + PCRELS(0) + ", MISALIGNED_INSTRUCTION); RETURN_VALUES(0, 0); }\n";
		return;
	}

	if (binfo.jump_pc != 0) {
		if (binfo.jump_pc > this->pc() || binfo.ignore_instruction_limit) {
			// unconditional forward jump + bracket
			code += " goto " + FUNCLABEL(binfo.jump_pc) + ";\n";
			return;
		}
		// backward jump
		code += " {\nif (" + LOOP_EXPRESSION + ") goto " + FUNCLABEL(binfo.jump_pc) + ";\n";
	} else if (binfo.call_pc != 0 && binfo.call_pc > this->pc()) {
		code += " {\n";
		// potentially call a function
		auto target_funcaddr = this->find_block_base(binfo.call_pc);
		// Allow directly calling a function, as long as it's a forward branch
		if (target_funcaddr != 0) {
			emit_function_call(target_funcaddr, binfo.call_pc);
			code += "}\n"; // Bracket (NOTE: not ending the function, just the branch)
			return;
		}
	} else {
		code += " {\n";
	}
	// else, exit binary translation
	exit_function(PCRELS(instr.Btype.signed_imm()), true); // Bracket (NOTE: not actually ending the function)
}

template <int W>
inline bool Emitter<W>::emit_function_call(address_t target_funcaddr, address_t dest_pc)
{
	// The callee may write any register
	this->invalidate_all_bounds_checks();
	// Store the registers
	this->store_loaded_registers();

	auto target_func = funclabel<W>("f", target_funcaddr);
	add_forward(target_func);
	if (!tinfo.ignore_instruction_limit) {
		// Call the function and get the return values
		add_code("retvals = " + target_func + "(cpu, ic, max_ic, " + STRADDR(dest_pc) + ");");
		// Update the local counter registers
		add_code("ic = retvals.ic; max_ic = retvals.max_ic;");
	} else {
		add_code("retvals = " + target_func + "(cpu, 0, max_ic, " + STRADDR(dest_pc) + ");");
		add_code("max_ic = retvals.max_ic;");
	}

	// Restore the registers
	this->reload_all_registers();

	if (tinfo.trace_instructions) {
		code += "api.trace(cpu, \"" + this->func + "\", cpu->pc, max_ic);\n";
	}

	// Hope and pray that the next PC is local to this block
	if (!tinfo.ignore_instruction_limit) {
		add_code("if (" + LOOP_EXPRESSION + ") { pc = cpu->pc; goto " + this->func + "_jumptbl; }");
		add_code("RETURN_VALUES(ic, max_ic);");
	} else {
		add_code("if (max_ic) { pc = cpu->pc; goto " + this->func + "_jumptbl; }");
		add_code("RETURN_VALUES(0, 0);");
	}
	return true;
}

// Linux system calls that hand control to another thread, which comes back
// with that thread's register file restored in place. clone(2) is the one that
// also leaves the PC where it was, so nothing downstream can notice the swap:
// the child resumes at the very instruction that made the call. The rest are
// here because two threads can be parked at the same system call - a spin loop
// that yields, or a futex wait - and then the PC does not move either.
static bool syscall_switches_thread(uint64_t sysno) noexcept
{
	switch (sysno) {
	case 93:  // exit
	case 94:  // exit_group, installed as the exit handler, see setup_posix_threads()
	case 98:  // futex
	case 124: // sched_yield
	case 131: // tgkill
	case 139: // rt_sigreturn
	case 220: // clone
	case 422: // futex_time64
	case 435: // clone3
		return true;
	default:
		return false;
	}
}

template <int W>
inline void Emitter<W>::emit_system_call(std::string syscall_reg)
{
	// A7 is loaded with an immediate right before the ECALL in almost all code,
	// so the system call number is usually known here.
	std::optional<uint64_t> sysno;
	if (syscall_reg == std::to_string(SYSCALL_EBREAK)) {
		sysno = SYSCALL_EBREAK;
	} else if (auto tracked_value = get_tracked_register(REG_ECALL); tracked_value) {
		if (*tracked_value < address_t(1) << 20) {
			sysno = uint64_t(*tracked_value);
			if constexpr (W != 16) { // No 128-bit to_string in C++
				syscall_reg = std::to_string(*tracked_value);
			}
		}
	}
	const bool switches_thread = sysno && syscall_switches_thread(*sysno);
	this->store_loaded_registers();
	if (tinfo.is_libtcc)
	{
		if (!tinfo.ignore_instruction_limit) {
			code += "max_ic = api.system_call(cpu, " + PCRELS(0) + ", ic, max_ic, " + syscall_reg + ");\n";
			code += "ic = INS_COUNTER(cpu);\n";
		} else {
			code += "max_ic = api.system_call(cpu, " + PCRELS(0) + ", 0, max_ic, " + syscall_reg + ");\n";
		}
		code += "if (!max_ic) {\n";
		if (!tinfo.ignore_instruction_limit) {
			code += "  RETURN_VALUES(ic, MAX_COUNTER(cpu));\n"
					"}\n";
		} else {
			code += "  RETURN_VALUES(0, MAX_COUNTER(cpu));\n"
					"}\n";
		}
	}
	else
	{
		code += "cpu->pc = " + PCRELS(0) + ";\n";
		if (!tinfo.ignore_instruction_limit) {
			code += "if (UNLIKELY(do_syscall(cpu, ic, max_ic, " + syscall_reg + "))) {\n";
			code += "  cpu->pc += 4; RETURN_VALUES(ic, MAX_COUNTER(cpu));}\n"; // Correct for +4 expectation outside of bintr
			code += "max_ic = MAX_COUNTER(cpu);\n"; // Restore max counter
		} else {
			code += "if (UNLIKELY(do_syscall(cpu, 0, max_ic, " + syscall_reg + "))) {\n";
			code += "  cpu->pc += 4; RETURN_VALUES(0, MAX_COUNTER(cpu));}\n";
		}
	}
	this->reset_all_tracked_registers();
#ifdef RISCV_EXT_VECTOR
	this->reset_vector_config();
#endif
	if (switches_thread) {
		// The handler swapped in another thread's register file. Exit without
		// storing the cached registers, which belong to the thread that made
		// the call; re-entering the block loads the new thread's registers.
		const char* counter = (tinfo.ignore_instruction_limit) ? "0" : "ic";
		add_code("cpu->pc = " + PCRELS(4) + ";",
			std::string("RETURN_VALUES(") + counter + ", MAX_COUNTER(cpu));");
		// Whatever follows is only reachable through the dispatch table now
		this->add_reentry_next();
		return;
	}
	this->reload_syscall_registers();
}

#ifdef RISCV_EXT_C
#include "tr_emit_rvc.cpp"
#endif

template <int W>
void Emitter<W>::analyze_dirty_registers()
{
	const auto& instrs = tinfo.instr;
	const size_t n = instrs.size();
	m_dirty_at.assign(n, 0);
	m_dirty_out.assign(n, 0);
	if (!uses_register_caching() || n == 0)
		return;

	enum Kind { SIMPLE, BRANCH, JUMP, INDIRECT, SYSCALL, OTHER };
	struct Node {
		uint32_t writes = 0;
		Kind kind = SIMPLE;
		int  target = -1; // instruction index, for BRANCH and JUMP inside the block
	};
	std::vector<Node> nodes(n);
	std::vector<address_t> addr_of(n);
	std::unordered_map<address_t, int> index_of;
	address_t pc = tinfo.basepc;
	for (size_t i = 0; i < n; i++) {
		addr_of[i] = pc;
		index_of.emplace(pc, int(i));
		pc += compressed_enabled ? instrs[i].length() : 4;
	}
	const auto lookup = [&] (address_t target) -> int {
		auto it = index_of.find(target);
		return (it != index_of.end()) ? it->second : -1;
	};
	const auto reg_bit = [] (unsigned reg) -> uint32_t {
		return (reg != 0) ? (1u << reg) : 0u;
	};

	for (size_t i = 0; i < n; i++) {
		const auto& instr = instrs[i];
		Node& node = nodes[i];
		if (compressed_enabled && instr.is_compressed()) {
#ifdef RISCV_EXT_C
			const rv32c_instruction ci { instr };
			// Every rd field of every compressed format, plus x1 for the links
			node.writes = reg_bit(instr.whole >> 7 & 0x1F)
				| reg_bit(8 + (instr.whole >> 2 & 0x7))
				| reg_bit(8 + (instr.whole >> 7 & 0x7)) | reg_bit(1);
			const unsigned quadrant = instr.whole & 0x3;
			const unsigned funct3 = instr.whole >> 13 & 0x7;
			if (quadrant == 0b01 && (funct3 == 0b101 || (W == 4 && funct3 == 0b001))) {
				node.kind = JUMP; // C.J, C.JAL
				node.target = lookup(addr_of[i] + ci.CJ.signed_imm());
			} else if (quadrant == 0b01 && (funct3 == 0b110 || funct3 == 0b111)) {
				node.kind = BRANCH; // C.BEQZ, C.BNEZ
				node.target = lookup(addr_of[i] + ci.CB.signed_imm());
			} else if (quadrant == 0b10 && funct3 == 0b100 && ci.CR.rs2 == 0) {
				// C.JR, C.JALR, C.EBREAK
				node.kind = (ci.CR.rd != 0) ? INDIRECT : SYSCALL;
			}
#endif
			continue;
		}
		node.writes = reg_bit(instr.Itype.rd);
		switch (instr.opcode()) {
		case RV32I_LOAD: case RV32I_STORE: case RV32I_OP_IMM: case RV32I_OP:
		case RV32I_LUI: case RV32I_AUIPC: case RV64I_OP_IMM32: case RV64I_OP32:
		case RV32F_LOAD: case RV32F_STORE: case RV32F_FMADD: case RV32F_FMSUB:
		case RV32F_FNMSUB: case RV32F_FNMADD: case RV32F_FPFUNC: case RV32A_ATOMIC:
			break;
		case RV32I_BRANCH:
			node.kind = BRANCH;
			node.target = lookup(addr_of[i] + instr.Btype.signed_imm());
			break;
		case RV32I_JAL:
			node.kind = JUMP;
			node.target = lookup((addr_of[i] + instr.Jtype.jump_offset()) & ~address_t(ALIGN_MASK));
			break;
		case RV32I_JALR:
			node.kind = INDIRECT;
			break;
		case RV32I_SYSTEM:
			// ECALL and EBREAK store everything dirty, and reload a0/a1
			node.kind = (instr.Itype.funct3 == 0 && instr.Itype.imm < 2) ? SYSCALL : OTHER;
			break;
		case Dyncall::opcode:
			// Count fields are not register operands. Keep unsaved dirty
			// registers flowing through the call, including around backedges.
			node.writes = Dyncall::valid(instr.whole)
				? Dyncall::arg_mask(Dyncall::outputs(instr.whole)) : ~0u;
			node.kind = OTHER;
			break;
		default:
			node.kind = OTHER;
			break;
		}
	}

	// Which instructions can be entered from the dispatch switch, or through a
	// label at all: the block entry, every jump target, whatever follows an
	// instruction that is not straight-line, and code after a run of zeroes.
	std::vector<bool> is_label(n, false);
	is_label[0] = true;
	for (const auto addr : tinfo.jump_locations)
		if (const int idx = lookup(addr); idx >= 0) is_label[idx] = true;
	for (const auto addr : tinfo.global_jump_locations)
		if (const int idx = lookup(addr); idx >= 0) is_label[idx] = true;
	unsigned zeroes = 0;
	for (size_t i = 0; i < n; i++) {
		if (nodes[i].kind != SIMPLE && i + 1 < n)
			is_label[i + 1] = true;
		if (instrs[i].is_illegal())
			zeroes++;
		else {
			if (zeroes >= 4) is_label[i] = true;
			zeroes = 0;
		}
	}

	// Fixpoint over the may-dirty sets. The dispatch switch is one node:
	// its input is the union over every indirect jump, and it flows into
	// every label.
	std::vector<uint32_t> in(n, 0);
	uint32_t dispatch = 0;
	bool changed = true;
	while (changed) {
		changed = false;
		uint32_t new_dispatch = 0;
		for (size_t i = 0; i < n; i++) {
			const Node& node = nodes[i];
			uint32_t d = in[i];
			if (is_label[i]) d |= dispatch;
			if (i > 0 && nodes[i - 1].kind != JUMP && nodes[i - 1].kind != INDIRECT)
				d |= m_dirty_out[i - 1]; // fall-through
			if (d != in[i]) { in[i] = d; changed = true; }
			// A sync at this instruction stores what is dirty on entry plus
			// whatever the instruction itself wrote before it. A system call
			// syncs everything, so nothing is dirty after it.
			m_dirty_at[i] = d | node.writes;
			const uint32_t out = (node.kind == SYSCALL) ? 0u : m_dirty_at[i];
			if (out != m_dirty_out[i]) { m_dirty_out[i] = out; changed = true; }
			if ((node.kind == BRANCH || node.kind == JUMP) && node.target >= 0) {
				if ((in[node.target] | out) != in[node.target]) {
					in[node.target] |= out; changed = true;
				}
			}
			if (node.kind == INDIRECT)
				new_dispatch |= out;
		}
		if (new_dispatch != dispatch) { dispatch = new_dispatch; changed = true; }
	}
}

template <int W>
void Emitter<W>::emit()
{
	this->analyze_dirty_registers();
	this->add_mapping(this->pc(), this->func);
	code.append(FUNCLABEL(this->pc()) + ":;\n");
	auto next_pc = tinfo.basepc;
	address_t current_callable_pc = 0;
	this->m_pc = tinfo.basepc;
	this->m_last_pc = tinfo.basepc;

	for (int i = 0; i < int(tinfo.instr.size()); i++) {
		this->m_idx = i;
		this->instr = tinfo.instr[i];
		this->m_last_pc = this->m_pc;
		this->m_pc = next_pc;
		if constexpr (compressed_enabled)
			this->m_instr_length = this->instr.length();
		else
			this->m_instr_length = 4;
		next_pc = this->m_pc + this->m_instr_length;

		if (this->instr.is_illegal()) {
			this->m_zero_insn_counter ++;
		} else {
			if (this->m_zero_insn_counter >= 4) {
				// After a ream of zero instructions, we predict a jump target
				mapping_labels.insert(i);
			}
			this->m_zero_insn_counter = 0;
		}

		// If the address is a return address or a global JAL target
		if (i > 0 && (mapping_labels.count(i) || tinfo.global_jump_locations.count(this->pc()))) {
			this->increment_counter_so_far();
			// Re-entry through the current function
			code.append(FUNCLABEL(this->pc()) + ":;\n");
			this->mappings.push_back({
				this->pc(), this->func
			});
			// Since someone can jump here, we need to forget all tracked register values
			this->reset_all_tracked_registers();
		}
		// known jump locations
		else if (i > 0 && tinfo.jump_locations.count(this->pc())) {
			this->increment_counter_so_far();
			code.append(FUNCLABEL(this->pc()) + ":;\n");
			// Since someone can jump here, we need to forget all tracked register values
			this->reset_all_tracked_registers();
		}

		// Rare: jump target at PC+2 inside a 4-byte instruction. Emit a skip-over trap.
		if (UNLIKELY(compressed_enabled && this->m_instr_length == 4 && tinfo.jump_locations.count(this->pc() + 2))) {
			code.append("goto " + FUNCLABEL(this->pc() + 2) + "_skip;\n");
			code.append(FUNCLABEL(this->pc() + 2) + ":;\n");
			code.append("api.exception(cpu, " + STRADDR(this->pc() + 2) + ", MISALIGNED_INSTRUCTION); RETURN_VALUES(0, 0);\n");
			code.append(FUNCLABEL(this->pc() + 2) + "_skip:;\n");
			this->reset_all_tracked_registers();
		}

		if (tinfo.return_locations.count(this->pc())) {
			// A called function starts here: its returns can be predicted
			// from the call sites, see RV32I_JALR below.
			current_callable_pc = this->pc();
		}

		this->m_instr_counter += 1;

		if (tinfo.trace_instructions) {
			char buffer[128];
			const int len = snprintf(buffer, sizeof(buffer),
				"api.trace(cpu, \"%s\", 0x%" PRIx64 ", 0x%X);\n",
				this->func.c_str(), uint64_t(this->pc()), instr.whole);
			code.append(buffer, len);
		}

		if (tinfo.ebreak_locations->count(this->pc())) {
			this->emit_system_call(std::to_string(SYSCALL_EBREAK));
		}

		// instruction generation
#ifdef RISCV_EXT_C
		if (instr.is_compressed()) {
			// Compressed 16-bit instructions
			auto original = instr.whole;
			instr = this->emit_rvc();

			if (instr.is_compressed())
			{
				const uint16_t compressed_instr = instr.half[0];
				// Unexpanded instruction (except all-zeroes, which is illegal)
				if (tinfo.trace_instructions && compressed_instr != 0x0)
					printf("Unexpanded instruction: 0x%04hx at PC 0x%lX (original 0x%x)\n", compressed_instr, long(this->pc()), original);
				// When illegal opcode is encountered, reveal PC
				if (m_zero_insn_counter <= 1 || compressed_instr != 0x0) {
					code += "api.exception(cpu, " + STRADDR(this->pc()) + ", ILLEGAL_OPCODE);\n";
					if (tinfo.is_libtcc) {
						code += "RETURN_VALUES(0, 0);\n";
					}
				}
				this->reset_all_tracked_registers();
				continue;
			}
		}
#endif

		switch (instr.opcode()) {
		case RV32I_LOAD:
			load_register(instr.Itype.rs1);
			if (instr.Itype.rd != 0) {
			switch (instr.Itype.funct3) {
			case 0x0: // I8
				this->memory_load<int8_t>(to_reg(instr.Itype.rd), "int8_t", instr.Itype.rs1, instr.Itype.signed_imm());
				break;
			case 0x1: // I16
				this->memory_load<int16_t>(to_reg(instr.Itype.rd), "int16_t", instr.Itype.rs1, instr.Itype.signed_imm());
				break;
			case 0x2: // I32
				this->memory_load<int32_t>(to_reg(instr.Itype.rd), "int32_t", instr.Itype.rs1, instr.Itype.signed_imm());
				break;
			case 0x3: // I64
				this->memory_load<int64_t>(to_reg(instr.Itype.rd), "int64_t", instr.Itype.rs1, instr.Itype.signed_imm());
				break;
			case 0x4: // U8
				this->memory_load<uint8_t>(to_reg(instr.Itype.rd), "uint8_t", instr.Itype.rs1, instr.Itype.signed_imm());
				break;
			case 0x5: // U16
				this->memory_load<uint16_t>(to_reg(instr.Itype.rd), "uint16_t", instr.Itype.rs1, instr.Itype.signed_imm());
				break;
			case 0x6: // U32
				this->memory_load<uint32_t>(to_reg(instr.Itype.rd), "uint32_t", instr.Itype.rs1, instr.Itype.signed_imm());
				break;
			default:
				UNKNOWN_INSTRUCTION();
			}
			this->reset_tracked_register(instr.Itype.rd);
			} else {
				// We don't care about where we are in the page when rd=0
				const auto temp = "tmp" + PCRELS(0);
				add_code("uint8_t " + temp + ";");
				this->memory_load<uint8_t>(temp, "volatile uint8_t", instr.Itype.rs1, instr.Itype.signed_imm());
				add_code("(void)" + temp + ";");
			} break;
		case RV32I_STORE:
			load_register(instr.Stype.rs1);
			switch (instr.Stype.funct3) {
			case 0x0: // I8
				this->memory_store("int8_t", sizeof(int8_t), instr.Stype.rs1, instr.Stype.signed_imm(), from_reg(instr.Stype.rs2));
				break;
			case 0x1: // I16
				this->memory_store("int16_t", sizeof(int16_t), instr.Stype.rs1, instr.Stype.signed_imm(), from_reg(instr.Stype.rs2));
				break;
			case 0x2: // I32
				this->memory_store("int32_t", sizeof(int32_t), instr.Stype.rs1, instr.Stype.signed_imm(), from_reg(instr.Stype.rs2));
				break;
			case 0x3: // I64
				this->memory_store("int64_t", sizeof(int64_t), instr.Stype.rs1, instr.Stype.signed_imm(), from_reg(instr.Stype.rs2));
				break;
			default:
				UNKNOWN_INSTRUCTION();
			}
			break;
		case RV32I_BRANCH: {
			this->increment_counter_so_far();
			load_register(instr.Btype.rs1);
			load_register(instr.Btype.rs2);
			const auto offset = instr.Btype.signed_imm();
			uint64_t dest_pc = this->pc() + offset;
			uint64_t jump_pc = 0;
			uint64_t call_pc = 0;
			// goto branch: restarts function
			if (dest_pc == this->begin_pc()) {
				// restart function
				jump_pc = dest_pc;
			}
			// forward label: branch inside code block
			else if (offset > 0 && dest_pc < this->end_pc()) {
				// forward label: future address
				labels.insert(dest_pc);
				jump_pc = dest_pc;
			} else if (tinfo.jump_locations.count(dest_pc)) {
				// existing jump location
				if (dest_pc >= this->begin_pc() && dest_pc < this->end_pc()) {
					jump_pc = dest_pc;
				}
			} else if (tinfo.global_jump_locations.count(dest_pc) && this->within_segment(dest_pc)) {
				// global jump location
				call_pc = dest_pc;
			}
			switch (instr.Btype.funct3) {
			case 0x0: // EQ
				emit_branch({ false, tinfo.ignore_instruction_limit, jump_pc, call_pc }, " == ");
				break;
			case 0x1: // NE
				emit_branch({ false, tinfo.ignore_instruction_limit, jump_pc, call_pc }, " != ");
				break;
			case 0x2:
			case 0x3:
				UNKNOWN_INSTRUCTION();
				break;
			case 0x4: // LT
				emit_branch({ true, tinfo.ignore_instruction_limit, jump_pc, call_pc }, " < ");
				break;
			case 0x5: // GE
				emit_branch({ true, tinfo.ignore_instruction_limit, jump_pc, call_pc }, " >= ");
				break;
			case 0x6: // LTU
				emit_branch({ false, tinfo.ignore_instruction_limit, jump_pc, call_pc }, " < ");
				break;
			case 0x7: // GEU
				emit_branch({ false, tinfo.ignore_instruction_limit, jump_pc, call_pc }, " >= ");
				break;
			}
			// Not-taken path: bounds-check windows survive (no register writes).
			this->reset_all_tracked_constants(); // For now
			} break;
		case RV32I_JALR: {
			// jump to register + immediate
			this->reset_all_tracked_registers(); // For now
			this->increment_counter_so_far();
			if (instr.Itype.rd != 0 && instr.Itype.rd == instr.Itype.rs1) {
				// NOTE: We need to remember RS1 because it is clobbered by RD
				const auto src = from_reg(instr.Itype.rs1);
				const auto dst = to_reg(instr.Itype.rd);
				add_code(
					"JUMP_TO(" + src + " + " + from_imm(instr.Itype.signed_imm()) + ");",
					dst + " = " + PCRELS(m_instr_length) + ";"
				);
			} else if (instr.Itype.rd != 0) {
				add_code(
					to_reg(instr.Itype.rd) + " = " + PCRELS(m_instr_length) + ";",
					"JUMP_TO(" + from_reg(instr.Itype.rs1) + " + " + from_imm(instr.Itype.signed_imm()) + ");"
				);
			} else {
				// A plain return from a function with known call sites: compare
				// the return address against the sites in this block and jump
				// directly, which is always correct whichever function we are
				// really in, and skips the indirect dispatch when it hits.
				std::vector<address_t> predicted;
				if (instr.Itype.rs1 != 0 && instr.Itype.signed_imm() == 0 && current_callable_pc != 0) {
					auto it = tinfo.return_locations.find(current_callable_pc);
					if (it != tinfo.return_locations.end()) {
						for (const address_t site : it->second) {
							if (site > this->begin_pc() && site < this->end_pc()
								&& std::find(predicted.begin(), predicted.end(), site) == predicted.end())
								predicted.push_back(site);
						}
						if (predicted.size() > MAX_PREDICTED_RETURN_SITES)
							predicted.clear();
					}
				}
				// In counting mode the whole dispatch is guarded by the limit
				// below, so a predicted (possibly backward) jump is also counted.
				if (!predicted.empty() && !tinfo.ignore_instruction_limit)
					add_code("if (" + LOOP_EXPRESSION + ") {");
				for (const address_t site : predicted) {
					add_code("if (" + from_reg(instr.Itype.rs1) + " == " + STRADDR(site) + ") goto " + FUNCLABEL(site) + ";");
				}
				if (!predicted.empty() && !tinfo.ignore_instruction_limit)
					add_code("}");
				add_code(
					"JUMP_TO(" + from_reg(instr.Itype.rs1) + " + " + from_imm(instr.Itype.signed_imm()) + ");"
				);
			}
			// Untrack current callable PC
			current_callable_pc = 0;
			// The dispatch table covers the whole block, and exits the
			// function for any address outside of it.
			if (!tinfo.ignore_instruction_limit) {
				add_code("if (" + LOOP_EXPRESSION + ") goto " + this->func + "_jumptbl;");
				exit_function("pc", false);
			} else {
				add_code("goto " + this->func + "_jumptbl;");
			}
			this->add_reentry_next();
			} break;
		case RV32I_JAL: {
			this->reset_all_tracked_registers(); // For now
			this->increment_counter_so_far();
			if (instr.Jtype.rd != 0) {
				add_code(to_reg(instr.Jtype.rd) + " = " + PCRELS(m_instr_length) + ";\n");
			}
			// XXX: mask off unaligned jumps - is this OK?
			const auto dest_pc = (this->pc() + instr.Jtype.jump_offset()) & ~address_t(ALIGN_MASK);
			bool add_reentry = instr.Jtype.rd != 0;
			bool already_exited = false;
			// forward label: jump inside code block
			if (dest_pc >= this->begin_pc() && dest_pc < this->end_pc()) {
				// forward labels require creating future labels
				if (dest_pc > this->pc()) {
					labels.insert(dest_pc);
					add_code("goto " + FUNCLABEL(dest_pc) + ";");
					already_exited = true; // Unconditional jump
				} else if (tinfo.ignore_instruction_limit) {
					// jump backwards: without counters
					add_code("goto " + FUNCLABEL(dest_pc) + ";");
					// Random jumps around often have useful code immediately after,
					// so make sure it's accessible (add a re-entry point)
					// TODO: Check if the next instruction is a public symbol address
					if (instr.Jtype.rd == 0)
						add_reentry = true;
					already_exited = true; // Unconditional jump
				} else {
					// jump backwards: use counters
					add_code("if (" + LOOP_EXPRESSION + ") goto " + FUNCLABEL(dest_pc) + ";");
					// Random jumps around often have useful code immediately after,
					// so make sure it's accessible (add a re-entry point)
					// TODO: Check if the next instruction is a public symbol address
					if (instr.Jtype.rd == 0)
						add_reentry = true;
				}
				// .. if we run out of instructions, we must jump manually and exit:
			}
			else if (this->tinfo.global_jump_locations.count(dest_pc) && this->within_segment(dest_pc)) {
				// Get the function name of the target block
				auto target_funcaddr = this->find_block_base(dest_pc);
				// Allow directly calling a function, as long as it's a forward jump
				/// XXX: This forward call is buggy, and crashes on Windows with LIBTCC
				/// Don't enable until it is fixed (or well understood)
				if (false && target_funcaddr != 0 && dest_pc > this->pc()) {
					//printf("Jump location OK (forward): 0x%lX for block 0x%lX -> 0x%lX\n", long(dest_pc),
					//	long(this->begin_pc()), long(this->end_pc()));
					already_exited = this->emit_function_call(target_funcaddr, dest_pc);

					if (!already_exited)
						exit_function("cpu->pc", false);
					already_exited = true;
				} else {
					//printf("Jump location inconvenient (backward): 0x%lX at func 0x%lX for block 0x%lX -> 0x%lX\n",
					//	long(dest_pc), long(target_funcaddr), long(this->begin_pc()), long(this->end_pc()));
				}
			}

			// Because of forward jumps we can't end the function here
			if (!already_exited)
				exit_function(STRADDR(dest_pc), false);
			if (add_reentry)
				this->add_reentry_next();
			} break;

		case RV32I_OP_IMM: {
			// NOP: Instruction without side-effect
			if (UNLIKELY(instr.Itype.rd == 0)) break;

			const auto dst = to_reg(instr.Itype.rd);
			std::string src = from_reg(instr.Itype.rs1);

			switch (instr.Itype.funct3) {
			case 0x0: // ADDI
				if (instr.Itype.signed_imm() == 0) {
					add_code(dst + " = " + src + ";");
				} else if (instr.Itype.rs1 == 0) {
					add_code(dst + " = " + from_imm(instr.Itype.signed_imm()) + ";");
				} else {
					emit_op(" + ", " += ", instr.Itype.rd, instr.Itype.rs1, from_imm(instr.Itype.signed_imm()));
				}
				break;
			case 0x1: // SLLI
				// SLLI: Logical left-shift 5/6-bit immediate
				switch (instr.Itype.imm) {
				case 0b011000000100: // SEXT.B
					add_code(
						dst + " = (int8_t)" + src + ";");
					break;
				case 0b011000000101: // SEXT.H
					add_code(
						dst + " = (int16_t)" + src + ";");
					break;
				case 0b011000000000: // CLZ
					if constexpr (W == 4)
						add_code(
							dst + " = " + src + " ? do_clz(" + src + ") : XLEN;");
					else
						add_code(
							dst + " = " + src + " ? do_clzl(" + src + ") : XLEN;");
					break;
				case 0b011000000001: // CTZ
					if constexpr (W == 4)
						add_code(
							dst + " = " + src + " ? do_ctz(" + src + ") : XLEN;");
					else
						add_code(
							dst + " = " + src + " ? do_ctzl(" + src + ") : XLEN;");
					break;
				case 0b011000000010: // CPOP
					if constexpr (W == 4)
						add_code(
							dst + " = do_cpop(" + src + ");");
					else
						add_code(
							dst + " = do_cpopl(" + src + ");");
					break;
				default:
					if (instr.Itype.high_bits() == 0 && (W != 4 || (instr.Itype.imm & 0x20) == 0)) {
						// SLLI: Logical left-shift immediate
						emit_op(" << ", " <<= ", instr.Itype.rd, instr.Itype.rs1,
							std::to_string(instr.Itype.shift64_imm() & (XLEN-1)));
					} else if (instr.Itype.high_bits() == 0x280 && (W != 4 || (instr.Itype.imm & 0x20) == 0)) {
						// BSETI: Bit-set immediate
						add_code(dst + " = " + src + " | ((addr_t)1 << (" + std::to_string(instr.Itype.imm & (XLEN-1)) + "));");
					}
					else if (instr.Itype.high_bits() == 0x480 && (W != 4 || (instr.Itype.imm & 0x20) == 0)) {
						// BCLRI: Bit-clear immediate
						add_code(dst + " = " + src + " & ~((addr_t)1 << (" + std::to_string(instr.Itype.imm & (XLEN-1)) + "));");
					}
					else if (instr.Itype.high_bits() == 0x680 && (W != 4 || (instr.Itype.imm & 0x20) == 0)) {
						// BINVI: Bit-invert immediate
						add_code(dst + " = " + src + " ^ ((addr_t)1 << (" + std::to_string(instr.Itype.imm & (XLEN-1)) + "));");
					}
					else if (W == 4 && instr.Itype.imm == 0b000010001111) {
						add_code(dst + " = do_zip32(" + src + ");");
					} else {
						UNKNOWN_INSTRUCTION();
					}
				}
				break;
			case 0x2: // SLTI
				// SLTI: Set less than immediate
				add_code(
					dst + " = ((saddr_t)" + src + " < " + from_imm(instr.Itype.signed_imm()) + ") ? 1 : 0;");
				break;
			case 0x3: // SLTU:
				add_code(
					dst + " = (" + src + " < (addr_t) " + from_imm(instr.Itype.signed_imm()) + ") ? 1 : 0;");
				break;
			case 0x4: // XORI:
				emit_op(" ^ ", " ^= ", instr.Itype.rd, instr.Itype.rs1, from_imm(instr.Itype.signed_imm()));
				break;
			case 0x5: // SRLI / SRAI / ORC.B:
				if (instr.Itype.is_rori() && (W != 4 || (instr.Itype.imm & 0x20) == 0)) {
					// RORI: Rotate right immediate
					add_code(
					"{const unsigned shift = " + from_imm(instr.Itype.imm & (XLEN-1)) + ";\n",
						dst + " = (" + src + " >> shift) | (" + src + " << ((XLEN - shift) & (XLEN-1))); }"
					);
				} else if (instr.Itype.imm == 0x287) {
					// ORC.B: Bitwise OR-combine
					add_code(
						"for (unsigned i = 0; i < sizeof(addr_t); i++)",
						"	((int8_t *)&" + dst + ")[i] = ((int8_t *)&" + src + ")[i] ? 0xFF : 0x0;"
					);
				} else if (instr.Itype.is_rev8<sizeof(dst)>()) {
					// REV8: Byte-reverse register
					if constexpr (W == 4)
						add_code(dst + " = do_bswap32(" + src + ");");
					else
						add_code(dst + " = do_bswap64(" + src + ");");
				} else if (instr.Itype.high_bits() == 0x0 && (W != 4 || (instr.Itype.imm & 0x20) == 0)) {
					// SRLI: Logical right-shift immediate
					emit_op(" >> ", " >>= ", instr.Itype.rd, instr.Itype.rs1,
						std::to_string(instr.Itype.shift64_imm() & (XLEN-1)));
				} else if (instr.Itype.high_bits() == 0x400 && (W != 4 || (instr.Itype.imm & 0x20) == 0)) {
					// SRAI: Arithmetic right-shift immediate
					add_code(
						dst + " = (saddr_t)" + src + " >> " + std::to_string(instr.Itype.imm & (XLEN-1)) + ";");
				} else if (instr.Itype.high_bits() == 0x480 && (W != 4 || (instr.Itype.imm & 0x20) == 0)) { // BEXTI: Bit-extract immediate
					add_code(
						dst + " = (" + src + " >> (" + std::to_string(instr.Itype.imm & (XLEN-1)) + ")) & 1;");
				} else if (instr.Itype.imm == 0b011010000111) {
					add_code(dst + " = do_brev8(" + src + ");");
				} else if (W == 4 && instr.Itype.imm == 0b000010001111) {
					add_code(dst + " = do_unzip32(" + src + ");");
				} else {
					UNKNOWN_INSTRUCTION();
				}
				break;
			case 0x6: // ORI
				add_code(
					dst + " = " + src + " | " + from_imm(instr.Itype.signed_imm()) + ";");
				break;
			case 0x7: // ANDI
				add_code(
					dst + " = " + src + " & " + from_imm(instr.Itype.signed_imm()) + ";");
				break;
			default:
				UNKNOWN_INSTRUCTION();
			}
			// Register tracking (mostly ADDI)
			if (instr.Itype.funct3 == 0) {
				// Track register value when rs1 == 0:
				if (instr.Itype.rs1 == 0) {
					this->track_register_value(instr.Itype.rd, instr.Itype.signed_imm());
				} else {
					if (auto tracked_value = get_tracked_register(instr.Itype.rs1)) {
						this->track_register_value(instr.Itype.rd, instr.Itype.signed_imm() + *tracked_value);
					} else {
						this->reset_tracked_register(instr.Itype.rd);
					}
				}
			} else {
				this->reset_tracked_register(instr.Itype.rd);
			}
			} break;
		case RV32I_OP:
			if (UNLIKELY(instr.Rtype.rd == 0)) break;

			switch (instr.Rtype.jumptable_friendly_op()) {
			case 0x0: // ADD
				if (instr.Rtype.rs2 == instr.Rtype.rd) {
					// Make sure we can perform rd += rs1
					emit_op(" + ", " += ", instr.Rtype.rd, instr.Rtype.rs2, from_reg(instr.Rtype.rs1));
				} else {
					emit_op(" + ", " += ", instr.Rtype.rd, instr.Rtype.rs1, from_reg(instr.Rtype.rs2));
				}
				break;
			case 0x200: // SUB
				emit_op(" - ", " -= ", instr.Rtype.rd, instr.Rtype.rs1, from_reg(instr.Rtype.rs2));
				break;
			case 0x1: // SLL
				add_code(
					to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + " << (" + from_reg(instr.Rtype.rs2) + " & (XLEN-1));");
				break;
			case 0x2: // SLT
				add_code(
					to_reg(instr.Rtype.rd) + " = ((saddr_t)" + from_reg(instr.Rtype.rs1) + " < (saddr_t)" + from_reg(instr.Rtype.rs2) + ") ? 1 : 0;");
				break;
			case 0x3: // SLTU
				add_code(
					to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs1) + " < " + from_reg(instr.Rtype.rs2) + ") ? 1 : 0;");
				break;
			case 0x4: // XOR
				emit_op(" ^ ", " ^= ", instr.Rtype.rd, instr.Rtype.rs1, from_reg(instr.Rtype.rs2));
				break;
			case 0x5: // SRL
				add_code(
					to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + " >> (" + from_reg(instr.Rtype.rs2) + " & (XLEN-1));");
				break;
			case 0x205: // SRA
				add_code(
					to_reg(instr.Rtype.rd) + " = (saddr_t)" + from_reg(instr.Rtype.rs1) + " >> (" + from_reg(instr.Rtype.rs2) + " & (XLEN-1));");
				break;
			case 0x6: // OR
				emit_op(" | ", " |= ", instr.Rtype.rd, instr.Rtype.rs1, from_reg(instr.Rtype.rs2));
				break;
			case 0x7: // AND
				emit_op(" & ", " &= ", instr.Rtype.rd, instr.Rtype.rs1, from_reg(instr.Rtype.rs2));
				break;
			// extension RV32M / RV64M
			case 0x10: // MUL
				add_code(
					to_reg(instr.Rtype.rd) + " = (addr_t)" + from_reg(instr.Rtype.rs1) + " * (addr_t)" + from_reg(instr.Rtype.rs2) + ";");
				break;
			case 0x11: // MULH (signed x signed)
				if constexpr (W == 4) {
					add_code(
						to_reg(instr.Rtype.rd) + " = (uint64_t)((int64_t)(saddr_t)" + from_reg(instr.Rtype.rs1) + " * (int64_t)(saddr_t)" + from_reg(instr.Rtype.rs2) + ") >> 32u;");
				} else {
					add_code(
						"{ addr_t lhs = " + from_reg(instr.Rtype.rs1) + "; addr_t rhs = " + from_reg(instr.Rtype.rs2) + ";",
						"MUL128(&" + to_reg(instr.Rtype.rd) + ", lhs, rhs);",
						"if ((saddr_t)lhs < 0) " + to_reg(instr.Rtype.rd) + " -= rhs;",
						"if ((saddr_t)rhs < 0) " + to_reg(instr.Rtype.rd) + " -= lhs; }"
					);
				}
				break;
			case 0x12: // MULHSU (signed x unsigned)
				if constexpr (W == 4) {
					add_code(
						to_reg(instr.Rtype.rd) + " = (uint64_t)((int64_t)(saddr_t)" + from_reg(instr.Rtype.rs1) + " * (uint64_t)" + from_reg(instr.Rtype.rs2) + ") >> 32u;");
				} else {
					add_code(
						"{ addr_t lhs = " + from_reg(instr.Rtype.rs1) + "; addr_t rhs = " + from_reg(instr.Rtype.rs2) + ";",
						"MUL128(&" + to_reg(instr.Rtype.rd) + ", lhs, rhs);",
						"if ((saddr_t)lhs < 0) " + to_reg(instr.Rtype.rd) + " -= rhs; }"
					);
				}
				break;
			case 0x13: // MULHU (unsigned x unsigned)
				add_code(
					(W == 4) ?
					to_reg(instr.Rtype.rd) + " = ((uint64_t) " + from_reg(instr.Rtype.rs1) + " * (uint64_t)" + from_reg(instr.Rtype.rs2) + ") >> 32u;" :
					"MUL128(&" + to_reg(instr.Rtype.rd) + ", " + from_reg(instr.Rtype.rs1) + ", " + from_reg(instr.Rtype.rs2) + ");"
				);
				break;
			case 0x14: // DIV
				// Division by zero is not an exception: rd = -1
				// Signed overflow is not an exception either: rd = the dividend
				if constexpr (W == 8) {
					add_code(
						"if (LIKELY(" + from_reg(instr.Rtype.rs2) + " != 0)) {",
						"	if (LIKELY(!(" + from_reg(instr.Rtype.rs1) + " == -9223372036854775808ull && " + from_reg(instr.Rtype.rs2) + " == -1ull)))",
						"		" + to_reg(instr.Rtype.rd) + " = (int64_t)" + from_reg(instr.Rtype.rs1) + " / (int64_t)" + from_reg(instr.Rtype.rs2) + ";",
						"	else " + to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + ";",
						"} else " + to_reg(instr.Rtype.rd) + " = (addr_t)-1;");
				} else {
					add_code(
						"if (LIKELY(" + from_reg(instr.Rtype.rs2) + " != 0)) {",
						"	if (LIKELY(!(" + from_reg(instr.Rtype.rs1) + " == 2147483648 && " + from_reg(instr.Rtype.rs2) + " == 4294967295)))",
						"		" + to_reg(instr.Rtype.rd) + " = (int32_t)" + from_reg(instr.Rtype.rs1) + " / (int32_t)" + from_reg(instr.Rtype.rs2) + ";",
						"	else " + to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + ";",
						"} else " + to_reg(instr.Rtype.rd) + " = (addr_t)-1;");
				}
				break;
			case 0x15: // DIVU
				add_code(
					"if (LIKELY(" + from_reg(instr.Rtype.rs2) + " != 0))",
					to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + " / " + from_reg(instr.Rtype.rs2) + ";",
					"else " + to_reg(instr.Rtype.rd) + " = (addr_t)-1;"
				);
				break;
			case 0x16: // REM
				// Division by zero is not an exception: rd = the dividend
				// Signed overflow is not an exception either: rd = 0
				if constexpr (W == 8) {
					add_code(
					"if (LIKELY(" + from_reg(instr.Rtype.rs2) + " != 0)) {",
					"	if (LIKELY(!(" + from_reg(instr.Rtype.rs1) + " == -9223372036854775808ull && " + from_reg(instr.Rtype.rs2) + " == -1ull)))",
					"		" + to_reg(instr.Rtype.rd) + " = (int64_t)" + from_reg(instr.Rtype.rs1) + " % (int64_t)" + from_reg(instr.Rtype.rs2) + ";",
					"	else " + to_reg(instr.Rtype.rd) + " = 0;",
					"} else " + to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + ";");
				} else {
					add_code(
					"if (LIKELY(" + from_reg(instr.Rtype.rs2) + " != 0)) {",
					"	if (LIKELY(!(" + from_reg(instr.Rtype.rs1) + " == 2147483648 && " + from_reg(instr.Rtype.rs2) + " == 4294967295)))",
					"		" + to_reg(instr.Rtype.rd) + " = (int32_t)" + from_reg(instr.Rtype.rs1) + " % (int32_t)" + from_reg(instr.Rtype.rs2) + ";",
					"	else " + to_reg(instr.Rtype.rd) + " = 0;",
					"} else " + to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + ";");
				}
				break;
			case 0x17: // REMU
				add_code(
				"if (LIKELY(" + from_reg(instr.Rtype.rs2) + " != 0))",
					to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + " % " + from_reg(instr.Rtype.rs2) + ";",
					"else " + to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + ";"
				);
				break;
			case 0x44: // PACK (ZEXT.H on RV32 when rs2 == 0)
				if constexpr (W == 4) {
					add_code(to_reg(instr.Rtype.rd) + " = (addr_t)(uint16_t)(" + from_reg(instr.Rtype.rs1) + ") | ((addr_t)(uint16_t)(" + from_reg(instr.Rtype.rs2) + ") << 16);");
				} else {
					add_code(to_reg(instr.Rtype.rd) + " = (addr_t)(uint32_t)(" + from_reg(instr.Rtype.rs1) + ") | ((addr_t)(uint32_t)(" + from_reg(instr.Rtype.rs2) + ") << 32);");
				}
				break;
			case 0x47: // PACKH
				add_code(to_reg(instr.Rtype.rd) + " = (addr_t)(uint8_t)(" + from_reg(instr.Rtype.rs1) + ") | ((addr_t)(uint8_t)(" + from_reg(instr.Rtype.rs2) + ") << 8);");
				break;
			case 0x51: // CLMUL
				add_code(
					"{ addr_t result = 0;",
					"for (unsigned i = 0; i < XLEN; i++)",
					"  if ((" + from_reg(instr.Rtype.rs2) + " >> i) & 1)",
					"    result ^= (" + from_reg(instr.Rtype.rs1) + " << i);",
					to_reg(instr.Rtype.rd) + " = result; }");
				break;
			case 0x52: // CLMULR
				add_code(
					"{ addr_t result = 0;",
					"for (unsigned i = 0; i < XLEN; i++)",
					"  if ((" + from_reg(instr.Rtype.rs2) + " >> i) & 1)",
					"    result ^= (" + from_reg(instr.Rtype.rs1) + " >> (XLEN - i - 1));",
					to_reg(instr.Rtype.rd) + " = result; }");
				break;
			case 0x53: // CLMULH
				add_code(
					"{ addr_t result = 0;",
					"for (unsigned i = 1; i < XLEN; i++)",
					"  if ((" + from_reg(instr.Rtype.rs2) + " >> i) & 1)",
					"    result ^= (" + from_reg(instr.Rtype.rs1) + " >> (XLEN - i));",
					to_reg(instr.Rtype.rd) + " = result; }");
				break;
			case 0x102: // SH1ADD
				add_code(to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs2) + " + (" + from_reg(instr.Rtype.rs1) + " << 1);");
				break;
			case 0x104: // SH2ADD
				add_code(to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs2) + " + (" + from_reg(instr.Rtype.rs1) + " << 2);");
				break;
			case 0x106: // SH3ADD
				add_code(to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs2) + " + (" + from_reg(instr.Rtype.rs1) + " << 3);");
				break;
			case 0x141: // BSET
				add_code(to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + " | ((addr_t)1 << (" + from_reg(instr.Rtype.rs2) + " & (XLEN-1)));");
				break;
			case 0x204: // XNOR
				add_code(to_reg(instr.Rtype.rd) + " = ~(" + from_reg(instr.Rtype.rs1) + " ^ " + from_reg(instr.Rtype.rs2) + ");");
				break;
			case 0x206: // ORN
				add_code(to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs1) + " | ~" + from_reg(instr.Rtype.rs2) + ");");
				break;
			case 0x207: // ANDN
				add_code(to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs1) + " & ~" + from_reg(instr.Rtype.rs2) + ");");
				break;
			case 0x241: // BCLR
				add_code(to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + " & ~((addr_t)1 << (" + from_reg(instr.Rtype.rs2) + " & (XLEN-1)));");
				break;
			case 0x245: // BEXT
				add_code(to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs1) + " >> (" + from_reg(instr.Rtype.rs2) + " & (XLEN-1))) & 1;");
				break;
			case 0x54: // MIN
				add_code(to_reg(instr.Rtype.rd) + " = ((saddr_t)" + from_reg(instr.Rtype.rs1) + " < (saddr_t)" + from_reg(instr.Rtype.rs2) + ") "
					" ? " + from_reg(instr.Rtype.rs1) + " : " + from_reg(instr.Rtype.rs2) + ";");
				break;
			case 0x55: // MINU
				add_code(to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs1) + " < " + from_reg(instr.Rtype.rs2) + ") "
					" ? " + from_reg(instr.Rtype.rs1) + " : " + from_reg(instr.Rtype.rs2) + ";");
				break;
			case 0x56: // MAX
				add_code(to_reg(instr.Rtype.rd) + " = ((saddr_t)" + from_reg(instr.Rtype.rs1) + " > (saddr_t)" + from_reg(instr.Rtype.rs2) + ") "
					" ? " + from_reg(instr.Rtype.rs1) + " : " + from_reg(instr.Rtype.rs2) + ";");
				break;
			case 0x57: // MAXU
				add_code(to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs1) + " > " + from_reg(instr.Rtype.rs2) + ") "
					" ? " + from_reg(instr.Rtype.rs1) + " : " + from_reg(instr.Rtype.rs2) + ";");
				break;
			case 0x75: // CZERO.EQZ
				// dst = (src2 == 0) ? 0 : src1;
				add_code(to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs2) + " == 0) ? 0 : " + from_reg(instr.Rtype.rs1) + ";");
				break;
			case 0x77: // CZERO.NEZ
				// dst = (src2 != 0) ? 0 : src1;
				add_code(to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs2) + " != 0) ? 0 : " + from_reg(instr.Rtype.rs1) + ";");
				break;
			// The complementary shift count is masked, as a zero rotate would
			// otherwise shift by the full register width
			case 0x301: // ROL: Rotate left
				add_code(
				"{const unsigned shift = " + from_reg(instr.Rtype.rs2) + " & (XLEN-1);\n",
					to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs1) + " << shift) | (" + from_reg(instr.Rtype.rs1) + " >> ((XLEN - shift) & (XLEN-1))); }"
				);
				break;
			case 0x305: // ROR: Rotate right
				add_code(
				"{const unsigned shift = " + from_reg(instr.Rtype.rs2) + " & (XLEN-1);\n",
					to_reg(instr.Rtype.rd) + " = (" + from_reg(instr.Rtype.rs1) + " >> shift) | (" + from_reg(instr.Rtype.rs1) + " << ((XLEN - shift) & (XLEN-1))); }"
				);
				break;
			case 0x341: // BINV
				add_code(to_reg(instr.Rtype.rd) + " = " + from_reg(instr.Rtype.rs1) + " ^ ((addr_t)1 << (" + from_reg(instr.Rtype.rs2) + " & (XLEN-1)));");
				break;
			default:
				//fprintf(stderr, "RV32I_OP: Unhandled function 0x%X\n",
				//		instr.Rtype.jumptable_friendly_op());
				UNKNOWN_INSTRUCTION();
			}
			this->reset_tracked_register(instr.Rtype.rd);
			break;
		case RV32I_LUI:
			if (UNLIKELY(instr.Utype.rd == 0))
				break;
			add_code(
				to_reg(instr.Utype.rd) + " = " + from_imm(instr.Utype.upper_imm()) + ";");
			this->track_register_value(instr.Utype.rd, instr.Utype.upper_imm());
			break;
		case RV32I_AUIPC:
			if (UNLIKELY(instr.Utype.rd == 0))
				break;
			add_code(
				to_reg(instr.Utype.rd) + " = " + PCRELS(instr.Utype.upper_imm()) + ";");
			this->track_register_value(instr.Utype.rd, PCRELA(instr.Utype.upper_imm()));
			break;
		case RV32I_FENCE:
			if (instr.Itype.funct3 == 0x1) {
				WELL_KNOWN_INSTRUCTION();
				exit_function(PCRELS(4), false);
				this->add_reentry_next();
			} else if (instr.Itype.funct3 != 0x0) {
				UNKNOWN_INSTRUCTION();
			}
			break;
		case RV32I_SYSTEM:
			if (instr.Itype.funct3 == 0x0) {
				this->increment_counter_so_far();
				// System calls and EBREAK
				if (instr.Itype.imm < 2) {
					std::string syscall_reg;
					if (instr.Itype.imm == 0) {
						// ECALL: System call
						syscall_reg = this->from_reg(REG_ECALL);
						this->emit_system_call(syscall_reg);
					} else { // EBREAK
						syscall_reg = std::to_string(SYSCALL_EBREAK);
						this->emit_system_call(syscall_reg);
					}
					break;
				} else if (instr.Itype.imm == 261 || instr.Itype.imm == 0x7FF) { // WFI / STOP
					code += "max_ic = 0;\n"; // Immediate stop PC + 4
					exit_function(PCRELS(4), false);
					this->add_reentry_next();
					break;
				} else {
					this->load_register(instr.Itype.rd);
					this->potentially_realize_register(instr.Itype.rd);
					this->load_register(instr.Itype.rs1);
					this->potentially_realize_register(instr.Itype.rs1);
					// Zero funct3, unknown imm: Don't exit
					// The system handler is opaque and may write any register
					this->invalidate_all_bounds_checks();
					code += "cpu->pc = " + PCRELS(0) + ";\n";
					if (tinfo.is_libtcc) {
						code += "if (api.system(cpu, " + std::to_string(instr.whole) +"))\n";
						code += "  RETURN_VALUES(0, 0);\n";
					} else {
						code += "api.system(cpu, " + std::to_string(instr.whole) +");\n";
					}
					this->reset_tracked_register(instr.Itype.rd);
					this->potentially_reload_register(instr.Itype.rd);
					break;
				}
			} else {
				// Non-zero funct3: CSR and other system functions
				this->load_register(instr.Itype.rd);
				this->potentially_realize_register(instr.Itype.rd);
				this->load_register(instr.Itype.rs1);
				this->potentially_realize_register(instr.Itype.rs1);
				// The system handler is opaque and may write any register
				this->invalidate_all_bounds_checks();
				code += "cpu->pc = " + PCRELS(0) + ";\n";
				if (!tinfo.ignore_instruction_limit)
					code += "INS_COUNTER(cpu) = ic;\n"; // Reveal instruction counters
				code += "MAX_COUNTER(cpu) = max_ic;\n";
				if (tinfo.is_libtcc) {
					code += "if (api.system(cpu, " + std::to_string(instr.whole) +"))\n";
					code += "  RETURN_VALUES(0, 0);\n";
				} else {
					code += "api.system(cpu, " + std::to_string(instr.whole) +");\n";
				}
				this->reset_tracked_register(instr.Itype.rd);
				this->potentially_reload_register(instr.Itype.rd);
			} break;
		case RV64I_OP_IMM32: {
			if constexpr (W < 8) {
				UNKNOWN_INSTRUCTION();
				break;
			}
			if (UNLIKELY(instr.Itype.rd == 0))
				break;
			const auto dst = to_reg(instr.Itype.rd);
			const auto src = "(uint32_t)" + from_reg(instr.Itype.rs1);
			if (instr.Itype.funct3 == 0x0) {
				// Track register value when rs1 == 0:
				if (instr.Itype.rs1 == 0) {
					this->track_register_value(instr.Itype.rd, (int32_t)instr.Itype.signed_imm());
				} else {
					this->reset_tracked_register(instr.Itype.rd);
				}
			} else {
				this->reset_tracked_register(instr.Itype.rd);
			}
			switch (instr.Itype.funct3) {
			case 0x0:
				// ADDIW: Add sign-extended 12-bit immediate
				add_code(dst + " = " + SIGNEXTW + " (" + src + " + " + from_imm(instr.Itype.signed_imm()) + ");");
				break;
			case 0x1: // SLLI.W / SLLI.UW:
				if (instr.Itype.high_bits() == 0x000 && (instr.Itype.imm & 0x20) == 0) {
					add_code(dst + " = " + SIGNEXTW + " (" + src + " << " + from_imm(instr.Itype.shift_imm()) + ");");
				} else if (instr.Itype.high_bits() == 0x080) {
					// SLLI.UW (full 6-bit RV64 shamt)
					add_code(dst + " = ((addr_t)" + src + " << " + from_imm(instr.Itype.shift64_imm()) + ");");
				} else {
					switch (instr.Itype.imm) {
					case 0b011000000000: // CLZ.W
						add_code(dst + " = " + src + " ? do_clz(" + src + ") : 32;");
						break;
					case 0b011000000001: // CTZ.W
						add_code(dst + " = " + src + " ? do_ctz(" + src + ") : 32;");
						break;
					case 0b011000000010: // CPOP.W
						add_code(dst + " = do_cpop(" + src + ");");
						break;
					default:
						UNKNOWN_INSTRUCTION();
					}
				}
				break;
			case 0x5: // SRLIW / SRAIW:
				if (instr.Itype.high_bits() == 0x0 && (instr.Itype.imm & 0x20) == 0) { // SRLIW
					add_code(dst + " = " + SIGNEXTW + " (" + src + " >> " + from_imm(instr.Itype.shift_imm()) + ");");
				} else if (instr.Itype.high_bits() == 0x400 && (instr.Itype.imm & 0x20) == 0) { // SRAIW: preserve the sign bit
					add_code(
						dst + " = (int32_t)" + src + " >> " + from_imm(instr.Itype.shift_imm()) + ";");
				} else if (instr.Itype.high_bits() == 0x600 && (instr.Itype.imm & 0x20) == 0) { // RORIW
					add_code(
						"{const unsigned shift = " + from_imm(instr.Itype.imm) + " & 31;\n",
						"const uint32_t word = " + src + ";\n",
						dst + " = (int32_t)((word >> shift) | (word << ((32 - shift) & 31))); }"
					);
				} else {
					UNKNOWN_INSTRUCTION();
				}
				break;
			default:
				UNKNOWN_INSTRUCTION();
			}
			} break;
		case RV64I_OP32: {
			if constexpr (W < 8) {
				UNKNOWN_INSTRUCTION();
				break;
			}
			if (UNLIKELY(instr.Rtype.rd == 0))
				break;
			const auto dst = to_reg(instr.Rtype.rd);
			const auto src1 = "(uint32_t)" + from_reg(instr.Rtype.rs1);
			const auto src2 = "(uint32_t)" + from_reg(instr.Rtype.rs2);

			switch (instr.Rtype.jumptable_friendly_op()) {
			case 0x0: // ADDW
				add_code(dst + " = " + SIGNEXTW + " (" + src1 + " + " + src2 + ");");
				break;
			case 0x200: // SUBW
				add_code(dst + " = " + SIGNEXTW + " (" + src1 + " - " + src2 + ");");
				break;
			case 0x1: // SLLW
				add_code(dst + " = " + SIGNEXTW + " (" + src1 + " << (" + src2 + " & 0x1F));");
				break;
			case 0x5: // SRLW
				add_code(dst + " = " + SIGNEXTW + " (" + src1 + " >> (" + src2 + " & 0x1F));");
				break;
			case 0x205: // SRAW
				add_code(dst + " = (int32_t)" + src1 + " >> (" + src2 + " & 31);");
				break;
			// M-extension
			case 0x10: // MULW
				add_code(dst + " = " + SIGNEXTW + "(" + src1 + " * " + src2 + ");");
				break;
			case 0x14: // DIVW
				// Division by zero is not an exception: rd = -1
				// Signed overflow is not an exception either: rd = the dividend
				add_code(
				"if (LIKELY(" + src2 + " != 0)) {",
				"	if (LIKELY(!((int32_t)" + src1 + " == -2147483648 && (int32_t)" + src2 + " == -1)))",
				"		" + dst + " = " + SIGNEXTW + " ((int32_t)" + src1 + " / (int32_t)" + src2 + ");",
				"	else " + dst + " = " + SIGNEXTW + " (" + src1 + ");",
				"} else " + dst + " = (addr_t)(int32_t)-1;");
				break;
			case 0x15: // DIVUW
				add_code(
				"if (LIKELY(" + src2 + " != 0))",
				dst + " = " + SIGNEXTW + " (" + src1 + " / " + src2 + ");",
				"else " + dst + " = (addr_t)(int32_t)-1;");
				break;
			case 0x16: // REMW
				// Division by zero is not an exception: rd = the dividend
				// Signed overflow is not an exception either: rd = 0
				add_code(
				"if (LIKELY(" + src2 + " != 0)) {",
				"	if (LIKELY(!((int32_t)" + src1 + " == -2147483648 && (int32_t)" + src2 + " == -1)))",
				"		" + dst + " = " + SIGNEXTW + " ((int32_t)" + src1 + " % (int32_t)" + src2 + ");",
				"	else " + dst + " = 0;",
				"} else " + dst + " = " + SIGNEXTW + " (" + src1 + ");");
				break;
			case 0x17: // REMUW
				add_code(
				"if (LIKELY(" + src2 + " != 0))",
				dst + " = " + SIGNEXTW + " (" + src1 + " % " + src2 + ");",
				"else " + dst + " = " + SIGNEXTW + " (" + src1 + ");");
				break;
			case 0x40: // ADD.UW
				add_code(dst + " = " + from_reg(instr.Rtype.rs2) + " + " + src1 + ";");
				break;
			case 0x44: // ZEXT.H / PACKW
				if (instr.Rtype.rs2 == 0) {
					add_code(dst + " = (uint16_t)(" + src1 + ");");
				} else {
					add_code(dst + " = (int32_t)((uint16_t)(" + src1 + ") | ((uint32_t)(uint16_t)(" + src2 + ") << 16));");
				}
				break;
			case 0x102: // SH1ADD.UW
				add_code(dst + " = " + from_reg(instr.Rtype.rs2) + " + ((addr_t)" + src1 + " << 1);");
				break;
			case 0x104: // SH2ADD.UW
				add_code(dst + " = " + from_reg(instr.Rtype.rs2) + " + ((addr_t)" + src1 + " << 2);");
				break;
			case 0x106: // SH3ADD.UW
				add_code(dst + " = " + from_reg(instr.Rtype.rs2) + " + ((addr_t)" + src1 + " << 3);");
				break;
			case 0x301: // ROLW: Rotate left 32-bit
				add_code(
				"{const unsigned shift = " + from_reg(instr.Rtype.rs2) + " & 31;\n",
					"const uint32_t word = (uint32_t)" + from_reg(instr.Rtype.rs1) + ";\n",
					dst + " = (int32_t)((word << shift) | (word >> ((32 - shift) & 31))); }"
				);
				break;
			case 0x305: // RORW: Rotate right (32-bit)
				add_code(
				"{const unsigned shift = " + from_reg(instr.Rtype.rs2) + " & 31;\n",
					"const uint32_t word = (uint32_t)" + from_reg(instr.Rtype.rs1) + ";\n",
					dst + " = (int32_t)((word >> shift) | (word << ((32 - shift) & 31))); }"
				);
				break;
			default:
				UNKNOWN_INSTRUCTION();
			}
			this->reset_tracked_register(instr.Rtype.rd);
			} break;
		case RV32F_LOAD: {
			const rv32f_instruction fi{instr};
			switch (fi.Itype.funct3) {
			case 0x1: // FLH (Zfhmin), boxed at sixteen bits
				this->memory_load<uint16_t>(from_fpreg(fi.Itype.rd) + ".i32[0]", "uint16_t", fi.Itype.rs1, fi.Itype.signed_imm());
				if constexpr (nanboxing) {
					code += from_fpreg(fi.Itype.rd) + ".i64 |= (int64_t)0xFFFFFFFFFFFF0000ULL;\n";
				}
				break;
			case 0x2: // FLW
				this->memory_load<uint32_t>(from_fpreg(fi.Itype.rd) + ".i32[0]", "uint32_t", fi.Itype.rs1, fi.Itype.signed_imm());
				if constexpr (nanboxing) {
					code += from_fpreg(fi.Itype.rd) + ".i32[1] = ~0;\n";
				}
				break;
			case 0x3: // FLD
				this->memory_load<uint64_t>(from_fpreg(fi.Itype.rd) + ".i64", "uint64_t", fi.Itype.rs1, fi.Itype.signed_imm());
				break;
#ifdef RISCV_EXT_VECTOR
		// RVV loads: funct3 encodes EEW, non-overlapping with scalar FP widths.
		case 0x0: case 0x5: case 0x6: case 0x7:
			this->emit_vector_instruction();
			break;
#endif
			default:
				UNKNOWN_INSTRUCTION();
				break;
			}
			} break;
		case RV32F_STORE: {
			const rv32f_instruction fi{instr};
			switch (fi.Itype.funct3) {
			case 0x1: // FSH (Zfhmin)
				this->memory_store("int16_t", sizeof(int16_t), fi.Stype.rs1, fi.Stype.signed_imm(), from_fpreg(fi.Stype.rs2) + ".i32[0]");
				break;
			case 0x2: // FSW
				this->memory_store("int32_t", sizeof(int32_t), fi.Stype.rs1, fi.Stype.signed_imm(), from_fpreg(fi.Stype.rs2) + ".i32[0]");
				break;
			case 0x3: // FSD
				this->memory_store("int64_t", sizeof(int64_t), fi.Stype.rs1, fi.Stype.signed_imm(), from_fpreg(fi.Stype.rs2) + ".i64");
				break;
#ifdef RISCV_EXT_VECTOR
		// RVV stores
		case 0x0: case 0x5: case 0x6: case 0x7:
			this->emit_vector_instruction();
			break;
#endif
			default:
				UNKNOWN_INSTRUCTION();
				break;
			}
			} break;
		case RV32F_FMADD:
		case RV32F_FMSUB:
		case RV32F_FNMADD:
		case RV32F_FNMSUB: {
			// FMA: single rounding via std::fma; TCC would otherwise split into two roundings.
			//   FMADD  =  rs1*rs2 + rs3 →  fma(rs1, rs2,  rs3)
			//   FMSUB  =  rs1*rs2 - rs3 →  fma(rs1, rs2, -rs3)
			//   FNMADD = -rs1*rs2 - rs3 → -fma(rs1, rs2,  rs3)
			//   FNMSUB = -rs1*rs2 + rs3 → -fma(rs1, rs2, -rs3)
			const rv32f_instruction fi{instr};
			const auto dst = from_fpreg(fi.R4type.rd);
			const auto rs1 = from_fpreg(fi.R4type.rs1);
			const auto rs2 = from_fpreg(fi.R4type.rs2);
			const auto rs3 = from_fpreg(fi.R4type.rs3);
			const bool negateResult = (instr.opcode() == RV32F_FNMADD || instr.opcode() == RV32F_FNMSUB);
			const bool subtractC    = (instr.opcode() == RV32F_FMSUB  || instr.opcode() == RV32F_FNMSUB);
			const std::string resultSign = negateResult ? "-" : "";
			const std::string cSign      = subtractC    ? "-" : "";
			if (fi.R4type.funct2 == 0x0) { // float32
				if constexpr (nanboxing && W == 8) {
					code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu"
						" || (uint32_t)" + rs3 + ".i32[1] != 0xFFFFFFFFu) ";
					code += "load_fl(&" + dst + ", 0x7FC00000u);\nelse ";
				}
				code += "set_fl(&" + dst + ", " + resultSign + "api.fmaf32("
				      + rs1 + ".f32[0], " + rs2 + ".f32[0], " + cSign + rs3 + ".f32[0]));\n";
			} else if (fi.R4type.funct2 == 0x1) { // float64
				code += "set_dbl(&" + dst + ", " + resultSign + "api.fmaf64("
				      + rs1 + ".f64, " + rs2 + ".f64, " + cSign + rs3 + ".f64));\n";
			} else {
				UNKNOWN_INSTRUCTION();
			}
			} break;
		case RV32F_FPFUNC: {
			const rv32f_instruction fi{instr};
			const auto dst = from_fpreg(fi.R4type.rd);
			const auto rs1 = from_fpreg(fi.R4type.rs1);
			const auto rs2 = from_fpreg(fi.R4type.rs2);
			const bool f32 = (fi.R4type.funct2 == 0x0);
			// NaN/sNaN predicates for FCSR flag computation. Only emitted under fcsr_emulation.
			auto is_nan = [] (const std::string& r, bool is_f32) -> std::string {
				if (is_f32)
					return "((" + r + ".i32[0] & 0x7f800000u) == 0x7f800000u && (" + r + ".i32[0] & 0x007fffffu) != 0)";
				return "((" + r + ".i64 & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (" + r + ".i64 & 0x000fffffffffffffull) != 0)";
			};
			auto is_snan = [] (const std::string& r, bool is_f32) -> std::string {
				if (is_f32)
					return "((" + r + ".i32[0] & 0x7fc00000u) == 0x7f800000u && (" + r + ".i32[0] & 0x003fffffu) != 0)";
				return "((" + r + ".i64 & 0x7ff8000000000000ull) == 0x7ff0000000000000ull && (" + r + ".i64 & 0x0007ffffffffffffull) != 0)";
			};
			auto raise_nv = [] (const std::string& cond) -> std::string {
				return "if (" + cond + ") cpu->fcsr |= 0x10;\n";
			};
			// Canonical qNaN substitution on NaN results (spec §11.3). Only under fcsr_emulation.
			auto emit_arith = [&] (const std::string& expr, bool is_f32) -> std::string {
				if constexpr (fcsr_emulation) {
					if (is_f32)
						return "{ const float fr = " + expr + "; if (fr != fr) load_fl(&" + dst + ", 0x7fc00000u); else set_fl(&" + dst + ", fr); }\n";
					return "{ const double dr = " + expr + "; if (dr != dr) load_dbl(&" + dst + ", 0x7ff8000000000000ull); else set_dbl(&" + dst + ", dr); }\n";
				} else {
					if (is_f32)
						return "set_fl(&" + dst + ", " + expr + ");\n";
					return "set_dbl(&" + dst + ", " + expr + ");\n";
				}
			};
			if (fi.R4type.funct2 < 0x2) { // fp32 / fp64
			switch (instr.fpfunc()) {
			case RV32F__FEQ_LT_LE:
				if (UNLIKELY(fi.R4type.rd == 0)) {
					UNKNOWN_INSTRUCTION();
					break;
				}
				// FLE/FLT are signaling compares: any NaN operand raises NV. FEQ is
				// a quiet compare and only raises NV for a signaling NaN. All of
				// them already yield 0 for unordered operands in plain C.
				// A non-NaN-boxed single-precision operand is read as the
				// canonical quiet NaN. FLE/FLT are signaling compares, so a
				// NaN operand (quiet or signaling) raises NV and the compare
				// is false; FEQ is quiet and raises NV only for sNaN. The
				// FLE/FLT NV is added in the fcsr_emulation block below via
				// the is_nan() test (a non-boxed operand is a NaN), so here
				// only the false result is forced.
				if constexpr (nanboxing && W == 8) {
					if (f32) {
						// A non-NaN-boxed operand is read as the canonical
						// quiet NaN: the compare is false and FLE/FLT raise
						// NV (signaling compare). The comparison and the
						// normal NV logic run in the else branch.
						code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu) { ";
						code += to_reg(fi.R4type.rd) + " = 0;";
						if constexpr (fcsr_emulation) {
							const unsigned op = fi.R4type.funct3 | (fi.R4type.funct2 << 4);
							if (op != 0x2 && op != 0x12) // FLE/FLT only
								code += " cpu->fcsr |= 0x10;";
						}
						code += " }\nelse { ";
					}
				}
				if constexpr (fcsr_emulation) {
					const unsigned op = fi.R4type.funct3 | (fi.R4type.funct2 << 4);
					if (op == 0x2 || op == 0x12) // FEQ.S / FEQ.D
						code += raise_nv(is_snan(rs1, f32) + " || " + is_snan(rs2, f32));
					else if (op <= 0x1 || (op >= 0x10 && op <= 0x11))
						code += raise_nv(is_nan(rs1, f32) + " || " + is_nan(rs2, f32));
				}
				switch (fi.R4type.funct3 | (fi.R4type.funct2 << 4)) {
				case 0x0: // FLE.S
					code += to_reg(fi.R4type.rd) + " = (" + rs1 + ".f32[0] <= " + rs2 + ".f32[0]) ? 1 : 0;\n";
					break;
				case 0x1: // FLT.S
					code += to_reg(fi.R4type.rd) + " = (" + rs1 + ".f32[0] < " + rs2 + ".f32[0]) ? 1 : 0;\n";
					break;
				case 0x2: // FEQ.S
					code += to_reg(fi.R4type.rd) + " = (" + rs1 + ".f32[0] == " + rs2 + ".f32[0]) ? 1 : 0;\n";
					break;
				case 0x10: // FLE.D
					code += to_reg(fi.R4type.rd) + " = (" + rs1 + ".f64 <= " + rs2 + ".f64) ? 1 : 0;\n";
					break;
				case 0x11: // FLT.D
					code += to_reg(fi.R4type.rd) + " = (" + rs1 + ".f64 < " + rs2 + ".f64) ? 1 : 0;\n";
					break;
				case 0x12: // FEQ.D
					code += to_reg(fi.R4type.rd) + " = (" + rs1 + ".f64 == " + rs2 + ".f64) ? 1 : 0;\n";
					break;
				default:
					UNKNOWN_INSTRUCTION();
				}
				if constexpr (nanboxing && W == 8) {
					if (f32) code += " }\n";
				}
				this->reset_tracked_register(fi.R4type.rd);
				break;
			case RV32F__FMIN_MAX:
				// RISC-V FMIN/FMAX: -0.0 < +0.0, unlike std::fmin/fmax.
				// Non-NaN-boxed operand → canonical qNaN.
				if constexpr (nanboxing && W == 8) {
					if (f32) {
						code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu) ";
						// Spelled as int: the JIT compiles this text as C99,
						// where <stdbool.h> is not in scope.
						code += "{ const int nb1 = (uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu;"
							" const int nb2 = (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu;"
							" if (nb1 && nb2) load_fl(&" + dst + ", 0x7fc00000u);"
							" else if (nb1) set_fl(&" + dst + ", " + rs2 + ".f32[0]);"
							" else set_fl(&" + dst + ", " + rs1 + ".f32[0]); }\nelse ";
					}
				}
				if constexpr (fcsr_emulation) {
					const unsigned op = fi.R4type.funct3 | (fi.R4type.funct2 << 4);
					if (op <= 0x1 || (op >= 0x10 && op <= 0x11))
						code += raise_nv(is_snan(rs1, f32) + " || " + is_snan(rs2, f32));
				}
				switch (fi.R4type.funct3 | (fi.R4type.funct2 << 4)) {
				case 0x0: // FMIN.S
					code += "set_fl(&" + dst + ", api.fmin32_rv(" + rs1 + ".f32[0], " + rs2 + ".f32[0]));\n";
					break;
				case 0x1: // FMAX.S
					code += "set_fl(&" + dst + ", api.fmax32_rv(" + rs1 + ".f32[0], " + rs2 + ".f32[0]));\n";
					break;
				case 0x10: // FMIN.D
					code += "set_dbl(&" + dst + ", api.fmin64_rv(" + rs1 + ".f64, " + rs2 + ".f64));\n";
					break;
				case 0x11: // FMAX.D
					code += "set_dbl(&" + dst + ", api.fmax64_rv(" + rs1 + ".f64, " + rs2 + ".f64));\n";
					break;
				default:
					UNKNOWN_INSTRUCTION();
				} break;
			case RV32F__FADD:
			case RV32F__FSUB: {
				const std::string fop = (instr.fpfunc() == RV32F__FSUB) ? " - " : " + ";
				if (f32) {
					if constexpr (nanboxing && W == 8) {
						code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu) ";
						code += "load_fl(&" + dst + ", 0x7FC00000u);\nelse ";
					}
					if constexpr (fcsr_emulation) {
						// NX when the exact (double-precision) result does not
						// round-trip to the single-precision result. The
						// operands are widened *before* the operation, which is
						// what makes the comparison meaningful; the double sum
						// is exact unless the operand exponents are more than
						// ~29 apart, so a handful of extreme cases under-report.
						code += "{ const float fa = " + rs1 + ".f32[0], fb = " + rs2 + ".f32[0];"
							" const double exact = (double)fa" + fop + "(double)fb;"
							" const float fr = fa" + fop + "fb;"
							" if (fr != fr) load_fl(&" + dst + ", 0x7fc00000u);"
							" else { set_fl(&" + dst + ", fr); if ((double)fr != exact) cpu->fcsr |= 1; } }\n";
					} else {
						code += emit_arith(rs1 + ".f32[0]" + fop + rs2 + ".f32[0]", true);
					}
				}
				else
					code += emit_arith(rs1 + ".f64" + fop + rs2 + ".f64", false);
				} break;
			case RV32F__FMUL:
				// Finite operands that produce an infinity overflowed; finite
				// operands that produce an inexact subnormal underflowed. Both
				// imply NX. The operands are read into locals first, because rd
				// is allowed to alias rs1/rs2.
				if constexpr (fcsr_emulation) {
					if (f32) {
						if constexpr (nanboxing && W == 8) {
							code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu) ";
							code += "load_fl(&" + dst + ", 0x7FC00000u);\nelse ";
						}
						code += "{ const uint32_t ia = " + rs1 + ".i32[0], ib = " + rs2 + ".i32[0];"
							" const float fa = " + rs1 + ".f32[0], fb = " + rs2 + ".f32[0];"
							" const float fr = fa * fb;"
							" if (fr != fr) load_fl(&" + dst + ", 0x7fc00000u); else set_fl(&" + dst + ", fr);"
							" if ((ia & 0x7f800000u) != 0x7f800000u && (ib & 0x7f800000u) != 0x7f800000u) {"
							" const uint32_t ir = " + dst + ".i32[0] & 0x7fffffffu;"
							" if (ir == 0x7f800000u) { cpu->fcsr |= 5;"
							" const unsigned rm = (cpu->fcsr >> 5) & 7;"
							" const int neg = (" + dst + ".i32[0] & 0x80000000u) != 0;"
							" if (rm == 1 || (rm == 2 && !neg) || (rm == 3 && neg))"
							" " + dst + ".i32[0] = (" + dst + ".i32[0] & 0x80000000u) | 0x7F7FFFFFu; }"
							" else if (ir < 0x00800000u && (double)fa * (double)fb != fr) cpu->fcsr |= 3; } }\n";
						break;
					}
				}
				if constexpr (nanboxing && W == 8) {
					if (f32) {
						code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu) ";
						code += "load_fl(&" + dst + ", 0x7FC00000u);\nelse ";
					}
				}
				code += emit_arith(f32 ? (rs1 + ".f32[0] * " + rs2 + ".f32[0]")
									   : (rs1 + ".f64 * " + rs2 + ".f64"), f32);
				break;
			case RV32F__FDIV:
				// DZ is only for a finite non-zero numerator over zero: 0/0 is
				// NV and inf/0 is exact. NV/NX/OF/UF are also required.
				if constexpr (fcsr_emulation) {
					if (f32) {
						if constexpr (nanboxing && W == 8) {
							code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu) ";
							code += "load_fl(&" + dst + ", 0x7FC00000u);\nelse ";
						}
						code += "{ const uint32_t ia = " + rs1 + ".i32[0], ib = " + rs2 + ".i32[0];"
							" const float fa = " + rs1 + ".f32[0], fb = " + rs2 + ".f32[0];"
							" const float fr = fa / fb;"
							" if (fr != fr) { load_fl(&" + dst + ", 0x7fc00000u);"
							" if ((ia & 0x7fffffffu) == 0 && (ib & 0x7fffffffu) == 0) cpu->fcsr |= 0x10;"
							" else if ((ia & 0x7f800000u) == 0x7f800000u && (ib & 0x7f800000u) == 0x7f800000u) cpu->fcsr |= 0x10; }"
							" else { set_fl(&" + dst + ", fr);"
							" if ((ia & 0x7fffffffu) != 0 && (ia & 0x7f800000u) != 0x7f800000u"
							" && (ib & 0x7fffffffu) == 0) cpu->fcsr |= 8;"
							" if ((double)fr != (double)fa / (double)fb) cpu->fcsr |= 1;"
							" const uint32_t ir = " + dst + ".i32[0] & 0x7fffffffu;"
							" if ((ib & 0x7fffffffu) != 0 && (ia & 0x7f800000u) != 0x7f800000u && ir == 0x7f800000u) cpu->fcsr |= 5;"
							" else if (ir < 0x00800000u && (double)fr != (double)fa / (double)fb) cpu->fcsr |= 3; } }\n";
					} else {
						code += "{ const uint64_t ia = " + rs1 + ".i64, ib = " + rs2 + ".i64;"
							" const double fa = " + rs1 + ".f64, fb = " + rs2 + ".f64;"
							" const double dr = fa / fb;"
							" if (dr != dr) { load_dbl(&" + dst + ", 0x7ff8000000000000ull);"
							" if ((ia & 0x7fffffffffffffffull) == 0 && (ib & 0x7fffffffffffffffull) == 0) cpu->fcsr |= 0x10;"
							" else if ((ia & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (ib & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) cpu->fcsr |= 0x10; }"
							" else { set_dbl(&" + dst + ", dr);"
							" if ((ia & 0x7fffffffffffffffull) != 0 && (ia & 0x7ff0000000000000ull) != 0x7ff0000000000000ull"
							" && (ib & 0x7fffffffffffffffull) == 0) cpu->fcsr |= 8;"
							" if ((long double)dr != (long double)fa / (long double)fb) cpu->fcsr |= 1;"
							" const uint64_t ir = " + dst + ".i64 & 0x7fffffffffffffffull;"
							" if ((ib & 0x7fffffffffffffffull) != 0 && (ia & 0x7ff0000000000000ull) != 0x7ff0000000000000ull && ir == 0x7ff0000000000000ull) cpu->fcsr |= 5;"
							" else if (ir < 0x0010000000000000ull && (long double)dr != (long double)fa / (long double)fb) cpu->fcsr |= 3; } }\n";
					}
				} else {
					if constexpr (nanboxing && W == 8) {
						if (f32) {
							code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu) ";
							code += "load_fl(&" + dst + ", 0x7FC00000u);\nelse ";
						}
					}
					code += emit_arith(f32 ? (rs1 + ".f32[0] / " + rs2 + ".f32[0]")
										   : (rs1 + ".f64 / " + rs2 + ".f64"), f32);
				}
				this->penalty(f32 ? 10 : 15); // division is a slow operation
				break;
			case RV32F__FSQRT:
				// sqrt of a NaN or of a negative number is the canonical qNaN.
				// It is an invalid operation for a negative input or a signaling
				// NaN, but *not* for a quiet NaN. The invalid-flag condition is
				// evaluated before the store, since rd may alias rs1.
				if constexpr (fcsr_emulation) {
					if (f32) {
						if constexpr (nanboxing && W == 8) {
							code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu) ";
							code += "load_fl(&" + dst + ", 0x7FC00000u);\nelse ";
						}
						code += "{ const int inv = " + is_snan(rs1, true) + " || " + rs1 + ".f32[0] < 0.0f;"
							" if (" + is_nan(rs1, true) + " || " + rs1 + ".f32[0] < 0.0f)"
							" { load_fl(&" + dst + ", 0x7fc00000u); if (inv) cpu->fcsr |= 0x10; }"
							" else { const float sq = api.sqrtf32(" + rs1 + ".f32[0]); set_fl(&" + dst + ", sq);"
							" if ((double)sq * (double)sq != (double)" + rs1 + ".f32[0]) cpu->fcsr |= 1; } }\n";
					} else {
						code += "{ const int inv = " + is_snan(rs1, false) + " || " + rs1 + ".f64 < 0.0;"
							" if (" + is_nan(rs1, false) + " || " + rs1 + ".f64 < 0.0)"
							" { load_dbl(&" + dst + ", 0x7ff8000000000000ull); if (inv) cpu->fcsr |= 0x10; }"
							" else { const double sq = api.sqrtf64(" + rs1 + ".f64); set_dbl(&" + dst + ", sq);"
							" if (sq * sq != " + rs1 + ".f64) cpu->fcsr |= 1; } }\n";
					}
				} else if (f32) {
					if constexpr (nanboxing && W == 8) {
						code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu) ";
						code += "load_fl(&" + dst + ", 0x7FC00000u);\nelse ";
					}
					code += "set_fl(&" + dst + ", api.sqrtf32(" + rs1 + ".f32[0]));\n";
				} else {
					code += "set_dbl(&" + dst + ", api.sqrtf64(" + rs1 + ".f64));\n";
				}
				this->penalty(f32 ? 10 : 15); // sqrt is a slow operation
				break;
			case RV32F__FSGNJ_NX:
				// Non-boxed → canonical qNaN. FSGNJN inverts the qNaN sign (§11.6).
				if constexpr (nanboxing && W == 8) {
					if (fi.R4type.funct2 == 0x0) {
						code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu || (uint32_t)" + rs2 + ".i32[1] != 0xFFFFFFFFu) ";
						code += "load_fl(&" + dst + ", " + (fi.R4type.funct3 == 0x1 ? "0xFFC00000u" : "0x7FC00000u") + ");\nelse ";
					}
				}
				switch (fi.R4type.funct3) {
				case 0x0: // FSGNJ
					// FMV rd, rs1
					if (fi.R4type.rs1 == fi.R4type.rs2) {
						code += dst + ".i64 = " + rs1 + ".i64;\n";
					} else {
					if (fi.R4type.funct2 == 0x0) { // fp32
						code += "load_fl(&" + dst + ", (" + rs2 + ".lsign.sign << 31) | " + rs1 + ".lsign.bits);\n";
					} else { // fp64
						code += "load_dbl(&" + dst + ", ((uint64_t)" + rs2 + ".usign.sign << 63) | " + rs1 + ".usign.bits);\n";
					} } break;
				case 0x1: // FSGNJ_N
					if (fi.R4type.funct2 == 0x0) { // fp32
						code += "load_fl(&" + dst + ", (~" + rs2 + ".lsign.sign << 31) | " + rs1 + ".lsign.bits);\n";
					} else { // fp64
						code += "load_dbl(&" + dst + ", (~(uint64_t)" + rs2 + ".usign.sign << 63) | " + rs1 + ".usign.bits);\n";
					} break;
				case 0x2: // FSGNJ_X
					if (fi.R4type.funct2 == 0x0) { // fp32
						code += "load_fl(&" + dst + ", ((" + rs1 + ".lsign.sign ^ " + rs2 + ".lsign.sign) << 31) | " + rs1 + ".lsign.bits);\n";
					} else { // fp64
						code += "load_dbl(&" + dst + ", ((uint64_t)(" + rs1 + ".usign.sign ^ " + rs2 + ".usign.sign) << 63) | " + rs1 + ".usign.bits);\n";
					} break;
				default:
					UNKNOWN_INSTRUCTION();
				} break;
			case RV32F__FCVT_SD_DS:
				// Only S↔D inlined; Zfhmin/quad go to handler.
				if (fi.R4type.rs2 != (fi.R4type.funct2 ^ 1)) {
					UNKNOWN_INSTRUCTION();
				} else if (fi.R4type.funct2 == 0x0) {
					code += "if (" + rs1 + ".f64 != " + rs1 + ".f64) load_fl(&" + dst + ", 0x7fc00000u); else set_fl(&" + dst + ", " + rs1 + ".f64);\n";
				} else if (fi.R4type.funct2 == 0x1) {
					if constexpr (nanboxing && W == 8) {
						code += "if ((uint32_t)" + rs1 + ".i32[1] != 0xFFFFFFFFu) ";
						code += "load_dbl(&" + dst + ", 0x7ff8000000000000ull);\nelse ";
					}
					code += "if (" + rs1 + ".f32[0] != " + rs1 + ".f32[0]) load_dbl(&" + dst + ", 0x7ff8000000000000ull); else set_dbl(&" + dst + ", " + rs1 + ".f32[0]);\n";
				} else {
					UNKNOWN_INSTRUCTION();
				} break;
			case RV32F__FCVT_SD_W: {
				if (fi.R4type.funct2 < 0x2 && fi.R4type.rs2 < 0x4) {
					// FCVT.{S,D}.[LWU]
					static const std::array<const char*, 4> int_type {
						"int32_t", "uint32_t", "int64_t", "uint64_t"
					};
					const std::string itype = int_type[fi.R4type.rs2];
					const std::string value =
						"(" + itype + ")" + from_reg(fi.R4type.rs1);
					const std::string setter = f32 ? "set_fl" : "set_dbl";
					if constexpr (fcsr_emulation) {
						// NX when the destination mantissa was too narrow for the
						// integer. long double is exact for any 64-bit integer
						// where it is an 80- or 128-bit format; where it is only
						// double, the 64-bit sources just never report NX.
						// The host cast is round-to-nearest-even; the other
						// RISC-V rounding modes need a nextafter adjustment on
						// the float grid (see the interpreter counterpart).
						const std::string nafn  = f32 ? "nextafterf" : "nextafter";
						const std::string ftype = f32 ? "float" : "double";
						const std::string inf   = f32 ? "__builtin_inff()" : "__builtin_inf()";
						code += "{ const " + itype + " iv = " + value + ";"
							" const long double iex = (long double)iv;"
							" const " + ftype + " fv = (" + ftype + ")iv;"
							" const long double cld = (long double)fv;"
							" " + ftype + " out = fv;"
							" if (cld != iex) { cpu->fcsr |= 1;"
							" const unsigned rm = (" + std::to_string((int)fi.R4type.funct3) + " == 0x7) ? ((cpu->fcsr >> 5) & 7) : " + std::to_string((int)fi.R4type.funct3) + ";"
							" if (rm == 1) { if ((iex > 0.0L && cld > iex) || (iex < 0.0L && cld < iex)) out = " + nafn + "(out, (" + ftype + ")0); }"
							" else if (rm == 2) { if (cld > iex) out = " + nafn + "(out, -" + inf + "); }"
							" else if (rm == 3) { if (cld < iex) out = " + nafn + "(out, " + inf + "); }"
							// RMM differs from RNE only at an exact halfway
							// point, where it takes the larger magnitude.
							" else if (rm == 4) { if (__builtin_fabsl(cld) < __builtin_fabsl(iex)) {"
							" const " + ftype + " away = " + nafn + "(out, out < (" + ftype + ")0 ? -" + inf + " : " + inf + ");"
							" if (__builtin_fabsl((long double)away - iex) == __builtin_fabsl(iex - cld)) out = away; } } }"
							" " + setter + "(&" + dst + ", out); }\n";
					} else {
						code += setter + "(&" + dst + ", " + value + ");\n";
					}
				} else {
					UNKNOWN_INSTRUCTION();
				}
				} break;
			case RV32F__FCVT_W_SD: {
				const auto rmm = fi.R4type.funct3; // rounding mode in funct3
				if (fi.R4type.rd != 0 &&
					(fi.R4type.funct2 == 0x0 || fi.R4type.funct2 == 0x1)) {
					const bool from_float = (fi.R4type.funct2 == 0x0);
					// A non-NaN-boxed single-precision source is read as the
					// canonical quiet NaN, which converts to the maximum value
					// with NV below. FCVT must not write rs1, so the
					// substitution goes into a local copy of the register.
					const bool boxcheck = nanboxing && W == 8 && from_float;
					std::string srcdecl;
					if (boxcheck) {
						srcdecl = " fp64reg fcvt_s = " + rs1 + ";"
							" if ((uint32_t)fcvt_s.i32[1] != 0xFFFFFFFFu)"
							" fcvt_s.i32[0] = 0x7fc00000;";
					}
					const std::string src = boxcheck ? "fcvt_s.f32[0]"
						: (from_float ? (rs1 + ".f32[0]") : (rs1 + ".f64"));
					// Round per the RISC-V rounding mode (funct3), matching the
					// interpreter's fcvt_to_integer(). A bare cast is RTZ only,
					// which is wrong for e.g. RMM (std::lround) and RDN (floor).
					const char* trunc_fn = from_float ? "truncf" : "trunc"; // RTZ
					const char* floor_fn = from_float ? "floorf" : "floor"; // RDN
					const char* ceil_fn  = from_float ? "ceilf"  : "ceil";  // RUP
					const char* round_fn = from_float ? "roundf" : "round"; // RMM
					const char* near_fn  = from_float ? "nearbyintf" : "nearbyint"; // RNE
					std::string rounded;
					if (rmm == 0x7) {
						// DYN: resolve the rounding mode from the fcsr CSR (frm
						// field, bits [7:5]) at runtime. src has no side effects,
						// so it is safe to repeat across the branches.
						const std::string frm = "((cpu->fcsr >> 5) & 7)";
						rounded =
							"(" + frm + "==1?" + trunc_fn + "(" + src + "):"
								+ frm + "==2?" + floor_fn + "(" + src + "):"
								+ frm + "==3?" + ceil_fn  + "(" + src + "):"
								+ frm + "==4?" + round_fn + "(" + src + "):"
								+ near_fn + "(" + src + "))";
					} else {
						const char* rfn;
						switch (rmm) {
						case 0x1: rfn = trunc_fn; break; // RTZ
						case 0x2: rfn = floor_fn; break; // RDN
						case 0x3: rfn = ceil_fn;  break; // RUP
						case 0x4: rfn = round_fn; break; // RMM
						default:  rfn = near_fn;         // RNE
						}
						rounded = std::string(rfn) + "(" + src + ")";
					}
					// Range check required: out-of-range float→int is UB in C. RISC-V pins the result to
					// the destination's extreme (NaN and positive overflow give
					// the maximum, negative overflow the minimum) and raises NV.
					// The bounds are powers of two, hence exactly representable.
					const char *lower = "0.0", *upper = "0.0";
					const char *maxval = "0", *minval = "0", *cast = "";
					switch (fi.R4type.rs2) {
					case 0x0: // FCVT.W (int32, sign-extended)
						lower = "-2147483648.0"; upper = "2147483648.0";
						maxval = "(int32_t)2147483647"; minval = "(int32_t)(-2147483647 - 1)";
						cast = "(int32_t)";
						break;
					case 0x1: // FCVT.WU (uint32 result, sign-extended to XLEN)
						lower = "0.0"; upper = "4294967296.0";
						maxval = "(int32_t)0xffffffffu"; minval = "(int32_t)0";
						cast = "(int32_t)(uint32_t)";
						break;
					case 0x2: // FCVT.L (int64)
						lower = "-9223372036854775808.0"; upper = "9223372036854775808.0";
						maxval = "(int64_t)9223372036854775807ll";
						minval = "(int64_t)(-9223372036854775807ll - 1)";
						cast = "(int64_t)";
						break;
					case 0x3: // FCVT.LU (uint64)
						lower = "0.0"; upper = "18446744073709551616.0";
						maxval = "(uint64_t)18446744073709551615ull"; minval = "(uint64_t)0";
						cast = "(uint64_t)";
						break;
					default: // Reserved rs2 encoding
						UNKNOWN_INSTRUCTION();
						break;
					}
					if (fi.R4type.rs2 < 0x4) {
						const std::string reg = to_reg(fi.R4type.rd);
						code += "{" + srcdecl + " const double fcvt_r = " + rounded + ";"
							" if (!(fcvt_r >= " + lower + " && fcvt_r < " + upper + ")) {"
							" " + reg + " = (fcvt_r != fcvt_r || fcvt_r >= " + upper + ") ? "
							+ maxval + " : " + minval + ";"
							+ (fcsr_emulation ? " cpu->fcsr |= 16;" : "")
							+ " } else { " + reg + " = " + cast + "fcvt_r;"
							+ (fcsr_emulation ? " if (fcvt_r != (double)(" + src + ")) cpu->fcsr |= 1;" : "")
							+ " } }\n";
					}
				} else {
					UNKNOWN_INSTRUCTION();
				}
				this->reset_tracked_register(fi.R4type.rd);
				} break;
			case RV32F__FMV_W_X:
				if (fi.R4type.rs2 != 0x0) {
					UNKNOWN_INSTRUCTION();
				} else if (fi.R4type.funct2 == 0x0) {
					code += "load_fl(&" + dst + ", " + from_reg(fi.R4type.rs1) + ");\n";
				} else if (W == 8 && fi.R4type.funct2 == 0x1) {
					code += "load_dbl(&" + dst + ", " + from_reg(fi.R4type.rs1) + ");\n";
				} else {
					UNKNOWN_INSTRUCTION();
				} break;
			case RV32F__FMV_X_W:
				if (fi.R4type.funct3 == 0x0 && fi.R4type.rs2 == 0x0) {
					if (fi.R4type.rd != 0 && fi.R4type.funct2 == 0x0) {
						code += to_reg(fi.R4type.rd) + " = " + rs1 + ".i32[0];\n";
					} else if (W == 8 && fi.R4type.rd != 0 && fi.R4type.funct2 == 0x1) { // 64-bit only
						code += to_reg(fi.R4type.rd) + " = " + rs1 + ".i64;\n";
					} else {
						UNKNOWN_INSTRUCTION();
					}
				} else { // FPCLASSIFY etc.
					UNKNOWN_INSTRUCTION();
				}
				this->reset_tracked_register(fi.R4type.rd);
				break;
			default:
				UNKNOWN_INSTRUCTION();
				break;
			} // fpfunc
			} else {
				// Unhandled format (Zfhmin/quad): use handler, reset tracked rd.
				UNKNOWN_INSTRUCTION();
				this->reset_tracked_register(fi.R4type.rd);
			}
			} break; // RV32F_FPFUNC
		case RV32A_ATOMIC: // General handler for atomics
			this->penalty(20); // Atomic operations are slow
			load_register(instr.Atype.rd);
			load_register(instr.Atype.rs1);
			load_register(instr.Atype.rs2);
			this->potentially_realize_register(instr.Atype.rd);
			this->potentially_realize_register(instr.Atype.rs1);
			this->potentially_realize_register(instr.Atype.rs2);
			WELL_KNOWN_INSTRUCTION();
			this->reset_tracked_register(instr.Atype.rd);
			this->potentially_reload_register(instr.Atype.rd);
			this->potentially_reload_register(instr.Atype.rs1);
			this->potentially_reload_register(instr.Atype.rs2);
			break;
		case RV32V_OP:
			// RVV: element-wise float inlined; all else to interpreter.
#ifdef RISCV_EXT_VECTOR
			this->emit_vector_instruction();
#else
			UNKNOWN_INSTRUCTION();
#endif
			break;
		case Dyncall::opcode: {
			if (!Dyncall::valid(instr.whole)) {
				UNKNOWN_INSTRUCTION();
				this->reset_all_tracked_registers();
				break;
			}
			const unsigned inputs = Dyncall::inputs(instr.whole);
			const unsigned outputs = Dyncall::outputs(instr.whole);
			for (unsigned i = 10; i < 10 + inputs; i++)
				this->load_register(i);
			this->store_registers(this->dirty_registers() & Dyncall::arg_mask(inputs));
			// Publish PC even without a budget: diagnostics attribute host calls
			// to this instruction. Counters are synchronized only when enabled.
			code += "cpu->pc = " + PCRELS(0) + ";\n";
			if (!tinfo.ignore_instruction_limit) {
				this->increment_counter_so_far();
				code += "INS_COUNTER(cpu) = ic; MAX_COUNTER(cpu) = max_ic;\n";
			}
			WELL_KNOWN_INSTRUCTION();
			for (unsigned i = 10; i < 10 + outputs; i++) {
				this->load_register(i);
				this->potentially_reload_register(i);
				this->reset_tracked_register(i);
			}
			if (!tinfo.ignore_instruction_limit) {
				code += "ic = INS_COUNTER(cpu); max_ic = MAX_COUNTER(cpu);\n";
				code += "if (UNLIKELY(ic >= max_ic)) {\n";
				this->exit_function(PCRELS(4), true);
			}
			break;
		}
		default:
			UNKNOWN_INSTRUCTION();
		}
	}
	// Fall-through exit at block end.
	this->increment_counter_so_far();
	exit_function(STRADDR(this->end_pc()));
#ifdef RISCV_EXT_VECTOR
	// Unreachable by fall-through: exit_function() above always returns.
	this->emit_vector_trap_epilogue();
#endif
}

template <int W>
std::vector<TransMapping<W>>
CPU<W>::emit(std::string& code, const TransInfo<W>& tinfo)
{
	Emitter<W> e(tinfo);
	e.emit();

	// Create register push and pop macros
	if (tinfo.use_register_caching) {
		for (const uint32_t mask : e.get_store_masks()) {
			code += "#define STORE_REGS_" + e.get_func() + "_" + hex_address(mask) + "() \\\n";
			for (int reg = 1; reg < e.CACHED_REGISTERS; reg++) {
				if (e.gpr_needs_store(reg) && (mask & (1u << reg))) {
					code += "  cpu->r[" + std::to_string(reg) + "] = " + e.loaded_regname(reg) + "; \\\n";
				}
			}
			code += "  ;\n";
		}
		code += "#define LOAD_REGS_" + e.get_func() + "() \\\n";
		for (int reg = 1; reg < e.CACHED_REGISTERS; reg++) {
			if (e.gpr_exists_at(reg)) {
				code += "  " + e.loaded_regname(reg) + " = cpu->r[" + std::to_string(reg) + "]; \\\n";
			}
		}
		code += "  ;\n";
		// A system call returns its values in A0 and A1
		code += "#define LOAD_SYS_REGS_" + e.get_func() + "() \\\n";
		for (int reg = 10; reg < 12; reg++) {
			if (e.gpr_exists_at(reg)) {
				code += "  " + e.loaded_regname(reg) + " = cpu->r[" + std::to_string(reg) + "]; \\\n";
			}
		}
		code += "  ;\n";
	}

	// Forward declarations
	for (const auto& entry : e.get_forward_declared()) {
		code += "static ReturnValues " + entry + "(CPU*, uint64_t, uint64_t, addr_t);\n";
	}

	// Function header
	code += "static ReturnValues " + e.get_func() + "(CPU* RESTRICT cpu, uint64_t ic, uint64_t max_ic, addr_t pc) {\n";
	// NOTE: Scratch shared by every exit point, see RETURN_VALUES
	code += "ReturnValues retvals;\n";
	if (e.used_fixed_store())
		code += "char* mstore;\n";

	// Function GPRs
	if (tinfo.use_register_caching) {
		for (int reg = 1; reg < e.CACHED_REGISTERS; reg++) {
			if (e.gpr_exists_at(reg)) {
				code += "addr_t " + e.loaded_regname(reg) + " = cpu->r[" + std::to_string(reg) + "];\n";
			}
		}
	}

	code += e.get_func() + "_jumptbl:;\n";
	{
		// Indirect dispatch over the entry points of this block. Switching on
		// the halfword index rather than the address doubles the case density,
		// which is what decides whether the compiler emits a single jump
		// table or a binary search of sub-tables: several data-dependent
		// branches on every return and indirect call. Only entry points are
		// cases: every case constrains register allocation at its label, and
		// making all branch targets reachable from here measured 25% slower.
		// Indirect jump targets found in data (switch and function pointer
		// tables) are entry points, so those stay inside the function too.
		// Computed gotos would be denser still, but their targets become
		// abnormal edges that block optimization around every label.
		std::vector<address_type<W>> cases;
		for (const auto& entry : e.get_mappings())
			cases.push_back(entry.addr);
		std::sort(cases.begin(), cases.end());
		cases.erase(std::unique(cases.begin(), cases.end()), cases.end());

		code += "switch ((pc - " + hex_address(e.begin_pc()) + "LL) >> 1) {\n";
		for (const auto addr : cases) {
			code += "case " + std::to_string(uint64_t(addr - e.begin_pc()) >> 1) + ": goto "
				+ funclabel<W>(e.get_func(), addr) + ";\n";
		}
		code += "default:;\n}\n";
	}
	code += "exception_is_handled:\n"; // Re-using exit point for exceptions
	for (int reg = 1; reg < e.CACHED_REGISTERS; reg++) {
		if (e.gpr_needs_store(reg)) {
			code += "  cpu->r[" + std::to_string(reg) + "] = " + e.loaded_regname(reg) + ";\n";
		}
	}
	code += "  cpu->pc = pc; RETURN_VALUES(ic, max_ic);\n";

	// Function code
	code += e.get_code();

	// Exception handler
	if (!tinfo.use_virtual_paging_fallback) {
		code += "exception:\n api.exception(cpu, pc, 2); max_ic = 0; goto exception_is_handled;\n";
	}
	code += "}\n";

	return std::move(e.get_mappings());
}

#ifdef RISCV_32I
template std::vector<TransMapping<4>> CPU<4>::emit(std::string&, const TransInfo<4>&);
#endif
#ifdef RISCV_64I
template std::vector<TransMapping<8>> CPU<8>::emit(std::string&, const TransInfo<8>&);
#endif
#ifdef RISCV_128I
template std::vector<TransMapping<16>> CPU<16>::emit(std::string&, const TransInfo<16>&);
#endif
} // riscv
