#pragma once
#include "../common.hpp"
#include "../instruction_list.hpp"
#include "../dyncall.hpp"
#include "../rv32i_instr.hpp"
#include "../rvfd.hpp"
#include "../rvv.hpp"
#include "../types.hpp"
#include "aj_api.hpp"
#include "aj_decode.hpp"

#include <vector>

namespace riscv
{
	template <int W> struct DecodedExecuteSegment;

	/// @brief Everything the emitter needs to know about the host machine layout.
	/// @details All displacements are relative to `CPU<W>&`, the first argument of
	/// every emitted function. They are measured from a live CPU instance rather
	/// than with offsetof(), which keeps them correct for any CPU<W> layout and
	/// for any Machine<W> the emitted code is later shared with.
	template <int W>
	struct AjInfo
	{
		int32_t reg_offset      = 0;   ///< &cpu.registers().get(0) - &cpu
		int32_t fpreg_offset    = 0;   ///< &cpu.registers().getfl(0) - &cpu
		int32_t arena_ptr       = 0;   ///< &memory.m_arena.data    - &cpu
		/// @brief Arena bounds baked when the execute segment is translated.
		/// @details Flat-arena bounds are immutable after construction and are
		/// shared unchanged by forks, so emitted code need not reload them from
		/// CPU memory at every guest access.
		uint64_t arena_rdbound_value = 0;
		uint64_t arena_wrbound_value = 0;
		uint64_t arena_roend_value   = 0;
		// The vector state, for the inlined vector loads and stores. vl, SEW
		// and LMUL are read at run-time: vsetvli is not inlined, so vtype is
		// unknown at translation time.
		int32_t rvv_regs        = 0;   ///< &cpu.registers().rvv().get(0) - &cpu
		int32_t rvv_vl          = 0;
		int32_t rvv_vsew        = 0;   ///< log2(SEW / 8)
		int32_t rvv_lmul        = 0;   ///< log2(LMUL), signed
		int32_t rvv_vill        = 0;
		/// @brief True when loads and stores may be inlined against the arena.
		bool inline_memory = false;
		/// @brief Nonzero when the arena is an N-bit encompassing one, holding
		/// the address mask that maps a guest address into it.
		/// @details Such an arena spans the whole reachable address space, so an
		/// access cannot land outside it and there is no bounds check and no
		/// out-of-line path at all -- masking the address *is* the translation,
		/// exactly as Memory<W>::read()/write() do it. Zero means the flat arena,
		/// which is indexed by the address itself behind a bounds check.
		uint64_t arena_mask = 0;
		bool unsafe_remove_checks = false;
		bool ignore_instruction_limit = false;
		bool direct_calls = true;
		unsigned direct_call_depth = 256;
		const AjCallbacks<W>* cb = nullptr;
	};

	/// @brief True when the F/D extension may be emitted as native code.
	/// @details FCSR emulation tracks the exception flags of every arithmetic
	/// operation by recomputing it in a wider type and comparing, which no host
	/// instruction sequence reproduces. Under that build every FP instruction
	/// stays with the interpreter, and only ends a region.
	static constexpr bool aj_fpu_emittable = !riscv::fcsr_emulation;

	/// @brief True when this host can compute a fused multiply-add.
	/// @details RISC-V requires FMADD and its three siblings to round once. A
	/// separate multiply and add rounds twice, so on a host without FMA those
	/// four opcodes end the region and the interpreter's std::fma runs instead.
	/// Defined in aj_emit.cpp, where the asmjit headers are available.
	bool aj_host_has_fma() noexcept;

	/// @brief One bit-manipulation operation, as the emitter thinks of it.
	/// @details Covers Zba, Zbb, Zbc and Zbs, plus the PACK family that ZEXT.H is
	/// the rs2 == 0 form of, and Zicond, which shares the OP opcode and so is
	/// classified here as well.
	enum class AjZb : uint8_t
	{
		kNone = 0,
		// --- Zba: address generation ---
		kShAdd,     ///< SH1ADD, SH2ADD, SH3ADD           (the scale is in funct3)
		kShAddUw,   ///< SH1ADD.UW, SH2ADD.UW, SH3ADD.UW  (RV64)
		kAddUw,     ///< ADD.UW                           (RV64)
		kSlliUw,    ///< SLLI.UW                          (RV64)
		// --- Zbb: logic with negate ---
		kAndn, kOrn, kXnor,
		// --- Zbb: min and max ---
		kMin, kMinu, kMax, kMaxu,
		// --- Zbb: bit counting ---
		kClz, kCtz, kCpop,
		kClzw, kCtzw, kCpopw,          ///< RV64
		// --- Zbb: extend, reverse, combine ---
		kSextB, kSextH, kRev8, kOrcB,
		kPack, kPackH,                 ///< ZEXT.H on RV32 is PACK with rs2 == 0
		kPackW,                        ///< RV64; ZEXT.H is its rs2 == 0 form
		// --- Zbb: rotate ---
		kRol, kRor, kRori,
		kRolw, kRorw, kRoriw,          ///< RV64
		// --- Zbs: single-bit ---
		kBset, kBclr, kBinv, kBext,
		kBseti, kBclri, kBinvi, kBexti,
		// --- Zbc: carry-less multiply ---
		kClmul, kClmulh, kClmulr,
		// --- Zicond: conditional zero ---
		kCzeroEqz, kCzeroNez,
	};

	/// @brief Classifies one instruction as a bit-manipulation operation.
	/// @details This is the single source of truth for the Zb encodings: both
	/// aj_is_emittable() and the emitter dispatch on it, so region discovery can
	/// never claim an encoding that emission then quietly skips -- which would
	/// leave rd holding a stale value rather than faulting.
	/// @tparam W The guest register width. RV64 adds the *W and *.UW forms, and
	/// widens the shift amount that a shamt-carrying encoding may hold.
	template <int W>
	inline AjZb aj_zb_classify(rv32i_instruction i) noexcept
	{
		constexpr bool RV64 = (W == 8);
		// On RV32 the shift amount is five bits wide and bit 5 belongs to funct7,
		// so every shamt-carrying encoding that sets it is reserved instead.
		const bool shamt_ok = RV64 || (i.Itype.imm & 0x20) == 0;

		switch (i.opcode())
		{
		case RV32I_OP:
			// The key is funct3 | funct7 << 4, the same one the interpreter's
			// jumptable_friendly_op() builds, so rvi_instr.cpp can be read
			// alongside this table.
			switch (i.Rtype.jumptable_friendly_op()) {
			case 0x44:  return AjZb::kPack;    // ZEXT.H on RV32 when rs2 == 0
			case 0x47:  return AjZb::kPackH;
			case 0x51:  return AjZb::kClmul;
			case 0x52:  return AjZb::kClmulr;
			case 0x53:  return AjZb::kClmulh;
			case 0x54:  return AjZb::kMin;
			case 0x55:  return AjZb::kMinu;
			case 0x56:  return AjZb::kMax;
			case 0x57:  return AjZb::kMaxu;
			case 0x75:  return AjZb::kCzeroEqz;
			case 0x77:  return AjZb::kCzeroNez;
			case 0x102: case 0x104: case 0x106:
				return AjZb::kShAdd;
			case 0x141: return AjZb::kBset;
			case 0x204: return AjZb::kXnor;
			case 0x206: return AjZb::kOrn;
			case 0x207: return AjZb::kAndn;
			case 0x241: return AjZb::kBclr;
			case 0x245: return AjZb::kBext;
			case 0x301: return AjZb::kRol;
			case 0x305: return AjZb::kRor;
			case 0x341: return AjZb::kBinv;
			default:    return AjZb::kNone;
			}

		case RV32I_OP_IMM:
			if (i.Itype.funct3 == 0x1) {
				// The unary forms name no shift amount at all, so the whole
				// 12-bit immediate identifies them.
				switch (i.Itype.imm) {
				case 0x600: return AjZb::kClz;
				case 0x601: return AjZb::kCtz;
				case 0x602: return AjZb::kCpop;
				case 0x604: return AjZb::kSextB;
				case 0x605: return AjZb::kSextH;
				default: break;
				}
				if (!shamt_ok) return AjZb::kNone;
				switch (i.Itype.high_bits()) {
				case 0x280: return AjZb::kBseti;
				case 0x480: return AjZb::kBclri;
				case 0x680: return AjZb::kBinvi;
				default:    return AjZb::kNone;
				}
			}
			if (i.Itype.funct3 == 0x5) {
				// ORC.B and REV8 carry no shift amount either, so they are
				// matched before the RV32 shamt rule can reject them.
				if (i.Itype.imm == 0x287)         return AjZb::kOrcB;
				if (i.Itype.template is_rev8<W>()) return AjZb::kRev8;
				if (!shamt_ok) return AjZb::kNone;
				if (i.Itype.high_bits() == 0x600) return AjZb::kRori;
				if (i.Itype.high_bits() == 0x480) return AjZb::kBexti;
			}
			return AjZb::kNone;

		case RV64I_OP_IMM32:
			if constexpr (RV64) {
				if (i.Itype.funct3 == 0x1) {
					switch (i.Itype.imm) {
					case 0x600: return AjZb::kClzw;
					case 0x601: return AjZb::kCtzw;
					case 0x602: return AjZb::kCpopw;
					default: break;
					}
					// SLLI.UW is the one OP-IMM-32 form with a full six-bit
					// shift amount, because it shifts a 64-bit value.
					if (i.Itype.high_bits() == 0x080) return AjZb::kSlliUw;
				}
				else if (i.Itype.funct3 == 0x5) {
					// RORIW rotates 32 bits, so its shift amount is five bits
					// wide even here.
					if (i.Itype.high_bits() == 0x600 && (i.Itype.imm & 0x20) == 0)
						return AjZb::kRoriw;
				}
			}
			return AjZb::kNone;

		case RV64I_OP32:
			if constexpr (RV64) {
				switch (i.Rtype.jumptable_friendly_op()) {
				case 0x40:  return AjZb::kAddUw;
				case 0x44:  return AjZb::kPackW;   // ZEXT.H when rs2 == 0
				case 0x102: case 0x104: case 0x106:
					return AjZb::kShAddUw;
				case 0x301: return AjZb::kRolw;
				case 0x305: return AjZb::kRorw;
				default:    break;
				}
			}
			return AjZb::kNone;

		default:
			return AjZb::kNone;
		}
	}

	/// @brief An instruction emitted as a call to the interpreter's handler
	/// instead of a native sequence.
	/// @details A handler call is semantically exact, but slow. It is used where
	/// a native form is too long to be worth it (the atomics, CBO.ZERO), where
	/// the operation is a table lookup or a software float (Zfa, Zfhmin), and
	/// for the vector instructions outside the inlined memory forms. Handlers
	/// reached this way must not modify PC nor stop the machine: translated code
	/// continues at the next instruction.
	enum class AjHandler : uint8_t
	{
		kNone = 0,
		kAtomic,    ///< A: LR/SC and the AMOs
		kCsr,       ///< Zicsr and Zicntr, plus the vector CSRs
		kCboZero,   ///< Zicboz: CBO.ZERO writes a block of zeroes
		kFloat,     ///< Zfa, Zfhmin, and any FP form with no native sequence
		kVector,    ///< V: everything but the inlined memory forms
	};

	/// @brief What an inlined vector transfer moves.
	/// @details Only transfers of a whole number of vector registers are inlined.
	/// The strided, indexed, masked and segmented accesses, the mask transfers,
	/// the fault-only-first loads and any transfer ending in a partial register
	/// are handler calls, as are the forms below when their run-time guard fails.
	enum class AjVecForm : uint8_t
	{
		kNone = 0,
		kUnitStride,      ///< vle<eew>.v / vse<eew>.v: vl*EEW contiguous bytes
		kWholeRegister,   ///< vl<n>re<eew>.v / vs<n>r.v: n registers, no vtype
	};
	struct AjVectorMem
	{
		AjVecForm form = AjVecForm::kNone;
		bool     is_store = false;
		unsigned eew_log2 = 0;   ///< log2(EEW / 8), the unit-stride forms
		unsigned nregs    = 1;   ///< registers moved by a whole-register transfer
		unsigned vreg     = 0;   ///< vd, or vs3 in a store
		unsigned rs1      = 0;   ///< the base address
	};

	/// @brief True for the LOAD-FP/STORE-FP widths that name a vector access.
	/// @details Widths 1-4 are the scalar formats (half, single, double, quad);
	/// the rest are vector EEWs: 0 is EEW 8, and 5, 6 and 7 are 16, 32 and 64.
	inline bool aj_is_vector_width(unsigned width) noexcept
	{
		return width == 0 || width >= 5;
	}

	/// @brief log2(EEW/8) for a unit-stride width, or -1 for the other widths.
	inline int aj_vector_unit_stride_eew(uint32_t width) noexcept
	{
		switch (width) {
		case 0b000: return 0;   // VLE8  / VSE8
		case 0b101: return 1;   // VLE16 / VSE16
		case 0b110: return 2;   // VLE32 / VSE32
		case 0b111: return 3;   // VLE64 / VSE64
		default:    return -1;
		}
	}

	/// @brief Classifies one vector load or store as inlinable, or not.
	/// @details nf, mew, mop, lumop, vm and width are all fixed by the encoding,
	/// leaving only vtype and vl to check at run-time -- and a whole-register
	/// transfer reads neither.
	inline AjVectorMem aj_vector_memory_form(rv32i_instruction i) noexcept
	{
		const rv32v_instruction vi { i };
		AjVectorMem info;
		if (i.opcode() == RV32F_STORE)
			info.is_store = true;
		else if (i.opcode() != RV32F_LOAD)
			return {};
		// The handler traps on every width it does not accept, the scalar
		// widths included: there the mop and lumop bits are part of an
		// immediate instead.
		if (!riscv::vector_extension || !aj_is_vector_width(vi.VL.width))
			return {};
		// mew means EEW > 64, and a mop other than 00 is a strided or indexed
		// access: both belong to the handler.
		if (vi.VL.mew || vi.VL.mop != 0b00)
			return {};
		info.vreg = info.is_store ? vi.VS.vs3 : vi.VL.vd;
		info.rs1  = info.is_store ? vi.VS.rs1 : vi.VL.rs1;
		const unsigned nf = vi.VL.nf + 1;   // fields per segment
		switch (info.is_store ? vi.VS.sumop : vi.VL.lumop) {
		case 0b00000: {   // the plain unit-stride access
			if (nf != 1 || !vi.VL.vm)
				return {};   // segmented, or masked per element
			const int eew = aj_vector_unit_stride_eew(vi.VL.width);
			if (eew < 0)
				return {};
			info.eew_log2 = unsigned(eew);
			info.form = AjVecForm::kUnitStride;
			return info; }
		case 0b01000:     // vl<n>re<eew>.v / vs<n>r.v
			// An illegal nf, or a group not aligned to its own size, must trap:
			// only the handler knows how.
			if (!vi.VL.vm || (nf != 1 && nf != 2 && nf != 4 && nf != 8))
				return {};
			if ((info.vreg % nf) != 0 || info.vreg + nf > 32)
				return {};
			info.nregs = nf;
			info.form = AjVecForm::kWholeRegister;
			return info;
		default:          // the mask transfers, fault-only-first, and the rest
			return {};
		}
	}

	/// @brief True for the CSRs this emulator answers itself.
	/// @details The machine-level CSRs, and any CSR a host program serves from
	/// the unhandled-CSR callback, end the region instead: only a CSR handled
	/// inside libriscv is known not to stop or redirect execution.
	inline bool aj_csr_is_known(uint32_t csr) noexcept
	{
		switch (csr) {
		case 0x001: case 0x002: case 0x003:   // fflags, frm, fcsr
		case 0xC00: case 0xC01: case 0xC02:   // cycle, time, instret
		case 0xC80: case 0xC81: case 0xC82:   // ... and their upper halves
			return true;
		case 0x008: case 0x009: case 0x00A: case 0x00F:   // vstart, vxsat, vxrm, vcsr
		case 0xC20: case 0xC21: case 0xC22:               // vl, vtype, vlenb
			return riscv::vector_extension != 0;
		default:
			return false;
		}
	}

	/// @brief True for Zimop's well-formed may-be-operations, which write zero.
	/// @details Mirrors machine.cpp: MOP.R.n fixes bits 24:22, and MOP.RR.n
	/// sets bit 25 instead.
	inline bool aj_zimop_wellformed(rv32i_instruction i) noexcept
	{
		const uint32_t hi = i.whole >> 25;
		return (hi & 0b1011000) == 0b1000000
			&& ((hi & 1) != 0 || (i.whole & (0b111 << 22)) == (0b111 << 22));
	}

	/// @brief Classifies one instruction as covered by an interpreter handler.
	/// @details Shared by aj_is_emittable() and the emitter, so region discovery
	/// cannot accept an encoding that emission then skips.
	template <int W>
	inline AjHandler aj_handler_classify(rv32i_instruction i) noexcept
	{
		switch (i.opcode())
		{
		case RV32I_FENCE:
			// Zicbom's three hints emit nothing at all; only Zicboz's CBO.ZERO
			// touches memory.
			if (i.Itype.funct3 == RV32I_CBO_FUNCT3 && i.Itype.imm == RV32I_CBO_ZERO)
				return AjHandler::kCboZero;
			return AjHandler::kNone;
		case RV32I_SYSTEM:
			// funct3 0 is ECALL/EBREAK/WFI, which redirect execution, and
			// funct3 4 is Zimop, which is emitted natively.
			if (i.Itype.funct3 != 0x0 && i.Itype.funct3 != 0x4
				&& aj_csr_is_known(i.Itype.imm))
				return AjHandler::kCsr;
			return AjHandler::kNone;
		case RV32A_ATOMIC:
			return riscv::atomics_enabled ? AjHandler::kAtomic : AjHandler::kNone;
		case RV32F_LOAD:
		case RV32F_STORE:
			// funct3 1 is Zfhmin's half-precision access, and quad is not
			// implemented. Every other width names a vector access.
			if (i.Itype.funct3 == 0x1)
				return AjHandler::kFloat;
			if (aj_is_vector_width(i.Itype.funct3))
				return riscv::vector_extension ? AjHandler::kVector : AjHandler::kNone;
			return AjHandler::kNone;
		case RV32V_OP:
			// The arithmetic, the reductions, the permutes and vsetvl: the
			// handler owns the vl/vtype/mask machinery, and the backend's part
			// is to keep the region from ending here.
			return riscv::vector_extension ? AjHandler::kVector : AjHandler::kNone;
		case RV32F_FMADD:
		case RV32F_FMSUB:
		case RV32F_FNMADD:
		case RV32F_FNMSUB:
		case RV32F_FPFUNC:
			// Everything aj_fp_native() rejects: Zfa, Zfhmin's conversions and
			// moves, the fused multiply-adds on a host without FMA, and the
			// reserved encodings, which trap as they do in the interpreter.
			return AjHandler::kFloat;
		default:
			return AjHandler::kNone;
		}
	}

	/// @brief True when this F/D encoding has a native form in the emitter.
	/// @details Split out of aj_is_emittable() because the emitter needs the same
	/// answer: what this rejects is still emittable as a handler call, and the
	/// two paths must agree on which is which.
	/// @tparam W The guest register width. Only RV64 has the 64-bit integer
	/// conversions and the moves of a whole double into an integer register.
	template <int W>
	inline bool aj_fp_native(rv32i_instruction i) noexcept
	{
		if (!aj_fpu_emittable)
			return false;
		switch (i.opcode())
		{
		case RV32F_LOAD:   // FLW, FLD
		case RV32F_STORE:  // FSW, FSD
			// funct3 is the access width, in the same bits either way. Both
			// guest widths carry FLD/FSD: RV32D moves doubles through a machine
			// whose integer registers are half as wide.
			return i.Itype.funct3 == 0x2 || i.Itype.funct3 == 0x3;
		case RV32F_FMADD:
		case RV32F_FMSUB:
		case RV32F_FNMADD:
		case RV32F_FNMSUB:
			// funct2 is the precision: 0 is single, 1 is double, and the
			// half- and quad-precision encodings are not implemented at all.
			return aj_host_has_fma() && rv32f_instruction(i).R4type.funct2 <= 0x1;
		case RV32F_FPFUNC: {
			const rv32f_instruction fi { i };
			if (fi.R4type.funct2 > 0x1)
				return false;
			const bool is_double = (fi.R4type.funct2 == 0x1);
			switch (i.fpfunc()) {
			case RV32F__FADD: case RV32F__FSUB:
			case RV32F__FMUL: case RV32F__FDIV:
				return true;
			case RV32F__FSQRT:
				// rs2 is not a register here; every value but 0 is reserved.
				return fi.R4type.rs2 == 0x0;
			case RV32F__FCVT_SD_DS:
				// rs2 names the *source* format, where funct2 names the
				// destination: 0 is single, 1 is double, 2 is half (Zfhmin) and
				// 4/5 are Zfa's FROUND/FROUNDNX, which are not conversions at
				// all. Only a real S<->D conversion has a native form.
				return fi.R4type.rs2 <= 0x1 && fi.R4type.rs2 != fi.R4type.funct2;
			case RV32F__FSGNJ_NX:
				return fi.R4type.funct3 <= 0x2;
			case RV32F__FMIN_MAX:
				// funct3 2 and 3 are Zfa's FMINM/FMAXM, which differ over NaN.
				return fi.R4type.funct3 <= 0x1;
			case RV32F__FEQ_LT_LE:
				// funct3 4 and 5 are Zfa's quiet FLEQ/FLTQ.
				return fi.R4type.funct3 <= 0x2;
			case RV32F__FCVT_W_SD:   // FCVT.{W,WU,L,LU}.{S,D}
			case RV32F__FCVT_SD_W:   // FCVT.{S,D}.{W,WU,L,LU}
				// The 64-bit integer forms only exist on RV64, and rs2 8 is
				// Zfa's FCVTMOD.W.D.
				return fi.R4type.rs2 <= (W == 8 ? 0x3u : 0x1u);
			case RV32F__FMV_X_W:
				// funct3 0 is FMV.X.W / FMV.X.D, funct3 1 is FCLASS. Only the
				// move of a whole double into an integer register needs RV64;
				// FCLASS reads the bit fields and fits a 32-bit result.
				// rs2 must be 0: Zfa reuses rs2 1 for FMVH.X.D.
				if (fi.R4type.rs2 != 0x0)
					return false;
				if (fi.R4type.funct3 == 0x1)
					return true;
				return fi.R4type.funct3 == 0x0 && (!is_double || W == 8);
			case RV32F__FMV_W_X:
				// rs2 must be 0: Zfa's FLI.{S,D} shares this funct7 with rs2 1,
				// and reads a constant table index out of rs1 rather than a
				// register. Emitting it as FMV.W.X silently loads garbage.
				return fi.R4type.rs2 == 0x0
					&& fi.R4type.funct3 == 0x0 && (!is_double || W == 8);
			default:
				return false;
			}
		}
		default:
			return false;
		}
	}

	/// @brief True if the instruction can be emitted by the asmjit backend.
	/// Anything that is false terminates the region.
	/// @details This predicate is shared between region discovery and emission,
	/// so that the two can never disagree about where a region ends. It must
	/// stay an exact mirror of the opcodes handled in aj_emit.cpp.
	/// @tparam W The guest register width. RV64 adds LD/LWU/SD and the *W
	/// opcodes, widens the shift amount that a shift-immediate may carry, and
	/// is the only width with the 64-bit FCVT forms and FMV.X.D/FMV.D.X.
	template <int W>
	inline bool aj_is_emittable(rv32i_instruction i) noexcept
	{
		// The shift-amount field is 5 bits on RV32 and 6 bits on RV64, so the
		// number of bits left over to identify the operation differs.
		constexpr uint32_t SHMASK = (W == 8) ? 0xFC0 : 0xFE0;

		switch (i.opcode())
		{
		case Dyncall::opcode:
			return Dyncall::valid(i.whole);
		case RV32I_LUI:
		case RV32I_AUIPC:
		case RV32I_JAL:
			return true;
		case RV32I_FENCE:
			// MISC-MEM carries more than FENCE: funct3 1 is FENCE.I, which the
			// interpreter uses to invalidate stale execute segments and must
			// keep, and funct3 2 is the Zicbo group. Only plain FENCE becomes
			// a barrier.
			if (i.Itype.funct3 == 0x0)
				return true;
			if (i.Itype.funct3 == RV32I_CBO_FUNCT3) {
				// CBO.INVAL, CBO.CLEAN and CBO.FLUSH have no cache to act on
				// in an emulator, and so are emitted as nothing.
				switch (i.Itype.imm) {
				case RV32I_CBO_INVAL: case RV32I_CBO_CLEAN: case RV32I_CBO_FLUSH:
					return true;
				default:
					break;
				}
			}
			return aj_handler_classify<W>(i) != AjHandler::kNone;
		case RV32I_SYSTEM:
			// Zawrs may return immediately, and libriscv does, so WRS.NTO and
			// WRS.STO are no-ops. The rest of funct3 0 -- ECALL, EBREAK, WFI,
			// STOP -- ends the region.
			if (i.Itype.funct3 == 0x0)
				return i.Itype.imm == 0x00D || i.Itype.imm == 0x01D;
			// Zimop writes zero to rd, which is one instruction.
			if (i.Itype.funct3 == 0x4)
				return aj_zimop_wellformed(i);
			return aj_handler_classify<W>(i) != AjHandler::kNone;
		case RV32I_JALR:
			return i.Itype.funct3 == 0x0;
		case RV32I_BRANCH:
			switch (i.Btype.funct3) {
			case 0x0: case 0x1: case 0x4: case 0x5: case 0x6: case 0x7:
				return true;
			default: // 0x2 and 0x3 are not valid branch encodings
				return false;
			}
		case RV32I_LOAD:
			switch (i.Itype.funct3) {
			case 0x0: case 0x1: case 0x2: case 0x4: case 0x5:
				return true;   // LB LH LW LBU LHU
			case 0x3: case 0x6:
				return W == 8; // LD, LWU
			default:
				return false;
			}
		case RV32I_STORE:
			return i.Stype.funct3 <= (W == 8 ? 0x3u : 0x2u);   // SB SH SW (SD)
		case RV32I_OP_IMM:
			switch (i.Itype.funct3) {
			case 0x1: // SLLI, or one of the Zb* encodings sharing funct3
				if ((i.Itype.imm & SHMASK) == 0x000)
					return true;
				break;
			case 0x5: // SRLI / SRAI, or RORI and friends
				if ((i.Itype.imm & SHMASK) == 0x000
					|| (i.Itype.imm & SHMASK) == 0x400)
					return true;
				break;
			default: // ADDI, SLTI, SLTIU, XORI, ORI, ANDI
				return true;
			}
			return aj_zb_classify<W>(i) != AjZb::kNone;
		case RV32I_OP:
			if (i.Rtype.funct7 == 0b0000000)
				return true;  // ADD SLL SLT SLTU XOR SRL OR AND
			if (i.Rtype.funct7 == 0b0100000
				&& (i.Rtype.funct3 == 0x0 || i.Rtype.funct3 == 0x5))
				return true;  // SUB, SRA -- the rest of funct7 0x20 is Zbb
			if (i.Rtype.funct7 == 0b0000001)
				return true;  // M: MUL MULH MULHSU MULHU DIV DIVU REM REMU
			return aj_zb_classify<W>(i) != AjZb::kNone;
		case RV64I_OP_IMM32:
			if constexpr (W == 8) {
				switch (i.Itype.funct3) {
				case 0x0: // ADDIW
					return true;
				case 0x1: // SLLIW, always a 5-bit shift amount
					if ((i.Itype.imm & 0xFE0) == 0x000)
						return true;
					break;
				case 0x5: // SRLIW / SRAIW
					if ((i.Itype.imm & 0xFE0) == 0x000
						|| (i.Itype.imm & 0xFE0) == 0x400)
						return true;
					break;
				default:
					break;
				}
				return aj_zb_classify<W>(i) != AjZb::kNone;
			}
			return false;
		case RV64I_OP32:
			if constexpr (W == 8) {
				if (i.Rtype.funct7 == 0b0000000
					&& (i.Rtype.funct3 == 0x0     // ADDW
						|| i.Rtype.funct3 == 0x1  // SLLW
						|| i.Rtype.funct3 == 0x5))// SRLW
					return true;
				if (i.Rtype.funct7 == 0b0100000
					&& (i.Rtype.funct3 == 0x0     // SUBW
						|| i.Rtype.funct3 == 0x5))// SRAW
					return true;
				if (i.Rtype.funct7 == 0b0000001
					&& (i.Rtype.funct3 == 0x0     // MULW
						|| i.Rtype.funct3 >= 0x4))// DIVW DIVUW REMW REMUW
					return true;
				return aj_zb_classify<W>(i) != AjZb::kNone;
			}
			return false;

		// --- F/D extension ---
		// Everything without a native form -- Zfa, Zfhmin's half-precision
		// accesses, conversions and moves, the fused multiply-adds on a host
		// without FMA, and the reserved encodings -- is emitted as a call to
		// the interpreter's handler instead.
		case RV32F_LOAD:
		case RV32F_STORE:
		case RV32F_FMADD:
		case RV32F_FMSUB:
		case RV32F_FNMADD:
		case RV32F_FNMSUB:
		case RV32F_FPFUNC:
			return aj_fp_native<W>(i)
				|| aj_handler_classify<W>(i) != AjHandler::kNone;

		default:
			// The A and V opcodes reach the handler classifier, which rejects
			// them when their extension is disabled. Everything left over ends
			// the region, and the interpreter resumes at the exit address.
			return aj_handler_classify<W>(i) != AjHandler::kNone;
		}
	}
	template <int W>
	inline bool aj_is_emittable(const AjDecoded& d) noexcept
	{
		return d.valid && aj_is_emittable<W>(d.instr);
	}

	/// @brief True when the instruction never falls through to the next address.
	inline bool aj_is_unconditional_jump(rv32i_instruction i) noexcept
	{
		return i.opcode() == RV32I_JAL || i.opcode() == RV32I_JALR;
	}

	/// @brief Emits one region as a single host function.
	/// @param entries The guest addresses the emitted function may be entered at,
	/// ascending and non-empty. More than one means the prologue dispatches on the
	/// entry PC that the interpreter left in AjState::pc.
	/// @param instrs The region's reachable instruction addresses, ascending.
	/// @return nullptr if the region could not be emitted for any reason.
	/// @details Defined in aj_emit.cpp.
	template <int W>
	aj_block_func<W> aj_emit_region(AjCode&, const MachineOptions<W>&,
		const DecodedExecuteSegment<W>&, const AjInfo<W>&,
		const aj_block_func<W>* mappings,
		const std::vector<std::pair<address_type<W>, size_t>>& call_targets,
		const std::vector<address_type<W>>& entries,
		const std::vector<address_type<W>>& instrs);
}
