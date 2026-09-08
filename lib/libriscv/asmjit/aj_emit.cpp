#include "aj_emit.hpp"
#include "aj_runtime.hpp"
#include "../cpu.hpp"
#include "../machine.hpp"
#include "../decoded_exec_segment.hpp"
#include "../memory.hpp"
#include "../riscvbase.hpp"
#include "../rvfd_util.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <set>
#include <unordered_map>
#include <vector>

namespace riscv
{
#if RISCV_ASMJIT_HAS_BACKEND

using namespace asmjit;
using namespace asmjit::ujit;

// --- host escapes ---------------------------------------------------------
// The only host-specific code; everything else goes through asmjit's universal compiler.

/// @brief A full memory barrier, matching what the interpreter gives FENCE.
static inline void host_fence(BackendCompiler& cc)
{
#if defined(ASMJIT_UJIT_X86)
	cc.mfence();
#else
	cc.dmb(a64::Predicate::DB::kSY);
#endif
}

/// @brief Sign-extend 32-bit src to 64-bit dst (*W instruction result).
static inline void host_sign_extend_word(BackendCompiler& cc, const Gp& dst, const Gp& src)
{
#if defined(ASMJIT_UJIT_X86)
	cc.movsxd(dst, src);
#else
	cc.sxtw(dst, src);
#endif
}

/// @brief True when CLZ/CTZ return XLEN for zero input (lzcnt/tzcnt do; bsr/bsf don't).
static inline bool host_count_zeros_is_exact(const UniCompiler& uc, bool leading)
{
#if defined(ASMJIT_UJIT_X86)
	return leading ? uc.has_lzcnt() : uc.has_bmi();
#else
	(void)uc; (void)leading;
	return true;
#endif
}

/// @brief dst = popcount(src). Uses POPCNT when available, else SWAR reduction.
static inline void host_popcount(UniCompiler& uc, const Gp& dst, const Gp& src)
{
#if defined(ASMJIT_UJIT_X86)
	if (uc.has_popcnt()) {
		uc.cc->popcnt(dst, src);
		return;
	}
#endif
	const uint32_t width = uint32_t(dst.size()) * 8;
	const bool wide = (width == 64);
	const uint64_t m1 = wide ? 0x5555555555555555ull : 0x55555555ull;
	const uint64_t m2 = wide ? 0x3333333333333333ull : 0x33333333ull;
	const uint64_t m4 = wide ? 0x0F0F0F0F0F0F0F0Full : 0x0F0F0F0Full;

	Gp t = uc.new_similar_reg(dst, "pcnt");
	Gp u = uc.new_similar_reg(dst, "pcnt2");
	uc.shr (t, src, Imm(1));
	uc.and_(t, t, Imm(m1));
	uc.sub (t, src, t);            // t: two-bit counts
	uc.shr (u, t, Imm(2));
	uc.and_(u, u, Imm(m2));
	uc.and_(t, t, Imm(m2));
	uc.add (t, t, u);              // t: four-bit counts
	uc.shr (u, t, Imm(4));
	uc.add (t, t, u);
	uc.and_(t, t, Imm(m4));        // t: one count per byte, each at most 8
	// Fold bytes: total <= 64 so no carry past the low byte.
	for (uint32_t s = 8; s < width; s *= 2) {
		uc.shr(u, t, Imm(s));
		uc.add(t, t, u);
	}
	uc.and_(dst, t, Imm(0xFF));
}

/// @brief dst = high XLEN bits of 2*XLEN product a*b.
static inline void host_mul_hi(UniCompiler& uc, const Gp& dst, const Gp& a, const Gp& b, bool is_signed)
{
	BackendCompiler& cc = *uc.cc;
#if defined(ASMJIT_UJIT_X86)
	// xDX:xAX <- xAX * src; allocator pins the virtual pair.
	Gp lo = uc.new_similar_reg(dst, "mullo");
	Gp hi = uc.new_similar_reg(dst, "mulhi");
	cc.mov(lo, a);
	if (is_signed) cc.imul(hi, lo, b); else cc.mul(hi, lo, b);
	cc.mov(dst, hi);
#else
	if (dst.size() == 8) {
		if (is_signed) cc.smulh(dst, a, b); else cc.umulh(dst, a, b);
	} else {
		// AArch64: widening 32x32→64 multiply, take upper half.
		Gp t = uc.new_gp64("mulw");
		if (is_signed) cc.smull(t, a, b); else cc.umull(t, a, b);
		uc.shr(t, t, Imm(32));
		uc.mov(dst, t.r32());
	}
#endif
}

/// @brief True when the host has PCLMULQDQ (x86) for Zbc carry-less multiply.
static inline bool host_has_clmul(const UniCompiler& uc)
{
#if defined(ASMJIT_UJIT_X86)
	return uc.has_pclmulqdq();
#else
	(void)uc;
	return false;
#endif
}

/// @brief 128-bit carry-less product of two 64-bit values. Only valid when host_has_clmul().
static inline void host_clmul64(UniCompiler& uc, const Gp& lo, const Gp& hi,
	const Gp& a, const Gp& b)
{
#if defined(ASMJIT_UJIT_X86)
	BackendCompiler& cc = *uc.cc;
	Vec va = uc.new_vec128("clmula");
	Vec vb = uc.new_vec128("clmulb");
	uc.s_mov_u64(va, a);
	uc.s_mov_u64(vb, b);
	// imm=0: multiply low halves of both operands.
	if (uc.has_avx()) cc.vpclmulqdq(va, va, vb, Imm(0));
	else              cc.pclmulqdq(va, vb, Imm(0));
	uc.s_mov_u64(lo, va);
	// Shift down: s_extract_u64() always returns the low lane on AVX.
	Vec vh = uc.new_vec128("clmulh");
	uc.v_srlb_u128(vh, va, 8);
	uc.s_mov_u64(hi, vh);
#else
	(void)uc; (void)lo; (void)hi; (void)a; (void)b;
#endif
}

/// @brief True when the host has hardware FMA (required for correctly-rounded FMADD).
static inline bool host_has_fma(const UniCompiler& uc)
{
#if defined(ASMJIT_UJIT_X86)
	return uc.has_fma();
#else
	(void)uc;
	return true;   // AArch64 has FMADD/FMSUB in the base ISA
#endif
}

/// @brief Signed div/rem; caller has excluded b==0 and b==-1 (the two idiv faults).
static inline void host_sdiv(UniCompiler& uc, const Gp& dst, const Gp& a, const Gp& b, bool want_rem)
{
	BackendCompiler& cc = *uc.cc;
#if defined(ASMJIT_UJIT_X86)
	// idiv requires sign-extending the dividend into xDX:xAX.
	Gp quot = uc.new_similar_reg(dst, "quot");
	Gp rem  = uc.new_similar_reg(dst, "rem");
	cc.mov(quot, a);
	if (quot.size() == 8) cc.cqo(rem, quot); else cc.cdq(rem, quot);
	cc.idiv(rem, quot, b);
	cc.mov(dst, want_rem ? rem : quot);
#else
	if (!want_rem) {
		cc.sdiv(dst, a, b);
	} else {
		Gp quot = uc.new_similar_reg(dst, "quot");
		cc.sdiv(quot, a, b);
		cc.msub(dst, quot, b, a);   // dst = a - quot * b
	}
#endif
}

bool aj_host_has_fma() noexcept
{
	// Queries host features the same way the emitter's UniCompiler does.
	static const bool has_fma = [] {
		const CpuFeatures& f = CpuInfo::host().features();
#if defined(ASMJIT_UJIT_X86)
		return f.x86().has_fma();
#else
		(void)f;
		return true;
#endif
	}();
	return has_fma;
}

template <int W>
struct AjEmitter
{
	using address_t = address_type<W>;
	static_assert(W == 4 || W == 8, "The asmjit backend supports RV32 and RV64");
	static constexpr int  RVLEN = sizeof(address_t);
	static constexpr bool RV64  = (W == 8);

	UniCompiler& uc;
	BackendCompiler& cc;  // == *uc.cc, used only by the host escapes above
	const uint8_t* seg;   // PC-relative pointer to the execute segment
	const std::vector<address_t>& entries;  // entry addresses, ascending, non-empty
	const aj_block_func<W>* mappings;       // stable, fully sized mapping table
	const std::vector<std::pair<address_t, size_t>>& call_targets;
	address_t entry;      // entries.front(), the lowest address it is entered at
	const std::vector<address_t>& instrs;   // reachable addresses, ascending
	address_t seg_end;    // execute segment end, for instruction reads
	const AjInfo<W>& info;

	Gp cpu;               // arg0: CPU<W>*
	Gp st;                // arg1: AjState<W>*
	Gp counter;           // live instruction counter, always 64-bit
	Gp zero;              // a constant zero, standing in for x0
	Gp arena;             // base of the flat memory arena, when inlining
	Gp read_bound;        // RV64 bounds that do not fit a compare immediate
	Gp write_bound;

	std::array<Gp, 32> vreg {};
	std::bitset<32> writeset;   // static: every rd the region writes
	std::bitset<32> readset;    // static: writeset + every rs1/rs2 it reads
	bool needs_zero = false;
	bool needs_arena = false;

	// f-registers: same eager preload/writeback as integer, but f0 is ordinary.
	std::array<Vec, 32> fvreg {};
	std::bitset<32> fp_writeset;
	std::bitset<32> fp_readset;
	Vec canon32, canon64;       // the canonical NaNs, when a conversion needs them
	bool needs_canon32 = false;
	bool needs_canon64 = false;

	std::set<address_t> branch_targets;         // in-region targets
	std::unordered_map<address_t, Label> labels;
	uint32_t pending = 0;       // instructions retired since the last counter flush
	Label exit_tail;            // one writeback/return sequence per region
	Gp exit_pc;
	Gp exit_counter;
	bool has_exit = false;

	// Cold paths emitted after the body; each captures its `pending` at branch-off.
	std::vector<std::function<void()>> deferred;

	AjEmitter(UniCompiler& u, const uint8_t* s, const std::vector<address_t>& ens,
		const aj_block_func<W>* maps,
		const std::vector<std::pair<address_t, size_t>>& targets,
		const std::vector<address_t>& list, address_t se, const AjInfo<W>& in)
		: uc(u), cc(*u.cc), seg(s), entries(ens), mappings(maps), call_targets(targets),
		  entry(ens.front()),
		  instrs(list), seg_end(se), info(in) {}

	// Entry and emit order differ when a back-edge reaches below the entry.
	bool entry_is_first() const noexcept { return entry == instrs.front(); }
	// Multiple entries require a PC-based dispatch in the prologue.
	bool needs_entry_dispatch() const noexcept { return entries.size() > 1; }
	// Target must be in the emitted instruction set, not just in [begin, end).
	bool in_region(address_t pc) const noexcept {
		return std::binary_search(instrs.begin(), instrs.end(), pc);
	}
	Label label_at(address_t pc) { return labels.at(pc); }

	// --- width-parametric primitives ---
	// Guest-width host registers give RISC-V wrapping/shift semantics for free.
	Gp new_ireg(const char* name) {
		if constexpr (RV64) return uc.new_gp64(name); else return uc.new_gp32(name);
	}
	Gp new_ireg(const char* fmt, unsigned i) {
		if constexpr (RV64) return uc.new_gp64(fmt, i); else return uc.new_gp32(fmt, i);
	}
	/// @brief A full-width guest constant: an address, a link target, LUI's result.
	static Imm rvimm(address_t v) {
		if constexpr (RV64) return Imm(uint64_t(v)); else return Imm(uint32_t(v));
	}
	/// @brief A sign-extended constant, which is what every I-type immediate is.
	static Imm sximm(int32_t v) { return Imm(int64_t(v)); }

	Mem reg_mem(unsigned i) const {
		return mem_ptr(cpu, info.reg_offset + int32_t(i * RVLEN));
	}
	/// @brief f-registers are 64-bit on both RV32 and RV64 (RV32D needs full-width doubles).
	Mem freg_mem(unsigned i) const {
		return mem_ptr(cpu, info.fpreg_offset + int32_t(i * 8));
	}
	static constexpr int32_t off_counter() { return int32_t(offsetof(AjState<W>, counter)); }
	static constexpr int32_t off_max()     { return int32_t(offsetof(AjState<W>, max_counter)); }
	static constexpr int32_t off_pc()      { return int32_t(offsetof(AjState<W>, pc)); }
	static constexpr int32_t off_depth()   { return int32_t(offsetof(AjState<W>, call_depth)); }

	// --- register cache ---
	// Eager preload, not lazy: a lazy load after a label would clobber loop-carried values on back-edges.
	void emit_prologue_loads() {
		if (needs_zero) {
			zero = new_ireg("zero");
			uc.mov(zero, Imm(0));
		}
		if (needs_arena) {
			arena = uc.new_gp_ptr("arena");
			uc.load(arena, mem_ptr(cpu, info.arena_ptr));
		}
		if constexpr (RV64) {
			if (arena_is_checked() && info.arena_rdbound_value > uint64_t(INT32_MAX)) {
				read_bound = new_ireg("rdbound");
				uc.mov(read_bound, Imm(info.arena_rdbound_value));
			}
			if (arena_is_checked() && info.arena_wrbound_value > uint64_t(INT32_MAX)) {
				write_bound = new_ireg("wrbound");
				uc.mov(write_bound, Imm(info.arena_wrbound_value));
			}
		}
		// Canonical NaNs are loop-invariant; materialized once here.
		if (needs_canon32) {
			Gp t = uc.new_gp32("canon32i");
			uc.mov(t, Imm(CANONICAL_NAN_F32));
			canon32 = uc.new_vec128("canon32");
			uc.s_mov_u32(canon32, t);
		}
		if (needs_canon64) {
			Gp t = uc.new_gp64("canon64i");
			uc.mov(t, Imm(CANONICAL_NAN_F64));
			canon64 = uc.new_vec128("canon64");
			uc.s_mov_u64(canon64, t);
		}
		for (unsigned i = 1; i < 32; i++) {
			if (!readset[i]) continue;
			vreg[i] = new_ireg("x%u", i);
			uc.load(vreg[i], reg_mem(i));
		}
		for (unsigned i = 0; i < 32; i++) {
			if (!fp_readset[i]) continue;
			fvreg[i] = uc.new_vec128("f%u", i);
			uc.v_loadu64_u64(fvreg[i], freg_mem(i));
		}
	}
	void reload_regs() {
		for (unsigned i = 1; i < 32; i++)
			if (readset[i]) uc.load(vreg[i], reg_mem(i));
		for (unsigned i = 0; i < 32; i++)
			if (fp_readset[i]) uc.v_loadu64_u64(fvreg[i], freg_mem(i));
	}
	Gp get(unsigned i) {               // source register
		if (i == 0) return zero;       // never written, so it can be shared
		return vreg[i];                // guaranteed loaded by the prologue
	}
	Gp def(unsigned i) {               // destination register
		if (i == 0) return new_ireg("sink"); // writes to x0 are discarded
		return vreg[i];                // in writeset, so the prologue loaded it
	}
	// f-registers have no x0 special case; both names exist for readability.
	Vec fget(unsigned i) { return fvreg[i]; }
	Vec fdef(unsigned i) { return fvreg[i]; }

	// --- counter accounting ---
	// `pending` counts retired instructions since the last flush. Must be zero
	// at every merge point (before binding a target label or emitting a branch).
	void flush_counter() {
		if (info.ignore_instruction_limit) { pending = 0; return; }
		if (pending) { uc.add(counter, counter, Imm(pending)); pending = 0; }
	}

	// --- exits ---
	// Stores the full static write-set, not a dirty set: an early exit in a loop
	// must include registers written by the previous iteration.
	void flush_regs() {
		for (unsigned i = 1; i < 32; i++)
			if (writeset[i]) uc.store(reg_mem(i), vreg[i]);
		for (unsigned i = 0; i < 32; i++)
			if (fp_writeset[i]) uc.v_storeu64_u64(freg_mem(i), fvreg[i]);
	}
	void store_pc(address_t next_pc) {
		Gp t = new_ireg("nextpc");
		uc.mov(t, rvimm(next_pc));
		uc.store(mem_ptr(st, off_pc()), t);
	}
	void ensure_exit_tail() {
		if (has_exit) return;
		has_exit = true;
		exit_tail = uc.new_label();
		exit_pc = new_ireg("exitpc");
		if (!info.ignore_instruction_limit)
			exit_counter = uc.new_gp64("exitcnt");
	}
	void set_exit_counter(uint32_t pend) {
		if (info.ignore_instruction_limit) return;
		if (pend == 0) uc.mov(exit_counter, counter);
		else           uc.add(exit_counter, counter, Imm(pend));
	}
	// Select the exit PC/count locally and branch to the region's single cold
	// writeback tail. This must not mutate counter/pending because branch
	// fall-through may share them.
	void emit_exit(address_t next_pc, uint32_t pend) {
		ensure_exit_tail();
		set_exit_counter(pend);
		uc.mov(exit_pc, rvimm(next_pc));
		uc.j(exit_tail);
	}
	// The JALR form: the next PC is only known at run time.
	void emit_exit_reg(const Gp& next_pc, uint32_t pend) {
		ensure_exit_tail();
		set_exit_counter(pend);
		uc.mov(exit_pc, next_pc);
		uc.j(exit_tail);
	}
	// Propagate a callee-selected PC/counter through this native caller.
	void emit_exit_state() {
		ensure_exit_tail();
		if (!info.ignore_instruction_limit)
			uc.load(exit_counter, mem_ptr(st, off_counter()));
		uc.load(exit_pc, mem_ptr(st, off_pc()));
		uc.j(exit_tail);
	}
	void emit_exit_tail() {
		if (!has_exit) return;
		uc.bind(exit_tail);
		flush_regs();
		if (!info.ignore_instruction_limit)
			uc.store(mem_ptr(st, off_counter()), exit_counter);
		uc.store(mem_ptr(st, off_pc()), exit_pc);
		uc.ret();
	}
	// Reloads max so a faulting helper's zero breaks the loop.
	void emit_backedge_check(address_t target) {
		if (info.ignore_instruction_limit) return;
		Label ok = uc.new_label();
		uc.j(ok, ucmp_lt(counter, mem_ptr(st, off_max())));
		emit_exit(target, 0);
		uc.bind(ok);
	}

	bool emit_direct_call(address_t target, address_t return_pc)
	{
		if (!info.direct_calls || info.direct_call_depth == 0)
			return false;
		const auto it = std::lower_bound(call_targets.begin(), call_targets.end(), target,
			[] (const auto& item, address_t pc) { return item.first < pc; });
		if (it == call_targets.end() || it->first != target)
			return false;

		Label fallback = uc.new_label();
		Label resume = uc.new_label();
		Label propagate = uc.new_label();
		Gp fnptr = uc.new_gp_ptr("callee");
		uc.mov(fnptr, Imm(uint64_t(uintptr_t(mappings + it->second))));
		uc.load(fnptr, mem_ptr(fnptr));
		uc.j(fallback, test_z(fnptr));

		Gp depth = uc.new_gp32("depth");
		uc.load(depth, mem_ptr(st, off_depth()));
		uc.j(fallback, ucmp_ge(depth, Imm(info.direct_call_depth)));
		uc.add(depth, depth, Imm(1));
		uc.store(mem_ptr(st, off_depth()), depth);

		flush_regs();
		if (!info.ignore_instruction_limit)
			uc.store(mem_ptr(st, off_counter()), counter);
		store_pc(target);

		InvokeNode* node;
		cc.invoke(Out(node), fnptr, FuncSignature::build<void, void*, void*>());
		node->set_arg(0, cpu);
		node->set_arg(1, st);

		Gp returned_depth = uc.new_gp32("depthret");
		uc.load(returned_depth, mem_ptr(st, off_depth()));
		uc.sub(returned_depth, returned_depth, Imm(1));
		uc.store(mem_ptr(st, off_depth()), returned_depth);
		reload_regs();
		if (!info.ignore_instruction_limit)
			uc.load(counter, mem_ptr(st, off_counter()));

		Gp maxc = uc.new_gp64("callmax");
		uc.load(maxc, mem_ptr(st, off_max()));
		uc.j(propagate, test_z(maxc));
		// A straight-line callee can return without encountering its own
		// back-edge check. Do the dispatcher's limit check here before the
		// caller is allowed to resume.
		if (!info.ignore_instruction_limit)
			uc.j(propagate, ucmp_ge(counter, maxc));
		Gp returned_pc = new_ireg("callpc");
		uc.load(returned_pc, mem_ptr(st, off_pc()));
		uc.j(propagate, cmp_ne(returned_pc, rvimm(return_pc)));
		uc.j(resume);

		uc.bind(propagate);
		emit_exit_state();
		uc.bind(fallback);
		emit_exit(target, 0);
		uc.bind(resume);
		return true;
	}

	// --- ALU helpers ---
	// UniCompiler resolves dst/src aliasing, so RISC-V three-operand forms map directly.
	enum Op { OP_ADD, OP_SUB, OP_AND, OP_OR, OP_XOR };

	void alu_rr(Op op, const Gp& d, const Gp& a, const Gp& b) {
		switch (op) {
		case OP_ADD: uc.add (d, a, b); break;
		case OP_SUB: uc.sub (d, a, b); break;
		case OP_AND: uc.and_(d, a, b); break;
		case OP_OR:  uc.or_ (d, a, b); break;
		case OP_XOR: uc.xor_(d, a, b); break;
		}
	}
	void alu_ri(Op op, const Gp& d, const Gp& a, int32_t v) {
		// Identity immediates (ADDI/ORI/XORI 0, ANDI -1) fold to a move.
		if ((v == 0 && (op == OP_ADD || op == OP_SUB || op == OP_OR || op == OP_XOR))
			|| (v == -1 && op == OP_AND)) {
			if (d.id() != a.id()) uc.mov(d, a);
			return;
		}
		switch (op) {
		case OP_ADD: uc.add (d, a, sximm(v)); break;
		case OP_SUB: uc.sub (d, a, sximm(v)); break;
		case OP_AND: uc.and_(d, a, sximm(v)); break;
		case OP_OR:  uc.or_ (d, a, sximm(v)); break;
		case OP_XOR: uc.xor_(d, a, sximm(v)); break;
		}
	}
	// 0=SLL, 1=SRL, 2=SRA. Both hosts mask the shift count to XLEN-1.
	void shift_rr(int kind, const Gp& d, const Gp& a, const Gp& b) {
		switch (kind) {
		case 0: uc.shl(d, a, b); break;
		case 1: uc.shr(d, a, b); break;
		case 2: uc.sar(d, a, b); break;
		}
	}
	void shift_ri(int kind, const Gp& d, const Gp& a, uint32_t shamt) {
		switch (kind) {
		case 0: uc.shl(d, a, Imm(shamt)); break;
		case 1: uc.shr(d, a, Imm(shamt)); break;
		case 2: uc.sar(d, a, Imm(shamt)); break;
		}
	}
	void setcc_rr(bool is_unsigned, unsigned rd, unsigned rs1, unsigned rs2) {
		const Gp a = get(rs1), b = get(rs2);
		uc.select(def(rd), Imm(1), Imm(0),
			is_unsigned ? ucmp_lt(a, b) : scmp_lt(a, b));
	}
	void setcc_ri(bool is_unsigned, unsigned rd, unsigned rs1, int32_t v) {
		const Gp a = get(rs1);
		// SLTIU compares against the *sign-extended* immediate, unsigned.
		uc.select(def(rd), Imm(1), Imm(0),
			is_unsigned ? ucmp_lt(a, sximm(v)) : scmp_lt(a, sximm(v)));
	}

	// --- M extension ---
	// Division by zero and signed overflow are defined, not trapping.
	// `b + 1 <=u 1` catches both b==0 and b==-1 in one branch.
	void emit_div(const Gp& d, const Gp& a, const Gp& b, bool want_rem, bool is_signed)
	{
		Label slow = uc.new_label(), done = uc.new_label();
		if (is_signed) {
			Gp guard = uc.new_similar_reg(d, "divguard");
			uc.add(guard, b, Imm(1));
			uc.j(slow, ucmp_le(guard, Imm(1)));
			host_sdiv(uc, d, a, b, want_rem);
		} else {
			uc.j(slow, cmp_eq(b, Imm(0)));
			if (want_rem) uc.umod(d, a, b); else uc.udiv(d, a, b);
		}
		uc.j(done);

		uc.bind(slow);
		if (is_signed) {
			Label divzero = uc.new_label();
			uc.j(divzero, cmp_eq(b, Imm(0)));
			// b == -1: a / -1 = -a (also handles MIN / -1 = MIN). a % -1 = 0.
			if (want_rem) uc.mov(d, Imm(0));
			else          uc.neg(d, a);
			uc.j(done);
			uc.bind(divzero);
		}
		// div/0: quotient = -1, remainder = dividend.
		if (want_rem) { if (d.id() != a.id()) uc.mov(d, a); }
		else          uc.mov(d, sximm(-1));
		uc.bind(done);
	}
	/// @brief M-extension OP/OP-32 dispatch; operands are pre-narrowed for *W forms.
	void emit_muldiv(unsigned funct3, const Gp& d, const Gp& a, const Gp& b)
	{
		switch (funct3) {
		case 0x0: uc.mul(d, a, b); break;                      // MUL / MULW
		case 0x1: host_mul_hi(uc, d, a, b, true);  break;      // MULH
		case 0x2: {                                            // MULHSU
			// MULHSU: high(a*b) - (a<0 ? b : 0); adjust before the multiply clobbers operands.
			Gp adj = uc.new_similar_reg(d, "hsu");
			uc.sar(adj, a, Imm(d.size() * 8 - 1));   // 0 or ~0
			uc.and_(adj, adj, b);
			host_mul_hi(uc, d, a, b, false);
			uc.sub(d, d, adj);
			} break;
		case 0x3: host_mul_hi(uc, d, a, b, false); break;      // MULHU
		case 0x4: emit_div(d, a, b, false, true);  break;      // DIV  / DIVW
		case 0x5: emit_div(d, a, b, false, false); break;      // DIVU / DIVUW
		case 0x6: emit_div(d, a, b, true,  true);  break;      // REM  / REMW
		case 0x7: emit_div(d, a, b, true,  false); break;      // REMU / REMUW
		}
	}

	// --- RV64 word operations ---
	// *W forms compute in 32-bit scratch and sign-extend the result.
	Gp word_of(unsigned reg) { return get(reg).r32(); }
	void finish_word(unsigned rd, const Gp& tmp32) {
		host_sign_extend_word(cc, def(rd), tmp32);
	}

	// --- bit manipulation: Zba, Zbb, Zbs, Zbc, Zicond ----------------------
	// aj_zb_classify() is shared with region discovery to prevent emitting an unhandled encoding.

	static constexpr uint32_t XLEN = uint32_t(RVLEN) * 8;

	/// @brief zext32(rs1) for Zba *.UW forms; writing r32 zeroes the upper half on both hosts.
	Gp zext32_of(unsigned rs1) {
		Gp z = uc.new_gp64("uw");
		uc.mov(z.r32(), get(rs1).r32());
		return z;
	}

	/// @brief 1 << (amount & (XLEN-1)), the Zbs single-bit mask. Host shift masking matches RISC-V.
	Gp bit_mask(const Gp& amount) {
		Gp m = new_ireg("bitm");
		uc.mov(m, Imm(1));
		uc.shl(m, m, amount);
		return m;
	}
	/// @brief Immediate bit mask, materialized because x86 ALU immediates are 32-bit.
	Gp bit_mask(uint32_t shamt) {
		Gp m = new_ireg("bitm");
		const uint64_t v = uint64_t(1) << (shamt & (XLEN - 1));
		uc.mov(m, RV64 ? Imm(v) : Imm(uint32_t(v)));
		return m;
	}

	/// @brief Sign-extend the low `bits` of `src` via a shift pair.
	void emit_sign_extend(const Gp& d, const Gp& a, uint32_t bits) {
		const uint32_t up = uint32_t(a.size()) * 8 - bits;
		Gp t = uc.new_similar_reg(d, "sext");
		uc.shl(t, a, Imm(up));
		uc.sar(d, t, Imm(up));
	}

	/// @brief CLZ/CTZ, including RISC-V's XLEN answer for a zero input.
	void emit_count_zeros(const Gp& d, const Gp& a, bool leading)
	{
		if (host_count_zeros_is_exact(uc, leading)) {
			if (leading) uc.clz(d, a); else uc.ctz(d, a);
			return;
		}
		// Host leaves zero undefined; use scratch to avoid clobbering d==a.
		const uint32_t width = uint32_t(a.size()) * 8;
		Gp t = uc.new_similar_reg(d, "cnt");
		Gp r = uc.new_similar_reg(d, "cntz");
		if (leading) uc.clz(t, a); else uc.ctz(t, a);
		uc.select(r, Imm(width), t, cmp_eq(a, Imm(0)));
		uc.mov(d, r);
	}

	/// @brief ORC.B: every byte holding any set bit becomes 0xFF.
	void emit_orc_b(const Gp& d, const Gp& a)
	{
		const bool wide = (XLEN == 64);
		const uint64_t low7  = wide ? 0x7F7F7F7F7F7F7F7Full : 0x7F7F7F7Full;
		const uint64_t high1 = wide ? 0x8080808080808080ull : 0x80808080ull;
		Gp t = uc.new_similar_reg(d, "orcb");
		Gp u = uc.new_similar_reg(d, "orcb2");
		Gp k = uc.new_similar_reg(d, "orcbk");
		uc.mov (k, RV64 ? Imm(low7) : Imm(uint32_t(low7)));
		uc.and_(t, a, k);
		uc.add (t, t, k);      // carries into bit 7 iff the low seven bits were set
		uc.or_ (t, t, a);      // bit 7 of each byte is now set iff the byte was
		uc.mov (k, RV64 ? Imm(high1) : Imm(uint32_t(high1)));
		uc.and_(t, t, k);
		// Only bit 7 of each byte survives, so shifting right by seven lands on
		// bit 0 of the same byte and the subtraction cannot borrow across one:
		// 0x80 - 0x01 is 0x7F, and a zero byte stays zero.
		uc.shr (u, t, Imm(7));
		uc.sub (u, t, u);
		uc.or_ (d, t, u);
	}

	/// @brief Zbc CLMUL/CLMULH/CLMULR: windows of the 2*XLEN-bit carry-less product.
	void emit_clmul(AjZb zb, rv32i_instruction i)
	{
		const unsigned rd = i.Rtype.rd, rs1 = i.Rtype.rs1, rs2 = i.Rtype.rs2;
		if (!host_has_clmul(uc)) {
			const void* fn = (zb == AjZb::kClmul)  ? (const void*)info.cb->clmul
						   : (zb == AjZb::kClmulh) ? (const void*)info.cb->clmulh
												   : (const void*)info.cb->clmulr;
			InvokeNode* node;
			cc.invoke(Out(node), uint64_t(uintptr_t(fn)),
				FuncSignature::build<address_t, address_t, address_t>());
			node->set_arg(0, get(rs1));
			node->set_arg(1, get(rs2));
			node->set_ret(0, def(rd));
			return;
		}
		Gp lo = uc.new_gp64("clmlo"), hi = uc.new_gp64("clmhi");
		if constexpr (RV64) {
			host_clmul64(uc, lo, hi, get(rs1), get(rs2));
			switch (zb) {
			case AjZb::kClmul:  uc.mov(def(rd), lo); break;
			case AjZb::kClmulh: uc.mov(def(rd), hi); break;
			default: {          // CLMULR: the window one bit below the high half
				Gp t = uc.new_gp64("clmr"), u = uc.new_gp64("clmr2");
				uc.shl(t, hi, Imm(1));
				uc.shr(u, lo, Imm(63));
				uc.or_(def(rd), t, u);
				} break;
			}
		} else {
			// RV32: the 63-bit product fits in the low lane; windows are shifts.
			Gp x = uc.new_gp64("clmx"), y = uc.new_gp64("clmy");
			uc.mov(x.r32(), get(rs1));   // a 32-bit write zero-extends
			uc.mov(y.r32(), get(rs2));
			host_clmul64(uc, lo, hi, x, y);
			if (zb == AjZb::kClmul) {
				uc.mov(def(rd), lo.r32());
			} else {
				Gp t = uc.new_gp64("clmw");
				uc.shr(t, lo, Imm(zb == AjZb::kClmulh ? 32 : 31));
				uc.mov(def(rd), t.r32());
			}
		}
	}

	/// @brief The Zba, Zbb, Zbc and Zbs forms of OP, OP-IMM, OP-32 and
	/// OP-IMM-32, plus Zicond's two conditional zeroes.
	void emit_zb(AjZb zb, rv32i_instruction i)
	{
		// Every shamt-carrying encoding reads it from the same field, at the
		// width the guest allows.
		const uint32_t shamt = RV64 ? i.Itype.shift64_imm() : i.Itype.shift_imm();
		// SH1ADD/SH2ADD/SH3ADD put the shift in funct3 as 2, 4 or 6.
		const uint32_t sh_scale = 1u << (i.Rtype.funct3 >> 1);

		switch (zb)
		{
		// --- Zba: address generation ---
		case AjZb::kShAdd:      // rd = rs2 + (rs1 << n)
			uc.add_ext(def(i.Rtype.rd), get(i.Rtype.rs2), get(i.Rtype.rs1), sh_scale);
			break;
		case AjZb::kShAddUw:    // rd = rs2 + (zext32(rs1) << n)
			// .UW forms produce a full 64-bit result, not a sign-extended word.
			uc.add_ext(def(i.Rtype.rd), get(i.Rtype.rs2), zext32_of(i.Rtype.rs1), sh_scale);
			break;
		case AjZb::kAddUw:
			uc.add(def(i.Rtype.rd), get(i.Rtype.rs2), zext32_of(i.Rtype.rs1));
			break;
		case AjZb::kSlliUw:
			// A full six-bit shift amount, unlike every other OP-IMM-32 form.
			uc.shl(def(i.Itype.rd), zext32_of(i.Itype.rs1), Imm(i.Itype.shift64_imm()));
			break;

		// --- Zbb: logic with negate ---
		case AjZb::kAndn:
			uc.bic(def(i.Rtype.rd), get(i.Rtype.rs1), get(i.Rtype.rs2));
			break;
		case AjZb::kOrn: {
			Gp t = new_ireg("orn");
			uc.not_(t, get(i.Rtype.rs2));
			uc.or_(def(i.Rtype.rd), get(i.Rtype.rs1), t);
			} break;
		case AjZb::kXnor: {
			Gp t = new_ireg("xnor");
			uc.xor_(t, get(i.Rtype.rs1), get(i.Rtype.rs2));
			uc.not_(def(i.Rtype.rd), t);
			} break;

		// --- Zbb: min and max ---
		case AjZb::kMin:
			uc.smin(def(i.Rtype.rd), get(i.Rtype.rs1), get(i.Rtype.rs2));
			break;
		case AjZb::kMinu:
			uc.umin(def(i.Rtype.rd), get(i.Rtype.rs1), get(i.Rtype.rs2));
			break;
		case AjZb::kMax:
			uc.smax(def(i.Rtype.rd), get(i.Rtype.rs1), get(i.Rtype.rs2));
			break;
		case AjZb::kMaxu:
			uc.umax(def(i.Rtype.rd), get(i.Rtype.rs1), get(i.Rtype.rs2));
			break;

		// --- Zbb: bit counting ---
		case AjZb::kClz:
			emit_count_zeros(def(i.Itype.rd), get(i.Itype.rs1), true);
			break;
		case AjZb::kCtz:
			emit_count_zeros(def(i.Itype.rd), get(i.Itype.rs1), false);
			break;
		case AjZb::kCpop:
			host_popcount(uc, def(i.Itype.rd), get(i.Itype.rs1));
			break;
		case AjZb::kClzw:
		case AjZb::kCtzw:
		case AjZb::kCpopw: {
			// Count <= 32, so sign-extension is a no-op; keeps *W forms uniform.
			Gp t = uc.new_gp32("w");
			if (zb == AjZb::kCpopw)
				host_popcount(uc, t, word_of(i.Itype.rs1));
			else
				emit_count_zeros(t, word_of(i.Itype.rs1), zb == AjZb::kClzw);
			finish_word(i.Itype.rd, t);
			} break;

		// --- Zbb: extend, reverse, combine ---
		case AjZb::kSextB:
			emit_sign_extend(def(i.Itype.rd), get(i.Itype.rs1), 8);
			break;
		case AjZb::kSextH:
			emit_sign_extend(def(i.Itype.rd), get(i.Itype.rs1), 16);
			break;
		case AjZb::kRev8:
			uc.bswap(def(i.Itype.rd), get(i.Itype.rs1));
			break;
		case AjZb::kOrcB:
			emit_orc_b(def(i.Itype.rd), get(i.Itype.rs1));
			break;
		case AjZb::kPack: {
			// Shift pair clears each upper half without a wide mask. rs2==x0 on RV32 is ZEXT.H.
			constexpr uint32_t half = XLEN / 2;
			Gp lo = new_ireg("packlo"), hi = new_ireg("packhi");
			uc.shl(lo, get(i.Rtype.rs1), Imm(half));
			uc.shr(lo, lo, Imm(half));
			uc.shl(hi, get(i.Rtype.rs2), Imm(half));
			uc.or_(def(i.Rtype.rd), lo, hi);
			} break;
		case AjZb::kPackH: {
			// Low bytes of rs1 and rs2, packed into rd[15:0].
			Gp lo = new_ireg("phlo"), hi = new_ireg("phhi");
			uc.and_(lo, get(i.Rtype.rs1), Imm(0xFF));
			uc.and_(hi, get(i.Rtype.rs2), Imm(0xFF));
			uc.shl(hi, hi, Imm(8));
			uc.or_(def(i.Rtype.rd), lo, hi);
			} break;
		case AjZb::kPackW: if constexpr (RV64) {
			// RV64: pairs 16-bit halves into a sign-extended word. rs2==x0 is ZEXT.H.
			Gp t = uc.new_gp32("w"), h = uc.new_gp32("wh");
			uc.and_(t, word_of(i.Rtype.rs1), Imm(0xFFFF));
			uc.shl(h, word_of(i.Rtype.rs2), Imm(16));
			uc.or_(t, t, h);
			finish_word(i.Rtype.rd, t);
			} break;

		// --- Zbb: rotate ---
		case AjZb::kRol:
			uc.rol(def(i.Rtype.rd), get(i.Rtype.rs1), get(i.Rtype.rs2));
			break;
		case AjZb::kRor:
			uc.ror(def(i.Rtype.rd), get(i.Rtype.rs1), get(i.Rtype.rs2));
			break;
		case AjZb::kRori:
			// Rotate amount is masked to XLEN; zero rotates by zero.
			uc.ror(def(i.Itype.rd), get(i.Itype.rs1), Imm(shamt));
			break;
		case AjZb::kRolw:
		case AjZb::kRorw: if constexpr (RV64) {
			Gp t = uc.new_gp32("w");
			if (zb == AjZb::kRolw)
				uc.rol(t, word_of(i.Rtype.rs1), word_of(i.Rtype.rs2));
			else
				uc.ror(t, word_of(i.Rtype.rs1), word_of(i.Rtype.rs2));
			finish_word(i.Rtype.rd, t);
			} break;
		case AjZb::kRoriw: if constexpr (RV64) {
			Gp t = uc.new_gp32("w");
			uc.ror(t, word_of(i.Itype.rs1), Imm(i.Itype.shift_imm()));
			finish_word(i.Itype.rd, t);
			} break;

		// --- Zbs: single-bit ---
		case AjZb::kBset:
			uc.or_(def(i.Rtype.rd), get(i.Rtype.rs1), bit_mask(get(i.Rtype.rs2)));
			break;
		case AjZb::kBclr:
			uc.bic(def(i.Rtype.rd), get(i.Rtype.rs1), bit_mask(get(i.Rtype.rs2)));
			break;
		case AjZb::kBinv:
			uc.xor_(def(i.Rtype.rd), get(i.Rtype.rs1), bit_mask(get(i.Rtype.rs2)));
			break;
		case AjZb::kBext: {
			Gp t = new_ireg("bext");
			uc.shr(t, get(i.Rtype.rs1), get(i.Rtype.rs2));
			uc.and_(def(i.Rtype.rd), t, Imm(1));
			} break;
		case AjZb::kBseti:
			uc.or_(def(i.Itype.rd), get(i.Itype.rs1), bit_mask(shamt));
			break;
		case AjZb::kBclri:
			uc.bic(def(i.Itype.rd), get(i.Itype.rs1), bit_mask(shamt));
			break;
		case AjZb::kBinvi:
			uc.xor_(def(i.Itype.rd), get(i.Itype.rs1), bit_mask(shamt));
			break;
		case AjZb::kBexti: {
			Gp t = new_ireg("bexti");
			uc.shr(t, get(i.Itype.rs1), Imm(shamt));
			uc.and_(def(i.Itype.rd), t, Imm(1));
			} break;

		// --- Zbc: carry-less multiply ---
		case AjZb::kClmul:
		case AjZb::kClmulh:
		case AjZb::kClmulr:
			emit_clmul(zb, i);
			break;

		// --- Zicond: conditional zero ---
		// rs2 is the condition, rs1 the value.
		case AjZb::kCzeroEqz:   // rd = (rs2 == 0) ? 0 : rs1
			uc.select(def(i.Rtype.rd), Imm(0), get(i.Rtype.rs1),
				cmp_eq(get(i.Rtype.rs2), Imm(0)));
			break;
		case AjZb::kCzeroNez:   // rd = (rs2 != 0) ? 0 : rs1
			uc.select(def(i.Rtype.rd), Imm(0), get(i.Rtype.rs1),
				cmp_ne(get(i.Rtype.rs2), Imm(0)));
			break;

		case AjZb::kNone:
			// Unreachable: aj_is_emittable() rejects unclassified encodings.
			failed = true;
			break;
		}
	}

	// --- memory ---
	// EA in a pointer-sized register; RV32 writes via r32 so the upper half is zero.
	Gp address_of(unsigned rs1, int32_t simm) {
		Gp a = uc.new_gp_ptr("addr");
		if constexpr (RV64) {
			if (rs1 == 0)        uc.mov(a, rvimm(address_t(int64_t(simm))));
			else if (simm == 0)  uc.mov(a, get(rs1));
			else                 uc.add(a, get(rs1), sximm(simm));
		} else {
			if (rs1 == 0) {
				uc.mov(a.r32(), Imm(uint32_t(simm)));
			} else if (simm == 0) {
				uc.mov(a.r32(), get(rs1));
			} else {
#if defined(ASMJIT_UJIT_X86)
				// A 32-bit LEA gives RV32 wrapping and zero extension in one instruction.
				cc.lea(a.r32(), mem_ptr(get(rs1).r64(), simm));
#else
				uc.add(a.r32(), get(rs1), sximm(simm));
#endif
			}
		}
		return a;
	}
	/// @brief Guest-width view of an address register.
	static Gp addr_value(const Gp& a) {
		if constexpr (RV64) return a; else return a.r32();
	}

	static FuncSignature load_signature() {
		return FuncSignature::build<address_t, void*, void*, address_t, address_t>();
	}
	static FuncSignature store_signature() {
		return FuncSignature::build<void, void*, void*, address_t, address_t, address_t>();
	}
	// A faulting helper zeroes max_counter; exit rather than running a half-executed instruction.
	void emit_fault_check(address_t pc, uint32_t pend) {
		Label ok = uc.new_label();
		Gp maxc = uc.new_gp64("maxc");
		uc.load(maxc, mem_ptr(st, off_max()));
		uc.j(ok, test_nz(maxc));
		emit_exit(pc, pend);
		uc.bind(ok);
	}
	void call_load_helper(address_t pc, unsigned funct3, const Gp& dst,
		const Gp& addr, uint32_t pend)
	{
		const auto* cb = info.cb;
		uint64_t fn = 0;
		switch (funct3) {
		case 0x0: fn = uint64_t(uintptr_t(cb->load_i8));  break;
		case 0x1: fn = uint64_t(uintptr_t(cb->load_i16)); break;
		case 0x2: fn = uint64_t(uintptr_t(cb->load_i32)); break;
		case 0x3: fn = uint64_t(uintptr_t(cb->load_i64)); break;  // RV64: LD
		case 0x4: fn = uint64_t(uintptr_t(cb->load_u8));  break;
		case 0x5: fn = uint64_t(uintptr_t(cb->load_u16)); break;
		case 0x6: fn = uint64_t(uintptr_t(cb->load_u32)); break;  // RV64: LWU
		}
		InvokeNode* node;
		cc.invoke(Out(node), fn, load_signature());
		node->set_arg(0, cpu);
		node->set_arg(1, st);
		node->set_arg(2, addr_value(addr));
		node->set_arg(3, rvimm(pc));
		node->set_ret(0, dst);
		emit_fault_check(pc, pend);
	}
	void call_store_helper(address_t pc, unsigned funct3, const Gp& value,
		const Gp& addr, uint32_t pend)
	{
		const auto* cb = info.cb;
		uint64_t fn = 0;
		switch (funct3) {
		case 0x0: fn = uint64_t(uintptr_t(cb->store_8));  break;
		case 0x1: fn = uint64_t(uintptr_t(cb->store_16)); break;
		case 0x2: fn = uint64_t(uintptr_t(cb->store_32)); break;
		case 0x3: fn = uint64_t(uintptr_t(cb->store_64)); break;  // RV64: SD
		}
		InvokeNode* node;
		cc.invoke(Out(node), fn, store_signature());
		node->set_arg(0, cpu);
		node->set_arg(1, st);
		node->set_arg(2, addr_value(addr));
		node->set_arg(3, value);
		node->set_arg(4, rvimm(pc));
		emit_fault_check(pc, pend);
	}
	/// @brief True when the arena requires a bounds check.
	bool arena_is_checked() const noexcept {
		return info.arena_mask == 0 && !info.unsafe_remove_checks;
	}
	/// @brief Arena index: direct for flat, masked for N-bit encompassing (over-allocation covers straddles).
	Gp arena_index(const Gp& addr) {
		// Full-width mask is a no-op (common for RV32 under a 32-bit arena).
		if (info.arena_mask == 0 || info.arena_mask >= uint64_t(address_t(~address_t(0))))
			return addr;
		Gp t = uc.new_gp_ptr("aidx");
		uc.and_(t, addr, Imm(info.arena_mask));
		return t;
	}
	// Single-sided bounds check matching the interpreter, with immutable arena
	// bounds baked into the generated function.
	void emit_arena_check(const Gp& addr, bool is_write, const Label& slow) {
		if (!arena_is_checked())
			return;
		Gp t = new_ireg("bchk");
		const uint64_t lo = is_write ? info.arena_roend_value : uint64_t(Memory<W>::RWREAD_BEGIN);
		const uint64_t hi = is_write ? info.arena_wrbound_value : info.arena_rdbound_value;
#if defined(ASMJIT_UJIT_X86)
		cc.lea(t, mem_ptr(addr, -int32_t(lo)));
#else
		const Gp a = addr_value(addr);
		uc.sub(t, a, rvimm(address_t(lo)));
#endif
		if constexpr (RV64) {
			if (hi > uint64_t(INT32_MAX))
				uc.j(slow, ucmp_ge(t, is_write ? write_bound : read_bound));
			else
				uc.j(slow, ucmp_ge(t, Imm(hi)));
		} else {
			// x86 compares accept every 32-bit bit pattern; do not reject the
			// near-4GB rvlinux arena merely because it exceeds INT32_MAX.
			uc.j(slow, ucmp_ge(t, Imm(uint32_t(hi))));
		}
	}
	void emit_load(address_t pc, unsigned funct3, unsigned rd, unsigned rs1, int32_t simm)
	{
		auto addr = address_of(rs1, simm);
		const Gp dst = def(rd);
		if (!info.inline_memory) {
			call_load_helper(pc, funct3, dst, addr, pending - 1);
			return;
		}
		Label slow = uc.new_label(), done = uc.new_label();
		emit_arena_check(addr, false, slow);
		const Mem m = mem_ptr(arena, arena_index(addr));
		switch (funct3) {
		case 0x0: uc.load_i8 (dst, m); break;                        // LB
		case 0x1: uc.load_i16(dst, m); break;                        // LH
		case 0x2: uc.load_i32(dst, m); break;                        // LW (widens on RV64)
		case 0x3: if constexpr (RV64) uc.load_i64(dst, m); break;    // LD
		case 0x4: uc.load_u8 (dst, m); break;                        // LBU
		case 0x5: uc.load_u16(dst, m); break;                        // LHU
		case 0x6: if constexpr (RV64) uc.load_u32(dst, m); break;    // LWU
		}
		uc.bind(done);
		if (arena_is_checked()) {
			deferred.push_back([=, this, pend = pending - 1] {
				uc.bind(slow);
				// A claimed entry would immediately re-enter this region at the
				// same faulting instruction. Use the precise helper only for that
				// rare case; all ordinary sites exit for interpreter re-execution.
				if (std::binary_search(entries.begin(), entries.end(), pc)) {
					call_load_helper(pc, funct3, dst, addr, pend);
					uc.j(done);
				} else emit_exit(pc, pend);
			});
		}
	}
	void emit_store(address_t pc, unsigned funct3, unsigned rs1, unsigned rs2, int32_t simm)
	{
		auto addr = address_of(rs1, simm);
		const Gp src = get(rs2);
		if (!info.inline_memory) {
			call_store_helper(pc, funct3, src, addr, pending - 1);
			return;
		}
		Label slow = uc.new_label(), done = uc.new_label();
		emit_arena_check(addr, true, slow);
		const Mem m = mem_ptr(arena, arena_index(addr));
		switch (funct3) {
		case 0x0: uc.store_u8 (m, src); break;                        // SB
		case 0x1: uc.store_u16(m, src); break;                        // SH
		case 0x2: uc.store_u32(m, src); break;                        // SW
		case 0x3: if constexpr (RV64) uc.store_u64(m, src); break;    // SD
		}
		uc.bind(done);
		if (arena_is_checked()) {
			deferred.push_back([=, this, pend = pending - 1] {
				uc.bind(slow);
				if (std::binary_search(entries.begin(), entries.end(), pc)) {
					call_store_helper(pc, funct3, src, addr, pend);
					uc.j(done);
				} else emit_exit(pc, pend);
			});
		}
	}

	// --- interpreter handler calls -----------------------------------------

	// A dyncall stub is an ordinary C function, so fa0-fa7 carry the
	// floating-point arguments and fa0-fa1 the floating-point results even
	// unless a counted call explicitly promises no FP access. Spilling is gated on
	// the region holding a newer value than memory, so a dyncall in a region
	// that caches no f-registers still costs nothing.
	static constexpr unsigned FP_INPUTS = 8, FP_OUTPUTS = 2;

	// The ordinary execute helper does not touch counters. Only budgeted
	// dyncalls need to publish the live count and collect handler penalties.
	static uint32_t execute_counted_dyncall(CPU<W>& cpu, AjState<W>* st,
		uint32_t instr, address_t pc, const void* handler) noexcept
	{
		auto& machine = cpu.machine();
		machine.set_instruction_counter(st->counter);
		machine.set_max_instructions(st->max_counter);
		cpu.registers().pc = pc;
		try {
			auto fn = reinterpret_cast<instruction_handler<W>>(
				reinterpret_cast<uintptr_t>(handler));
			fn(cpu, rv32i_instruction{instr});
		} catch (...) {
			cpu.set_current_exception(std::current_exception());
			st->max_counter = 0;
			machine.stop();
			return 1;
		}
		st->counter = machine.instruction_counter();
		st->max_counter = machine.max_instructions();
		return 0;
	}

	void emit_dyncall(address_t pc, rv32i_instruction i)
	{
		for (unsigned r = 10; r < 10 + Dyncall::inputs(i.whole); r++)
			if (writeset[r]) uc.store(reg_mem(r), vreg[r]);
		if (Dyncall::floats(i.whole))
			for (unsigned r = REG_FA0; r < REG_FA0 + FP_INPUTS; r++)
				if (fp_writeset[r]) uc.v_storeu64_u64(freg_mem(r), fvreg[r]);
		if (!info.ignore_instruction_limit) {
			Gp retired = uc.new_gp64("dyncnt");
			uc.add(retired, counter, Imm(pending));
			uc.store(mem_ptr(st, off_counter()), retired);
		}
		const auto handler = uint64_t(uintptr_t(CPU<W>::decode(i).handler));
		InvokeNode* node;
		Gp fault;
		if (info.ignore_instruction_limit) {
			cc.invoke(Out(node), uint64_t(uintptr_t(info.cb->execute)),
				FuncSignature::build<void, void*, void*, uint32_t, address_t, const void*>());
		} else {
			cc.invoke(Out(node), uint64_t(uintptr_t(execute_counted_dyncall)),
				FuncSignature::build<uint32_t, void*, void*, uint32_t, address_t, const void*>());
			fault = uc.new_gp32("dyncall_fault");
			node->set_ret(0, fault);
		}
		node->set_arg(0, cpu);
		node->set_arg(1, st);
		node->set_arg(2, Imm(i.whole));
		node->set_arg(3, rvimm(pc));
		node->set_arg(4, Imm(handler));
		for (unsigned r = 10; r < 10 + Dyncall::outputs(i.whole); r++)
			if (readset[r]) uc.load(vreg[r], reg_mem(r));
		if (Dyncall::floats(i.whole))
			for (unsigned r = REG_FA0; r < REG_FA0 + FP_OUTPUTS; r++)
				if (fp_readset[r]) uc.v_loadu64_u64(fvreg[r], freg_mem(r));
		if (info.ignore_instruction_limit) {
			emit_fault_check(pc, pending - 1);
		} else {
			Label ok = uc.new_label();
			uc.j(ok, test_z(fault));
			emit_exit(pc, pending - 1);
			uc.bind(ok);
			uc.load(counter, mem_ptr(st, off_counter()));
			pending = 0;
			emit_backedge_check(pc + 4);
		}
	}

	/// @brief Run one instruction via the interpreter handler; write back cached operands, reload rd.
	void emit_handler_call(address_t pc, rv32i_instruction i)
	{
		emit_handler_call(pc, i, pending - 1);
	}
	void emit_handler_call(address_t pc, rv32i_instruction i, uint32_t pend)
	{
		// All encodings reaching here use standard register fields (no R4-type rs3).
		const unsigned rd  = i.Itype.rd;
		const unsigned rs1 = i.Itype.rs1;
		const unsigned rs2 = (i.whole >> 20) & 0x1F;

		const auto store_int = [&] (unsigned r) {
			if (r != 0 && readset[r]) uc.store(reg_mem(r), vreg[r]);
		};
		const auto store_fp = [&] (unsigned r) {
			if (fp_readset[r]) uc.v_storeu64_u64(freg_mem(r), fvreg[r]);
		};
		store_int(rs1); store_int(rs2); store_int(rd);
		store_fp(rs1);  store_fp(rs2);  store_fp(rd);

		// Handler resolved at JIT time; decoding is pure on the constant encoding.
		const auto handler = uint64_t(uintptr_t(CPU<W>::decode(i).handler));
		InvokeNode* node;
		cc.invoke(Out(node), uint64_t(uintptr_t(info.cb->execute)),
			FuncSignature::build<void, void*, void*, uint32_t, address_t, const void*>());
		node->set_arg(0, cpu);
		node->set_arg(1, st);
		node->set_arg(2, Imm(i.whole));
		node->set_arg(3, rvimm(pc));
		node->set_arg(4, Imm(handler));

		if (rd != 0 && readset[rd]) uc.load(vreg[rd], reg_mem(rd));
		if (fp_readset[rd]) uc.v_loadu64_u64(fvreg[rd], freg_mem(rd));
		emit_fault_check(pc, pend);
	}

	// --- vector memory ------------------------------------------------------
	// Unit-stride and whole-register transfers are block copies between arena and register file.
	// All other vector memory ops fall back to the interpreter handler.
#ifdef RISCV_EXT_VECTOR
	static constexpr uint32_t VLEN = VectorLane::size();
	static_assert(VLEN % 16 == 0, "A vector register is copied 128 bits at a time");
	// Max inlined transfer is 8 regs (LMUL=8); must fit within the arena over-allocation.
	static_assert(8 * uint64_t(VLEN) <= Memory<W>::OVERALLOCATE,
		"A vector register group must fit within the arena over-allocation");

	static constexpr uint32_t vlen_log2() {
		uint32_t n = 0, v = VLEN;
		while (v > 1) { v >>= 1; n++; }
		return n;
	}
	/// @brief Max legal EMUL for a given vreg (alignment + fit within v0-v31).
	static int max_legal_emul(unsigned vreg) {
		int best = 0;
		for (int e = 1; e <= 3; e++) {
			const unsigned regs = 1u << e;
			if ((vreg % regs) == 0 && vreg + regs <= 32) best = e; else break;
		}
		return best;
	}
	/// @brief Copies one vector register between two host pointers.
	void emit_vector_lane(const Gp& dst, const Gp& src, int32_t offset) {
		for (uint32_t o = 0; o < VLEN; o += 16) {
			Vec t = uc.new_vec128("vlane");
			uc.v_loadu128(t, mem_ptr(src, offset + int32_t(o)));
			uc.v_storeu128(mem_ptr(dst, offset + int32_t(o)), t);
		}
	}
	/// @brief Whole-register load/store: fixed length from encoding, checks only address.
	void emit_vector_whole_register(address_t pc, const AjVectorMem& vmi, rv32i_instruction i)
	{
		auto addr = address_of(vmi.rs1, 0);
		Label slow = uc.new_label(), done = uc.new_label();
		emit_arena_check(addr, vmi.is_store, slow);
		Gp mem = uc.new_gp_ptr("vmem");
		uc.add(mem, arena, arena_index(addr));
		Gp file = uc.new_gp_ptr("vfile");
		uc.lea(file, mem_ptr(cpu, info.rvv_regs + int32_t(vmi.vreg * VLEN)));
		for (unsigned r = 0; r < vmi.nregs; r++) {
			const int32_t off = int32_t(r * VLEN);
			if (vmi.is_store) emit_vector_lane(mem, file, off);
			else              emit_vector_lane(file, mem, off);
		}
		uc.bind(done);
		deferred.push_back([=, this, pend = pending - 1] {
			uc.bind(slow);
			emit_handler_call(pc, i, pend);
			uc.j(done);
		});
	}
	/// @brief Unit-stride vle/vse: run-time guard checks vtype, EMUL legality, and whole-register length.
	void emit_vector_unit_stride(address_t pc, const AjVectorMem& vmi, rv32i_instruction i)
	{
		const uint32_t eew = vmi.eew_log2;
		Label slow = uc.new_label(), done = uc.new_label();

		Gp vill = uc.new_gp32("vill");
		uc.load_u8(vill, mem_ptr(cpu, info.rvv_vill));
		uc.j(slow, test_nz(vill));

		// EMUL = LMUL * EEW / SEW (log2).
		Gp emul = uc.new_gp32("vemul");
		uc.load(emul, mem_ptr(cpu, info.rvv_lmul));
		Gp sew = uc.new_gp32("vsew");
		uc.load(sew, mem_ptr(cpu, info.rvv_vsew));
		uc.sub(emul, emul, sew);
		if (eew != 0) uc.add(emul, emul, Imm(int32_t(eew)));
		uc.j(slow, scmp_lt(emul, Imm(-3)));
		uc.j(slow, scmp_gt(emul, Imm(max_legal_emul(vmi.vreg))));

		// Length in bytes → whole registers. Partial-register transfers go to handler.
		Gp bytes = new_ireg("vbytes");
		uc.load(bytes, mem_ptr(cpu, info.rvv_vl));
		if (eew != 0) uc.shl(bytes, bytes, Imm(eew));
		Gp rem = new_ireg("vrem");
		uc.and_(rem, bytes, Imm(VLEN - 1));
		uc.j(slow, test_nz(rem));
		Gp regs = new_ireg("vregs");
		uc.shr(regs, bytes, Imm(vlen_log2()));
		// dregs = 1 << max(EMUL, 0), the number of registers the group holds.
		Gp shift = uc.new_gp32("vshift");
		uc.smax(shift, emul, Imm(0));
		Gp shiftw = new_ireg("vshiftw");
		if constexpr (RV64) uc.mov(shiftw.r32(), shift);
		else                uc.mov(shiftw, shift);
		Gp dregs = new_ireg("vdregs");
		uc.mov(dregs, Imm(1));
		uc.shl(dregs, dregs, shiftw);
		uc.j(slow, ucmp_gt(regs, dregs));

		auto addr = address_of(vmi.rs1, 0);
		emit_arena_check(addr, vmi.is_store, slow);
		// An empty transfer touches no memory, and so cannot fault.
		uc.j(done, cmp_eq(regs, Imm(0)));

		Gp mem = uc.new_gp_ptr("vmem");
		uc.add(mem, arena, arena_index(addr));
		Gp file = uc.new_gp_ptr("vfile");
		uc.lea(file, mem_ptr(cpu, info.rvv_regs + int32_t(vmi.vreg * VLEN)));
		Label loop = uc.new_label();
		uc.bind(loop);
		if (vmi.is_store) emit_vector_lane(mem, file, 0);
		else              emit_vector_lane(file, mem, 0);
		uc.add(mem, mem, Imm(VLEN));
		uc.add(file, file, Imm(VLEN));
		uc.sub(regs, regs, Imm(1));
		uc.j(loop, test_nz(regs));
		uc.bind(done);

		deferred.push_back([=, this, pend = pending - 1] {
			uc.bind(slow);
			emit_handler_call(pc, i, pend);
			uc.j(done);
		});
	}
	/// @brief True when this vector transfer is inlined rather than called.
	bool vector_memory_inlinable(const AjVectorMem& vmi) const {
		return vmi.form != AjVecForm::kNone && info.inline_memory;
	}
	void emit_vector_memory(address_t pc, const AjVectorMem& vmi, rv32i_instruction i)
	{
		switch (vmi.form) {
		case AjVecForm::kUnitStride:    emit_vector_unit_stride(pc, vmi, i); break;
		case AjVecForm::kWholeRegister: emit_vector_whole_register(pc, vmi, i); break;
		default:                        emit_handler_call(pc, i); break;
		}
	}
#else
	bool vector_memory_inlinable(const AjVectorMem&) const { return false; }
	void emit_vector_memory(address_t pc, const AjVectorMem&, rv32i_instruction i) {
		emit_handler_call(pc, i);
	}
#endif

	// --- F/D extension -----------------------------------------------------
	// f-registers live in host XMM/NEON lanes. Single-precision results are NaN-boxed
	// only when the `nanboxing` policy is active.

	/// @brief NaN-box: fill upper 32 bits with ones so a double read yields NaN.
	void nanbox(const Vec& d) {
		if constexpr (riscv::nanboxing)
			uc.v_or_f64(d, d, uc.simd_const(&uc.ct().p_FFFFFFFF00000000, Bcst::k64, d));
	}
	/// @brief NaN-box single-precision results; doubles need no fixup.
	void fp_result(const Vec& d, bool is_double) {
		if (!is_double) nanbox(d);
	}

	// Bitwise ops stay in the caller's precision domain to avoid x86 bypass penalties.
	void v_and_fp(bool dbl, const Vec& d, const Vec& a, const Operand& b) {
		if (dbl) uc.v_and_f64(d, a, b); else uc.v_and_f32(d, a, b);
	}
	void v_andn_fp(bool dbl, const Vec& d, const Vec& a, const Operand& b) {
		if (dbl) uc.v_andn_f64(d, a, b); else uc.v_andn_f32(d, a, b);
	}
	void v_or_fp(bool dbl, const Vec& d, const Vec& a, const Operand& b) {
		if (dbl) uc.v_or_f64(d, a, b); else uc.v_or_f32(d, a, b);
	}
	void v_xor_fp(bool dbl, const Vec& d, const Vec& a, const Operand& b) {
		if (dbl) uc.v_xor_f64(d, a, b); else uc.v_xor_f32(d, a, b);
	}
	/// @brief dst = mask ? if_set : if_clear (mask is all-ones or all-zeros from a scalar compare).
	void select_masked(bool dbl, const Vec& dst, const Vec& mask,
		const Vec& if_set, const Vec& if_clear)
	{
		Vec a = uc.new_vec128("selA");
		Vec b = uc.new_vec128("selB");
		v_andn_fp(dbl, a, mask, if_clear);   // a = ~mask & if_clear
		v_and_fp(dbl, b, mask, if_set);
		v_or_fp(dbl, dst, a, b);
	}

	// FP helpers carry uint64_t: FLD/FSD are 64-bit on both RV32 and RV64.
	static FuncSignature fp_load_signature() {
		return FuncSignature::build<uint64_t, void*, void*, address_t, address_t>();
	}
	static FuncSignature fp_store_signature() {
		return FuncSignature::build<void, void*, void*, address_t, uint64_t, address_t>();
	}
	void call_fp_load_helper(address_t pc, bool is_double, const Vec& dst,
		const Gp& addr, uint32_t pend)
	{
		const uint64_t fn = uint64_t(uintptr_t(
			is_double ? info.cb->load_dbl : info.cb->load_fl));
		Gp bits = uc.new_gp64("fbits");
		InvokeNode* node;
		cc.invoke(Out(node), fn, fp_load_signature());
		node->set_arg(0, cpu);
		node->set_arg(1, st);
		node->set_arg(2, addr_value(addr));
		node->set_arg(3, rvimm(pc));
		node->set_ret(0, bits);
		uc.s_mov_u64(dst, bits);
		emit_fault_check(pc, pend);
	}
	void call_fp_store_helper(address_t pc, bool is_double, const Vec& value,
		const Gp& addr, uint32_t pend)
	{
		const uint64_t fn = uint64_t(uintptr_t(
			is_double ? info.cb->store_dbl : info.cb->store_fl));
		Gp bits = uc.new_gp64("fbits");
		uc.s_mov_u64(bits, value);   // the helper narrows a single-precision store
		InvokeNode* node;
		cc.invoke(Out(node), fn, fp_store_signature());
		node->set_arg(0, cpu);
		node->set_arg(1, st);
		node->set_arg(2, addr_value(addr));
		node->set_arg(3, bits);
		node->set_arg(4, rvimm(pc));
		emit_fault_check(pc, pend);
	}
	void emit_fp_load(address_t pc, unsigned funct3, unsigned rd, unsigned rs1, int32_t simm)
	{
		const bool is_double = (funct3 == 0x3);
		auto addr = address_of(rs1, simm);
		const Vec dst = fdef(rd);
		if (!info.inline_memory) {
			call_fp_load_helper(pc, is_double, dst, addr, pending - 1);
			fp_result(dst, is_double);
			return;
		}
		Label slow = uc.new_label(), done = uc.new_label();
		emit_arena_check(addr, false, slow);
		const Mem m = mem_ptr(arena, arena_index(addr));
		if (is_double) uc.v_loadu64_u64(dst, m);   // FLD
		else           uc.v_loadu32_u32(dst, m);   // FLW
		uc.bind(done);
		if (arena_is_checked()) {
			deferred.push_back([=, this, pend = pending - 1] {
				uc.bind(slow);
				if (std::binary_search(entries.begin(), entries.end(), pc)) {
					call_fp_load_helper(pc, is_double, dst, addr, pend);
					uc.j(done);
				} else emit_exit(pc, pend);
			});
		}
		// Emitted after the merge point, so it covers both paths.
		fp_result(dst, is_double);
	}
	void emit_fp_store(address_t pc, unsigned funct3, unsigned rs1, unsigned rs2, int32_t simm)
	{
		const bool is_double = (funct3 == 0x3);
		auto addr = address_of(rs1, simm);
		const Vec src = fget(rs2);
		if (!info.inline_memory) {
			call_fp_store_helper(pc, is_double, src, addr, pending - 1);
			return;
		}
		Label slow = uc.new_label(), done = uc.new_label();
		emit_arena_check(addr, true, slow);
		const Mem m = mem_ptr(arena, arena_index(addr));
		if (is_double) uc.v_storeu64_u64(m, src);  // FSD
		else           uc.v_storeu32_u32(m, src);  // FSW
		uc.bind(done);
		if (arena_is_checked()) {
			deferred.push_back([=, this, pend = pending - 1] {
				uc.bind(slow);
				if (std::binary_search(entries.begin(), entries.end(), pc)) {
					call_fp_store_helper(pc, is_double, src, addr, pend);
					uc.j(done);
				} else emit_exit(pc, pend);
			});
		}
	}

	/// @brief Move a 32-bit FP result into a guest int register, sign-extending on RV64.
	void set_word_result(unsigned rd, const Gp& t32) {
		if constexpr (RV64) host_sign_extend_word(cc, def(rd), t32);
		else                uc.mov(def(rd), t32);
	}

	/// @brief FMADD/FMSUB/FNMSUB/FNMADD: explicit sign flip + FMADD to keep NaN
	/// propagation bit-identical with the interpreter's std::fma path.
	void emit_fmadd(rv32i_instruction i)
	{
		const rv32f_instruction fi { i };
		const bool dbl = (fi.R4type.funct2 == 0x1);
		const Vec a = fget(fi.R4type.rs1), b = fget(fi.R4type.rs2);
		const Vec c = fget(fi.R4type.rs3), d = fdef(fi.R4type.rd);

		const auto negated = [&] (const Vec& src) {
			Vec t = uc.new_vec128("fneg");
			if (dbl) uc.s_neg_f64(t, src); else uc.s_neg_f32(t, src);
			return t;
		};
		const auto madd = [&] (const Vec& dst, const Vec& x, const Vec& y, const Vec& z) {
			if (dbl) uc.s_madd_f64(dst, x, y, z); else uc.s_madd_f32(dst, x, y, z);
		};
		switch (i.opcode()) {
		case RV32F_FMADD:                        // fma(rs1, rs2, rs3)
			madd(d, a, b, c);
			break;
		case RV32F_FMSUB:                        // fma(rs1, rs2, -rs3)
			madd(d, a, b, negated(c));
			break;
		case RV32F_FNMSUB:                       // fma(-rs1, rs2, rs3)
			madd(d, negated(a), b, c);
			break;
		case RV32F_FNMADD: {                     // -fma(rs1, rs2, rs3)
			Vec t = uc.new_vec128("fnmadd");
			madd(t, a, b, c);
			if (dbl) uc.s_neg_f64(d, t); else uc.s_neg_f32(d, t);
			} break;
		}
		fp_result(d, dbl);
	}

	/// @brief FMIN/FMAX with RISC-V NaN and signed-zero semantics.
	void emit_fminmax(const rv32f_instruction& fi, bool dbl)
	{
		const bool is_max = (fi.R4type.funct3 == 0x1);
		const Vec a = fget(fi.R4type.rs1), b = fget(fi.R4type.rs2);
		const Vec d = fdef(fi.R4type.rd);
		Vec ma = uc.new_vec128("fmina");
		Vec mb = uc.new_vec128("fminb");
		if (dbl) {
			uc.s_cmp_unord_f64(ma, a, a);
			uc.s_cmp_unord_f64(mb, b, b);
		} else {
			uc.s_cmp_unord_f32(ma, a, a);
			uc.s_cmp_unord_f32(mb, b, b);
		}

		Vec r = uc.new_vec128("fminr");
		if (dbl) {
			if (is_max) uc.s_max_f64(r, a, b); else uc.s_min_f64(r, a, b);
		} else {
			if (is_max) uc.s_max_f32(r, a, b); else uc.s_min_f32(r, a, b);
		}
		// One NaN returns the numeric operand; two return the canonical NaN.
		Vec t = uc.new_vec128("fmint");
		select_masked(dbl, t, ma, b, r);
		select_masked(dbl, r, mb, a, t);
		Vec both_nan = uc.new_vec128("fminnn");
		v_and_fp(dbl, both_nan, ma, mb);
		select_masked(dbl, t, both_nan, dbl ? canon64 : canon32, r);

		// For a ±0 tie, min is bitwise OR (-0 wins) and max is bitwise
		// AND (+0 wins). Hardware min/max operand-order rules differ here.
		const Operand z = uc.simd_const(&uc.ct().p_0000000000000000, Bcst::k32, a);
		Vec za = uc.new_vec128("fminza");
		Vec zb = uc.new_vec128("fminzb");
		if (dbl) {
			uc.s_cmp_eq_f64(za, a, z);
			uc.s_cmp_eq_f64(zb, b, z);
		} else {
			uc.s_cmp_eq_f32(za, a, z);
			uc.s_cmp_eq_f32(zb, b, z);
		}
		Vec both_zero = uc.new_vec128("fminzz");
		v_and_fp(dbl, both_zero, za, zb);
		Vec zero_result = uc.new_vec128("fminzr");
		if (is_max) v_and_fp(dbl, zero_result, a, b);
		else        v_or_fp (dbl, zero_result, a, b);
		select_masked(dbl, d, both_zero, zero_result, t);
		fp_result(d, dbl);
	}

	/// @brief FSGNJ/N/X: magnitude from rs1, sign from rs2. Also encodes FMV/FNEG/FABS.
	void emit_fsgnj(const rv32f_instruction& fi, bool dbl)
	{
		const Vec a = fget(fi.R4type.rs1), b = fget(fi.R4type.rs2);
		const Vec d = fdef(fi.R4type.rd);
		const Operand sign = dbl
			? uc.simd_const(&uc.ct().p_8000000000000000, Bcst::k64, d)
			: uc.simd_const(&uc.ct().p_8000000080000000, Bcst::k32, d);
		const Operand magnitude = dbl
			? uc.simd_const(&uc.ct().p_7FFFFFFFFFFFFFFF, Bcst::k64, d)
			: uc.simd_const(&uc.ct().p_7FFFFFFF7FFFFFFF, Bcst::k32, d);

		Vec sbit = uc.new_vec128("fsgnj");
		switch (fi.R4type.funct3) {
		case 0x0: // FSGNJ: rs2's sign
			v_and_fp(dbl, sbit, b, sign);
			break;
		case 0x1: // FSGNJN: rs2's sign, inverted
			v_andn_fp(dbl, sbit, b, sign);
			break;
		case 0x2: // FSGNJX: the two signs, xored
			v_xor_fp(dbl, sbit, a, b);
			v_and_fp(dbl, sbit, sbit, sign);
			break;
		}
		Vec mag = uc.new_vec128("fsgnjm");
		v_and_fp(dbl, mag, a, magnitude);
		v_or_fp(dbl, d, mag, sbit);
		fp_result(d, dbl);
	}

	/// @brief FEQ/FLT/FLE → integer 0 or 1. Ordered comparisons; NaN yields false.
	void emit_fcmp(const rv32f_instruction& fi, bool dbl)
	{
		const Vec a = fget(fi.R4type.rs1), b = fget(fi.R4type.rs2);
		Vec m = uc.new_vec128("fcmp");
		switch (fi.R4type.funct3) {
		case 0x0: if (dbl) uc.s_cmp_le_f64(m, a, b); else uc.s_cmp_le_f32(m, a, b); break;
		case 0x1: if (dbl) uc.s_cmp_lt_f64(m, a, b); else uc.s_cmp_lt_f32(m, a, b); break;
		case 0x2: if (dbl) uc.s_cmp_eq_f64(m, a, b); else uc.s_cmp_eq_f32(m, a, b); break;
		}
		// Extract one bit from the all-ones/all-zeros mask.
		Gp t = uc.new_gp32("fcmpr");
		uc.s_extract_u32(t, m, 0);
		uc.and_(t, t, Imm(1));
		set_word_result(fi.R4type.rd, t);
	}

	/// @brief FCVT.S.D / FCVT.D.S: canonicalize NaN in the source format before converting.
	void emit_fcvt_sd_ds(const rv32f_instruction& fi, bool to_double)
	{
		const Vec a = fget(fi.R4type.rs1);
		const Vec d = fdef(fi.R4type.rd);
		Vec m = uc.new_vec128("fnan");
		Vec t = uc.new_vec128("fcvt");
		if (to_double) {                      // FCVT.D.S
			uc.s_cmp_unord_f32(m, a, a);
			select_masked(false, t, m, canon32, a);
			uc.s_cvt_f32_to_f64(d, t);
		} else {                              // FCVT.S.D
			uc.s_cmp_unord_f64(m, a, a);
			select_masked(true, t, m, canon64, a);
			uc.s_cvt_f64_to_f32(d, t);
			nanbox(d);
		}
	}

	/// @brief FCVT.{W,WU,L,LU}.{S,D}: helper call (host disagrees on NaN/overflow saturation and RM).
	void emit_fcvt_to_int(const rv32f_instruction& fi, bool dbl)
	{
		const auto* cb = info.cb;
		const void* fn = nullptr;
		switch (fi.R4type.rs2) {
		case 0x0: fn = dbl ? (const void*)cb->fcvt_w_d  : (const void*)cb->fcvt_w_s;  break;
		case 0x1: fn = dbl ? (const void*)cb->fcvt_wu_d : (const void*)cb->fcvt_wu_s; break;
		case 0x2: fn = dbl ? (const void*)cb->fcvt_l_d  : (const void*)cb->fcvt_l_s;  break;
		case 0x3: fn = dbl ? (const void*)cb->fcvt_lu_d : (const void*)cb->fcvt_lu_s; break;
		}
		Gp rm = uc.new_gp32("frm");
		uc.mov(rm, Imm(fi.R4type.funct3));
		InvokeNode* node;
		cc.invoke(Out(node), uint64_t(uintptr_t(fn)),
			dbl ? FuncSignature::build<address_t, void*, double, uint32_t>()
				: FuncSignature::build<address_t, void*, float, uint32_t>());
		node->set_arg(0, cpu);
		node->set_arg(1, fget(fi.R4type.rs1));
		node->set_arg(2, rm);
		node->set_ret(0, def(fi.R4type.rd));
	}

	/// @brief FCVT.{S,D}.{W,WU,L,LU}: inline except LU (no host instruction for uint64→float).
	void emit_fcvt_from_int(const rv32f_instruction& fi, bool dbl)
	{
		const Vec d = fdef(fi.R4type.rd);
		const Gp src = get(fi.R4type.rs1);
		switch (fi.R4type.rs2) {
		case 0x0: // FCVT.{S,D}.W
			if (dbl) uc.s_cvt_int_to_f64(d, src.r32());
			else     uc.s_cvt_int_to_f32(d, src.r32());
			break;
		case 0x1: { // FCVT.{S,D}.WU -- widening to a signed 64-bit source is exact
			Gp z = uc.new_gp64("fzx");
			uc.mov(z.r32(), src.r32());   // writing the low half clears the upper
			if (dbl) uc.s_cvt_int_to_f64(d, z);
			else     uc.s_cvt_int_to_f32(d, z);
			} break;
		case 0x2: // FCVT.{S,D}.L
			if (dbl) uc.s_cvt_int_to_f64(d, src);
			else     uc.s_cvt_int_to_f32(d, src);
			break;
		case 0x3: { // FCVT.{S,D}.LU
			const void* fn = dbl ? (const void*)info.cb->fcvt_d_lu
								 : (const void*)info.cb->fcvt_s_lu;
			InvokeNode* node;
			cc.invoke(Out(node), uint64_t(uintptr_t(fn)),
				dbl ? FuncSignature::build<double, uint64_t>()
					: FuncSignature::build<float, uint64_t>());
			node->set_arg(0, src);
			node->set_ret(0, d);
			} break;
		}
		fp_result(d, dbl);
	}

	/// @brief OP-FP dispatch: all FP ops except loads, stores, and FMA.
	void emit_fpfunc(rv32i_instruction i)
	{
		const rv32f_instruction fi { i };
		const unsigned rd = fi.R4type.rd, rs1 = fi.R4type.rs1, rs2 = fi.R4type.rs2;
		const bool dbl = (fi.R4type.funct2 == 0x1);

		switch (i.fpfunc())
		{
		case RV32F__FADD:
			if (dbl) uc.s_add_f64(fdef(rd), fget(rs1), fget(rs2));
			else     uc.s_add_f32(fdef(rd), fget(rs1), fget(rs2));
			fp_result(fdef(rd), dbl);
			break;
		case RV32F__FSUB:
			if (dbl) uc.s_sub_f64(fdef(rd), fget(rs1), fget(rs2));
			else     uc.s_sub_f32(fdef(rd), fget(rs1), fget(rs2));
			fp_result(fdef(rd), dbl);
			break;
		case RV32F__FMUL:
			if (dbl) uc.s_mul_f64(fdef(rd), fget(rs1), fget(rs2));
			else     uc.s_mul_f32(fdef(rd), fget(rs1), fget(rs2));
			fp_result(fdef(rd), dbl);
			break;
		case RV32F__FDIV:
			if (dbl) uc.s_div_f64(fdef(rd), fget(rs1), fget(rs2));
			else     uc.s_div_f32(fdef(rd), fget(rs1), fget(rs2));
			fp_result(fdef(rd), dbl);
			break;
		case RV32F__FSQRT:
			if (dbl) uc.s_sqrt_f64(fdef(rd), fget(rs1));
			else     uc.s_sqrt_f32(fdef(rd), fget(rs1));
			fp_result(fdef(rd), dbl);
			break;
		case RV32F__FSGNJ_NX:
			emit_fsgnj(fi, dbl);
			break;
		case RV32F__FMIN_MAX:
			emit_fminmax(fi, dbl);
			break;
		case RV32F__FEQ_LT_LE:
			emit_fcmp(fi, dbl);
			break;
		case RV32F__FCVT_SD_DS:
			// funct2 names the *destination* format here, not the source.
			emit_fcvt_sd_ds(fi, dbl);
			break;
		case RV32F__FCVT_W_SD:
			emit_fcvt_to_int(fi, dbl);
			break;
		case RV32F__FCVT_SD_W:
			emit_fcvt_from_int(fi, dbl);
			break;
		case RV32F__FMV_X_W:
			if (fi.R4type.funct3 == 0x0) {
				if (dbl) {                       // FMV.X.D, RV64 only
					if constexpr (RV64) uc.s_mov_u64(def(rd), fget(rs1));
				} else {                         // FMV.X.W
					Gp t = uc.new_gp32("fmvx");
					uc.s_extract_u32(t, fget(rs1), 0);
					set_word_result(rd, t);
				}
			} else {                             // FCLASS
				Gp bits = dbl ? uc.new_gp64("fcls") : uc.new_gp32("fcls");
				if (dbl) uc.s_mov_u64(bits, fget(rs1));
				else     uc.s_extract_u32(bits, fget(rs1), 0);
				InvokeNode* node;
				cc.invoke(Out(node), uint64_t(uintptr_t(dbl
						? (const void*)info.cb->fclass64
						: (const void*)info.cb->fclass32)),
					dbl ? FuncSignature::build<address_t, uint64_t>()
						: FuncSignature::build<address_t, uint32_t>());
				node->set_arg(0, bits);
				node->set_ret(0, def(rd));
			}
			break;
		case RV32F__FMV_W_X:
			if (dbl) {                           // FMV.D.X, RV64 only
				if constexpr (RV64) uc.s_mov_u64(fdef(rd), get(rs1));
			} else {                             // FMV.W.X
				uc.s_mov_u32(fdef(rd), get(rs1).r32());
				nanbox(fdef(rd));
			}
			break;
		}
	}

	// --- pre-pass ---
	// Vector transfers need the base address register and the arena; vregs/vl/vtype stay in memory.
	void prepass_vector_memory(rv32i_instruction i) {
		const auto vmi = aj_vector_memory_form(i);
		if (!vector_memory_inlinable(vmi))
			return;
		readset.set(vmi.rs1);
		needs_arena = true;
	}

	// Single walk: collects read-set, write-set, and branch targets. Read-set must mirror get() usage.
	void prepass() {
		for (const address_t pc : instrs) {
			const auto i = aj_decode<W>(seg, pc, seg_end).instr;
			switch (i.opcode()) {
			case Dyncall::opcode:
				// Outputs must join the static write-set even when no guest
				// instruction writes them: exits must retain handler results.
				for (unsigned r = 10; r < 10 + Dyncall::outputs(i.whole); r++)
					writeset.set(r);
				break;
			case RV32I_LUI:
			case RV32I_AUIPC:
				writeset.set(i.Utype.rd);
				break;
			case RV32I_OP_IMM:
				// ADDI with rs1 == x0 is folded into a plain immediate load
				if (!(i.Itype.funct3 == 0x0 && i.Itype.rs1 == 0))
					readset.set(i.Itype.rs1);
				writeset.set(i.Itype.rd);
				break;
			case RV64I_OP_IMM32:
				readset.set(i.Itype.rs1);
				writeset.set(i.Itype.rd);
				break;
			case RV32I_OP:
			case RV64I_OP32:
				readset.set(i.Rtype.rs1);
				readset.set(i.Rtype.rs2);
				writeset.set(i.Rtype.rd);
				break;
			case RV32I_LOAD:
				readset.set(i.Itype.rs1);      // x0 folds into the immediate
				writeset.set(i.Itype.rd);
				needs_arena |= info.inline_memory;
				break;
			case RV32I_STORE:
				readset.set(i.Stype.rs1);      // x0 folds into the immediate
				readset.set(i.Stype.rs2);      // ...but the stored value does not
				needs_arena |= info.inline_memory;
				break;
			case RV32I_JALR:
				readset.set(i.Itype.rs1);
				writeset.set(i.Itype.rd);
				break;
			case RV32I_BRANCH: {
				readset.set(i.Btype.rs1);
				// A comparison against x0 becomes a compare against an immediate
				if (i.Btype.rs2 != 0)
					readset.set(i.Btype.rs2);
				const address_t t = pc + i.Btype.signed_imm();
				if (in_region(t)) branch_targets.insert(t);
				} break;
			case RV32I_JAL: {
				writeset.set(i.Jtype.rd);
				const address_t t = pc + i.Jtype.jump_offset();
				if (in_region(t)) branch_targets.insert(t);
				} break;

			case RV32I_SYSTEM:
				// Zimop writes zero to rd. The CSR instructions go to the
				// handler, which works on the register file in memory.
				if (i.Itype.funct3 == 0x4)
					writeset.set(i.Itype.rd);
				break;

			// Non-native F/D: handler call writes/reads via memory, so no set membership needed.
			case RV32F_LOAD:               // FLW, FLD
				if (!aj_fp_native<W>(i)) { prepass_vector_memory(i); break; }
				readset.set(i.Itype.rs1);
				fp_writeset.set(i.Itype.rd);
				needs_arena |= info.inline_memory;
				break;
			case RV32F_STORE:              // FSW, FSD
				if (!aj_fp_native<W>(i)) { prepass_vector_memory(i); break; }
				readset.set(i.Stype.rs1);
				fp_readset.set(i.Stype.rs2);
				needs_arena |= info.inline_memory;
				break;
			case RV32F_FMADD:
			case RV32F_FMSUB:
			case RV32F_FNMADD:
			case RV32F_FNMSUB: {
				if (!aj_fp_native<W>(i)) break;
				const rv32f_instruction fi { i };
				fp_readset.set(fi.R4type.rs1);
				fp_readset.set(fi.R4type.rs2);
				fp_readset.set(fi.R4type.rs3);
				fp_writeset.set(fi.R4type.rd);
				} break;
			case RV32F_FPFUNC: {
				if (!aj_fp_native<W>(i)) break;
				const rv32f_instruction fi { i };
				const unsigned rd = fi.R4type.rd, rs1 = fi.R4type.rs1, rs2 = fi.R4type.rs2;
				const bool dbl = (fi.R4type.funct2 == 0x1);
				switch (i.fpfunc()) {
				case RV32F__FADD: case RV32F__FSUB:
				case RV32F__FMUL: case RV32F__FDIV:
				case RV32F__FSGNJ_NX:
					fp_readset.set(rs1); fp_readset.set(rs2); fp_writeset.set(rd);
					break;
				case RV32F__FMIN_MAX:
					fp_readset.set(rs1); fp_readset.set(rs2); fp_writeset.set(rd);
					if (dbl) needs_canon64 = true; else needs_canon32 = true;
					break;
				case RV32F__FSQRT:
					fp_readset.set(rs1); fp_writeset.set(rd);
					break;
				case RV32F__FCVT_SD_DS:
					fp_readset.set(rs1); fp_writeset.set(rd);
					// Canonicalize in the source format.
					if (dbl) needs_canon32 = true; else needs_canon64 = true;
					break;
				case RV32F__FEQ_LT_LE:
					fp_readset.set(rs1); fp_readset.set(rs2); writeset.set(rd);
					break;
				case RV32F__FCVT_W_SD:
					fp_readset.set(rs1); writeset.set(rd);
					break;
				case RV32F__FCVT_SD_W:
					readset.set(rs1); fp_writeset.set(rd);
					break;
				case RV32F__FMV_X_W:   // and FCLASS
					fp_readset.set(rs1); writeset.set(rd);
					break;
				case RV32F__FMV_W_X:
					readset.set(rs1); fp_writeset.set(rd);
					break;
				default:
					break;
				}
				} break;

			default:
				break;
			}
		}
		// Conservative: an unused preload costs one prologue instruction.
		readset |= writeset;      // written registers must be loaded too, see flush_regs
		needs_zero = readset[0];
		readset.reset(0);
		writeset.reset(0);
		// f0 is an ordinary register, so the f-register sets keep every bit.
		fp_readset |= fp_writeset;

		// Each entry needs a label; a lone entry at the lowest address falls through.
		if (needs_entry_dispatch())
			branch_targets.insert(entries.begin(), entries.end());
		else if (!entry_is_first())
			branch_targets.insert(entry);

		for (const address_t t : branch_targets)
			labels.emplace(t, uc.new_label());
	}

	// --- emission ---
	bool failed = false;

	// Multi-entry dispatch: binary search on AjState::pc, O(log n) compares.
	void emit_entry_dispatch()
	{
		Gp pcv = new_ireg("entry_pc");
		uc.load(pcv, mem_ptr(st, off_pc()));
		// Function entry dominates in practice. Avoid the binary search entirely
		// for that overwhelmingly common case.
		uc.j(label_at(entries.front()), cmp_eq(pcv, rvimm(entries.front())));
		emit_entry_search(pcv, 1, entries.size());
	}
	void emit_entry_search(const Gp& pcv, size_t lo, size_t hi)
	{
		if (hi - lo == 1) {
			uc.j(label_at(entries[lo]));
			return;
		}
		const size_t mid = lo + (hi - lo) / 2;
		Label lower = uc.new_label();
		uc.j(lower, ucmp_lt(pcv, rvimm(entries[mid])));
		emit_entry_search(pcv, mid, hi);
		uc.bind(lower);
		emit_entry_search(pcv, lo, mid);
	}

	void emit_body()
	{
		// Labels bound only at branch targets. `fallthrough_pc` tracks linear control flow;
		// 0 means the previous instruction did not fall through.
		if (needs_entry_dispatch())
			emit_entry_dispatch();
		else if (!entry_is_first())
			uc.j(label_at(entry));
		address_t fallthrough_pc =
			(!needs_entry_dispatch() && entry_is_first()) ? instrs.front() : 0;
		for (const address_t pc : instrs)
		{
			const bool is_target = branch_targets.count(pc) != 0;
			if (is_target) {   // merge point: settle the counter first
				if (pc == fallthrough_pc) flush_counter();
				else pending = 0;   // only reachable through the label
				uc.bind(label_at(pc));
			} else if (pc != fallthrough_pc) {
				// Unreachable address: discovery guarantees reachability from the entry.
				failed = true;
				return;
			}

			const auto d = aj_decode<W>(seg, pc, seg_end);
			const auto i = d.instr;
			const address_t next = pc + d.length;
			fallthrough_pc = next;
			pending++;

			switch (i.opcode())
			{
			case Dyncall::opcode:
				emit_dyncall(pc, i);
				break;
			case RV32I_LUI:
				// The 20-bit upper immediate is sign-extended to the register width.
				uc.mov(def(i.Utype.rd), rvimm(address_t(int64_t(i.Utype.upper_imm()))));
				break;

			case RV32I_AUIPC:
				uc.mov(def(i.Utype.rd), rvimm(address_t(pc + address_t(int64_t(i.Utype.upper_imm())))));
				break;

			case RV32I_FENCE:
				if (i.Itype.funct3 == 0x0) {
					// Match the interpreter's full-barrier FENCE semantics.
					host_fence(cc);
				} else if (aj_handler_classify<W>(i) == AjHandler::kCboZero) {
					emit_handler_call(pc, i);
				}
				// Zicbom CBO ops are no-ops (no guest cache to act on).
				break;

			case RV32I_SYSTEM:
				if (i.Itype.funct3 == 0x4) {
					// Zimop: MOP writes zero to rd.
					if (i.Itype.rd != 0)
						uc.mov(def(i.Itype.rd), Imm(0));
				} else if (i.Itype.funct3 != 0x0) {
					emit_handler_call(pc, i);   // the CSR instructions
				}
				// Zawrs WRS.NTO/WRS.STO: always returns immediately, nothing to emit.
				break;

			case RV32A_ATOMIC:
				// LR/SC and AMOs go through the interpreter.
				emit_handler_call(pc, i);
				break;

			case RV32I_OP_IMM: {
				// Zb* first: BSETI/RORI share funct3 with SLLI/SRLI/SRAI.
				if (const AjZb zb = aj_zb_classify<W>(i); zb != AjZb::kNone) {
					emit_zb(zb, i);
					break;
				}
				const unsigned rd = i.Itype.rd, rs1 = i.Itype.rs1;
				const int32_t simm = i.Itype.signed_imm();
				const uint32_t shamt = RV64 ? i.Itype.shift64_imm() : i.Itype.shift_imm();
				switch (i.Itype.funct3) {
				case 0x0: // ADDI
					if (rs1 == 0) uc.mov(def(rd), rvimm(address_t(int64_t(simm))));
					else          alu_ri(OP_ADD, def(rd), get(rs1), simm);
					break;
				case 0x1: shift_ri(0, def(rd), get(rs1), shamt); break;     // SLLI
				case 0x2: setcc_ri(false, rd, rs1, simm); break;            // SLTI
				case 0x3: setcc_ri(true,  rd, rs1, simm); break;            // SLTIU
				case 0x4: alu_ri(OP_XOR, def(rd), get(rs1), simm); break;   // XORI
				case 0x5: // SRLI / SRAI
					shift_ri((i.Itype.imm & 0x400) ? 2 : 1, def(rd), get(rs1), shamt);
					break;
				case 0x6: alu_ri(OP_OR,  def(rd), get(rs1), simm); break;   // ORI
				case 0x7: alu_ri(OP_AND, def(rd), get(rs1), simm); break;   // ANDI
				}
				} break;

			case RV32I_OP: {
				// Zb* first: ANDN/ORN/XNOR share funct7 0x20 with SUB/SRA.
				if (const AjZb zb = aj_zb_classify<W>(i); zb != AjZb::kNone) {
					emit_zb(zb, i);
					break;
				}
				const unsigned rd = i.Rtype.rd, rs1 = i.Rtype.rs1, rs2 = i.Rtype.rs2;
				const bool alt = (i.Rtype.funct7 == 0b0100000);
				if (i.Rtype.funct7 == 0b0000001) {   // RV32M / RV64M
					emit_muldiv(i.Rtype.funct3, def(rd), get(rs1), get(rs2));
					break;
				}
				switch (i.Rtype.funct3) {
				case 0x0: // ADD / SUB
					alu_rr(alt ? OP_SUB : OP_ADD, def(rd), get(rs1), get(rs2));
					break;
				case 0x1: shift_rr(0, def(rd), get(rs1), get(rs2)); break;         // SLL
				case 0x2: setcc_rr(false, rd, rs1, rs2); break;                    // SLT
				case 0x3: setcc_rr(true,  rd, rs1, rs2); break;                    // SLTU
				case 0x4: alu_rr(OP_XOR, def(rd), get(rs1), get(rs2)); break;      // XOR
				case 0x5: shift_rr(alt ? 2 : 1, def(rd), get(rs1), get(rs2)); break; // SRL / SRA
				case 0x6: alu_rr(OP_OR,  def(rd), get(rs1), get(rs2)); break;      // OR
				case 0x7: alu_rr(OP_AND, def(rd), get(rs1), get(rs2)); break;      // AND
				}
				} break;

			case RV64I_OP_IMM32: if constexpr (RV64) {
				// Zb* first: SLLI.UW/RORIW share funct3 with SLLIW/SRLIW.
				if (const AjZb zb = aj_zb_classify<W>(i); zb != AjZb::kNone) {
					emit_zb(zb, i);
					break;
				}
				const unsigned rd = i.Itype.rd, rs1 = i.Itype.rs1;
				Gp t = uc.new_gp32("w");
				switch (i.Itype.funct3) {
				case 0x0: // ADDIW
					alu_ri(OP_ADD, t, word_of(rs1), i.Itype.signed_imm());
					break;
				case 0x1: // SLLIW
					shift_ri(0, t, word_of(rs1), i.Itype.shift_imm());
					break;
				case 0x5: // SRLIW / SRAIW
					shift_ri((i.Itype.imm & 0x400) ? 2 : 1, t, word_of(rs1), i.Itype.shift_imm());
					break;
				}
				finish_word(rd, t);
				} break;

			case RV64I_OP32: if constexpr (RV64) {
				// Zb* first (before word_of): .UW/ZEXT.H produce full 64-bit results.
				if (const AjZb zb = aj_zb_classify<W>(i); zb != AjZb::kNone) {
					emit_zb(zb, i);
					break;
				}
				const unsigned rd = i.Rtype.rd, rs1 = i.Rtype.rs1, rs2 = i.Rtype.rs2;
				const bool alt = (i.Rtype.funct7 == 0b0100000);
				Gp t = uc.new_gp32("w");
				if (i.Rtype.funct7 == 0b0000001) {   // MULW DIVW DIVUW REMW REMUW
					emit_muldiv(i.Rtype.funct3, t, word_of(rs1), word_of(rs2));
					finish_word(rd, t);
					break;
				}
				switch (i.Rtype.funct3) {
				case 0x0: // ADDW / SUBW
					alu_rr(alt ? OP_SUB : OP_ADD, t, word_of(rs1), word_of(rs2));
					break;
				case 0x1: // SLLW
					shift_rr(0, t, word_of(rs1), word_of(rs2));
					break;
				case 0x5: // SRLW / SRAW
					shift_rr(alt ? 2 : 1, t, word_of(rs1), word_of(rs2));
					break;
				}
				finish_word(rd, t);
				} break;

			case RV32I_LOAD:
				emit_load(pc, i.Itype.funct3, i.Itype.rd, i.Itype.rs1, i.Itype.signed_imm());
				break;

			case RV32I_STORE:
				emit_store(pc, i.Stype.funct3, i.Stype.rs1, i.Stype.rs2, i.Stype.signed_imm());
				break;

			// F/D split: native forms are inlined, the rest (Zfa, Zfhmin, etc.) go to handler.
			case RV32F_LOAD:
				if (aj_fp_native<W>(i))
					emit_fp_load(pc, i.Itype.funct3, i.Itype.rd, i.Itype.rs1, i.Itype.signed_imm());
				else if (const auto vmi = aj_vector_memory_form(i); vector_memory_inlinable(vmi))
					emit_vector_memory(pc, vmi, i);
				else
					emit_handler_call(pc, i);
				break;

			case RV32F_STORE:
				if (aj_fp_native<W>(i))
					emit_fp_store(pc, i.Stype.funct3, i.Stype.rs1, i.Stype.rs2, i.Stype.signed_imm());
				else if (const auto vmi = aj_vector_memory_form(i); vector_memory_inlinable(vmi))
					emit_vector_memory(pc, vmi, i);
				else
					emit_handler_call(pc, i);
				break;

			case RV32V_OP:
				// vsetvl, arithmetic, and permutes go to handler; only memory ops are inlined.
				emit_handler_call(pc, i);
				break;

			case RV32F_FMADD:
			case RV32F_FMSUB:
			case RV32F_FNMADD:
			case RV32F_FNMSUB:
				if (aj_fp_native<W>(i))
					emit_fmadd(i);
				else
					emit_handler_call(pc, i);
				break;

			case RV32F_FPFUNC:
				if (aj_fp_native<W>(i))
					emit_fpfunc(i);
				else
					emit_handler_call(pc, i);
				break;

			case RV32I_BRANCH: {
				const address_t target = pc + i.Btype.signed_imm();
				flush_counter();                 // settle before control flow splits
				const Gp a = get(i.Btype.rs1);
				const bool vs_zero = (i.Btype.rs2 == 0);
				const Gp b = vs_zero ? a : get(i.Btype.rs2);   // unused when vs_zero
				// When no instruction-limit check must sit on the taken path, branch
				// directly to the region label instead of jcc-skip + jmp-target.
				if (in_region(target) && (info.ignore_instruction_limit || target > pc)) {
					const Label L = label_at(target);
					switch (i.Btype.funct3) {
					case 0x0: uc.j(L, vs_zero ? cmp_eq(a, Imm(0))  : cmp_eq(a, b)); break;
					case 0x1: uc.j(L, vs_zero ? cmp_ne(a, Imm(0))  : cmp_ne(a, b)); break;
					case 0x4: uc.j(L, vs_zero ? scmp_lt(a, Imm(0)) : scmp_lt(a, b)); break;
					case 0x5: uc.j(L, vs_zero ? scmp_ge(a, Imm(0)) : scmp_ge(a, b)); break;
					case 0x6: uc.j(L, vs_zero ? ucmp_lt(a, Imm(0)) : ucmp_lt(a, b)); break;
					case 0x7: uc.j(L, vs_zero ? ucmp_ge(a, Imm(0)) : ucmp_ge(a, b)); break;
					}
					break;
				}
				Label notaken = uc.new_label();
				// Inverted condition jumps around the taken path (keeps taken inline).
				switch (i.Btype.funct3) {
				case 0x0: // BEQ
					uc.j(notaken, vs_zero ? cmp_ne(a, Imm(0)) : cmp_ne(a, b)); break;
				case 0x1: // BNE
					uc.j(notaken, vs_zero ? cmp_eq(a, Imm(0)) : cmp_eq(a, b)); break;
				case 0x4: // BLT
					uc.j(notaken, vs_zero ? scmp_ge(a, Imm(0)) : scmp_ge(a, b)); break;
				case 0x5: // BGE
					uc.j(notaken, vs_zero ? scmp_lt(a, Imm(0)) : scmp_lt(a, b)); break;
				case 0x6: // BLTU
					uc.j(notaken, vs_zero ? ucmp_ge(a, Imm(0)) : ucmp_ge(a, b)); break;
				case 0x7: // BGEU
					uc.j(notaken, vs_zero ? ucmp_lt(a, Imm(0)) : ucmp_lt(a, b)); break;
				}
				if (in_region(target)) {
					if (target <= pc) emit_backedge_check(target); // bound the loop
					uc.j(label_at(target));   // vregs stay live across the back-edge
				} else {
					emit_exit(target, 0);
				}
				uc.bind(notaken);
				} break;

			case RV32I_JAL: {
				const address_t target = pc + i.Jtype.jump_offset();
				bool continues = false;
				if (i.Jtype.rd != 0)
					uc.mov(def(i.Jtype.rd), rvimm(next));
				flush_counter();
				if (i.Jtype.rd != 0 && emit_direct_call(target, next)) {
					continues = true;
				} else if (in_region(target)) {
					if (target <= pc) emit_backedge_check(target);
					uc.j(label_at(target));
				} else {
					emit_exit(target, 0);
				}
				fallthrough_pc = continues ? next : 0;
				} break;

			case RV32I_JALR: {
				// Read rs1 before writing rd: they may alias, and JALR uses the old rs1.
				Gp target = new_ireg("jalr");
				const int32_t simm = i.Itype.signed_imm();
				if (i.Itype.rs1 == 0) {
					uc.mov(target, rvimm(address_t(int64_t(simm)) & ~address_t(1)));
				} else {
					alu_ri(OP_ADD, target, get(i.Itype.rs1), simm);
					uc.and_(target, target, sximm(-2));
				}
				if (i.Itype.rd != 0)
					uc.mov(def(i.Itype.rd), rvimm(next));
				flush_counter();
				emit_exit_reg(target, 0);
				fallthrough_pc = 0;
				} break;

			default:
				// Unreachable: discovery guaranteed emittability.
				emit_exit(pc, pending - 1);
				fallthrough_pc = 0;
				break;
			}

			// Exit when fall-through leaves the discovered instruction set.
			if (fallthrough_pc != 0 && !in_region(fallthrough_pc)) {
				emit_exit(fallthrough_pc, pending);
				pending = 0;
				fallthrough_pc = 0;
			}
		}

		// Emit deferred slow paths after the hot path.
		for (size_t n = 0; n < deferred.size(); n++)
			deferred[n]();
		emit_exit_tail();
	}
};

template <int W>
aj_block_func<W> aj_emit_region(AjCode& ajcode, const MachineOptions<W>& options,
	const DecodedExecuteSegment<W>& exec, const AjInfo<W>& info,
	const aj_block_func<W>* mappings,
	const std::vector<std::pair<address_type<W>, size_t>>& call_targets,
	const std::vector<address_type<W>>& entries,
	const std::vector<address_type<W>>& instrs)
{
	if (instrs.empty() || entries.empty())
		return nullptr;

	CodeHolder code;
	if (code.init(ajcode.rt.environment(), ajcode.rt.cpu_features()) != kErrorOk)
		return nullptr;

	StringLogger logger;
	if (options.asmjit_verbose)
		code.set_logger(&logger);

	BackendCompiler bc(&code);
	UniCompiler cc(&bc, ajcode.rt.cpu_features(), ajcode.rt.cpu_hints());

	FuncNode* fn = cc.add_func(FuncSignature::build<void, void*, void*>());
	if (fn == nullptr)
		return nullptr;

	AjEmitter<W> e { cc, exec.exec_data(), entries, mappings, call_targets,
		instrs, exec.exec_end(), info };
	e.cpu     = cc.new_gp_ptr("cpu");
	e.st      = cc.new_gp_ptr("state");
	if (!info.ignore_instruction_limit)
		e.counter = cc.new_gp64("counter");
	fn->set_arg(0, e.cpu);
	fn->set_arg(1, e.st);

	e.prepass();
	if (!info.ignore_instruction_limit)
		cc.load(e.counter, mem_ptr(e.st, AjEmitter<W>::off_counter()));
	e.emit_prologue_loads();   // every load lands here, ahead of every label
	e.emit_body();
	if (e.failed)
		return nullptr;

	cc.end_func();
	if (cc.finalize() != kErrorOk)
		return nullptr;

	aj_block_func<W> out = nullptr;
	if (ajcode.rt.add(&out, &code) != kErrorOk)
		return nullptr;

	if (options.asmjit_verbose) {
		printf("libriscv: asmjit region 0x%lX-0x%lX, %zu entry point(s) from 0x%lX (%zu instructions) ->\n%s",
			long(instrs.front()), long(instrs.back()), entries.size(),
			long(entries.front()), instrs.size(), logger.data());
	}
	return out;
}

#else // !RISCV_ASMJIT_HAS_BACKEND

bool aj_host_has_fma() noexcept
{
	return false;   // nothing is emitted on this host anyway
}

template <int W>
aj_block_func<W> aj_emit_region(AjCode&, const MachineOptions<W>&,
	const DecodedExecuteSegment<W>&, const AjInfo<W>&,
	const aj_block_func<W>*,
	const std::vector<std::pair<address_type<W>, size_t>>&,
	const std::vector<address_type<W>>&, const std::vector<address_type<W>>&)
{
	return nullptr;   // no code generator for this host
}

#endif

#ifdef RISCV_32I
	template aj_block_func<4> aj_emit_region<4>(AjCode&, const MachineOptions<4>&,
		const DecodedExecuteSegment<4>&, const AjInfo<4>&,
		const aj_block_func<4>*, const std::vector<std::pair<address_type<4>, size_t>>&,
		const std::vector<address_type<4>>&, const std::vector<address_type<4>>&);
#endif
#ifdef RISCV_64I
	template aj_block_func<8> aj_emit_region<8>(AjCode&, const MachineOptions<8>&,
		const DecodedExecuteSegment<8>&, const AjInfo<8>&,
		const aj_block_func<8>*, const std::vector<std::pair<address_type<8>, size_t>>&,
		const std::vector<address_type<8>>&, const std::vector<address_type<8>>&);
#endif
} // riscv
