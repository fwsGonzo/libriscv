#include "machine.hpp"
#include "decoder_cache.hpp"
#include "instruction_counter.hpp"
#include "threaded_bytecodes.hpp"
#include "rv32i_instr.hpp"
#include "rvfd.hpp"
#ifdef RISCV_EXT_COMPRESSED
#include "rvc.hpp"
#endif
#ifdef RISCV_EXT_VECTOR
#include "rvv.hpp"
#endif

/**
 * This file is included by threaded_dispatch.cpp and bytecode_dispatch.cpp
 * It implements the logic for switch-based and threaded dispatch.
 * 
 * All dispatch modes share bytecode_impl.cpp
**/

namespace riscv
{
	static constexpr bool VERBOSE_JUMPS = riscv::verbose_branches_enabled;
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
#warning "The emulator is currently built with sanitizer and fuzzing mode enabled"
	static constexpr bool FUZZING = true;
#else
	static constexpr bool FUZZING = false;
#endif

#define VIEW_INSTR() \
	auto instr = *(rv32i_instruction *)&decoder->instr;
#define VIEW_INSTR_AS(name, x) \
	auto &&name = *(x *)&decoder->instr;
#define NEXT_INSTR()                  \
	if constexpr (compressed_enabled) \
		decoder += 2;                 \
	else                              \
		decoder += 1;                 \
	EXECUTE_INSTR();
#define NEXT_C_INSTR() \
	decoder += 1;      \
	EXECUTE_INSTR();

#define NEXT_BLOCK(len, OF)                 \
	pc += len;                              \
	decoder += len >> DecoderCache<W>::SHIFT;              \
	if constexpr (FUZZING) /* Give OOB-aid to ASAN */      \
	decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT]; \
	if constexpr (OF) {						\
		if (UNLIKELY(counter.overflowed())) \
			goto check_jump;				\
	}										\
	pc += decoder->block_bytes();                            \
	counter.increment_counter(decoder->instruction_count()); \
	EXECUTE_INSTR();

#define NEXT_SEGMENT()                                       \
	decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT];  \
	pc += decoder->block_bytes();                            \
	counter.increment_counter(decoder->instruction_count()); \
	EXECUTE_INSTR();

#define PERFORM_BRANCH()                 \
	if constexpr (VERBOSE_JUMPS) fprintf(stderr, "Branch 0x%lX >= 0x%lX (decoder=%p)\n", long(pc), long(pc + fi.signed_imm()), decoder); \
	if (LIKELY(!counter.overflowed())) { \
		NEXT_BLOCK(fi.signed_imm(), false);     \
	}                                    \
	pc += fi.signed_imm();               \
	goto check_jump;

#define PERFORM_FORWARD_BRANCH()         \
	if constexpr (VERBOSE_JUMPS) fprintf(stderr, "Fw.Branch 0x%lX >= 0x%lX\n", long(pc), long(pc + fi.signed_imm())); \
	NEXT_BLOCK(fi.signed_imm(), false);

#define OVERFLOW_CHECKED_JUMP() \
	goto check_jump

// We will not use the undefined sanitizer outside of
// sanitizing and fuzzing modes. Sanitizers are not effective
// anyway without defining FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION.
#if defined(__GNUG__) && !defined(FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION)
#define DISPATCH_ATTR RISCV_HOT_PATH() __attribute__((no_sanitize("undefined")))
#else
#define DISPATCH_ATTR RISCV_HOT_PATH()
#endif

template <int W> DISPATCH_ATTR
bool CPU<W>::simulate(address_t pc, uint64_t inscounter, uint64_t maxcounter)
{
	static constexpr uint32_t XLEN = W * 8;
	using addr_t  = address_type<W>;
	using saddr_t = signed_address_type<W>;

#ifdef DISPATCH_MODE_THREADED
	static constexpr void *computed_opcode[] = {
		[RV32I_BC_INVALID] = &&execute_invalid,
		[RV32I_BC_ADDI]    = &&rv32i_addi,
		[RV32I_BC_LI]      = &&rv32i_li,
		[RV32I_BC_MV]      = &&rv32i_mv,
		[RV32I_BC_SLLI]    = &&rv32i_slli,
		[RV32I_BC_SLTI]    = &&rv32i_slti,
		[RV32I_BC_SLTIU]   = &&rv32i_sltiu,
		[RV32I_BC_XORI]    = &&rv32i_xori,
		[RV32I_BC_SRLI]    = &&rv32i_srli,
		[RV32I_BC_SRAI]    = &&rv32i_srai,
		[RV32I_BC_ORI]     = &&rv32i_ori,
		[RV32I_BC_ANDI]    = &&rv32i_andi,

		[RV32I_BC_LUI]     = &&rv32i_lui,
		[RV32I_BC_AUIPC]   = &&rv32i_auipc,

		[RV32I_BC_LDB]     = &&rv32i_ldb,
		[RV32I_BC_LDBU]    = &&rv32i_ldbu,
		[RV32I_BC_LDH]     = &&rv32i_ldh,
		[RV32I_BC_LDHU]    = &&rv32i_ldhu,
		[RV32I_BC_LDW]     = &&rv32i_ldw,

		[RV32I_BC_STB]     = &&rv32i_stb,
		[RV32I_BC_STH]     = &&rv32i_sth,
		[RV32I_BC_STW]     = &&rv32i_stw,
#ifdef RISCV_64I
		[RV32I_BC_LDWU]    = &&rv32i_ldwu,
		[RV32I_BC_LDD]     = &&rv32i_ldd,
		[RV32I_BC_STD]     = &&rv32i_std,
#endif

		[RV32I_BC_BEQ]     = &&rv32i_beq,
		[RV32I_BC_BNE]     = &&rv32i_bne,
		[RV32I_BC_BLT]     = &&rv32i_blt,
		[RV32I_BC_BGE]     = &&rv32i_bge,
		[RV32I_BC_BLTU]    = &&rv32i_bltu,
		[RV32I_BC_BGEU]    = &&rv32i_bgeu,
		[RV32I_BC_BEQ_FW]  = &&rv32i_beq_fw,
		[RV32I_BC_BNE_FW]  = &&rv32i_bne_fw,

		[RV32I_BC_JAL]     = &&rv32i_jal,
		[RV32I_BC_JALR]    = &&rv32i_jalr,
		[RV32I_BC_FAST_JAL] = &&rv32i_fast_jal,
		[RV32I_BC_FAST_CALL] = &&rv32i_fast_call,

		[RV32I_BC_OP_ADD]  = &&rv32i_op_add,
		[RV32I_BC_OP_SUB]  = &&rv32i_op_sub,
		[RV32I_BC_OP_SLL]  = &&rv32i_op_sll,
		[RV32I_BC_OP_SLT]  = &&rv32i_op_slt,
		[RV32I_BC_OP_SLTU] = &&rv32i_op_sltu,
		[RV32I_BC_OP_XOR]  = &&rv32i_op_xor,
		[RV32I_BC_OP_SRL]  = &&rv32i_op_srl,
		[RV32I_BC_OP_OR]   = &&rv32i_op_or,
		[RV32I_BC_OP_AND]  = &&rv32i_op_and,
		[RV32I_BC_OP_MUL]  = &&rv32i_op_mul,
		[RV32I_BC_OP_DIV]  = &&rv32i_op_div,
		[RV32I_BC_OP_DIVU] = &&rv32i_op_divu,
		[RV32I_BC_OP_REM]  = &&rv32i_op_rem,
		[RV32I_BC_OP_REMU] = &&rv32i_op_remu,
		[RV32I_BC_OP_SRA]  = &&rv32i_op_sra,
		[RV32I_BC_OP_ZEXT_H] = &&rv32i_op_zext_h,
		[RV32I_BC_OP_SH1ADD] = &&rv32i_op_sh1add,
		[RV32I_BC_OP_SH2ADD] = &&rv32i_op_sh2add,
		[RV32I_BC_OP_SH3ADD] = &&rv32i_op_sh3add,

		[RV32I_BC_SEXT_B] = &&rv32i_sext_b,
		[RV32I_BC_SEXT_H] = &&rv32i_sext_h,
		[RV32I_BC_BSETI]  = &&rv32i_bseti,
		[RV32I_BC_BEXTI]  = &&rv32i_bexti,

#ifdef RISCV_64I
		[RV64I_BC_ADDIW]  = &&rv64i_addiw,
		[RV64I_BC_SLLIW]  = &&rv64i_slliw,
		[RV64I_BC_SRLIW]  = &&rv64i_srliw,
		[RV64I_BC_SRAIW]  = &&rv64i_sraiw,
		[RV64I_BC_OP_ADDW] = &&rv64i_op_addw,
		[RV64I_BC_OP_SUBW] = &&rv64i_op_subw,
		[RV64I_BC_OP_MULW] = &&rv64i_op_mulw,
		[RV64I_BC_OP_ADD_UW] = &&rv64i_op_add_uw,
		[RV64I_BC_OP_SH1ADD_UW] = &&rv64i_op_sh1add_uw,
		[RV64I_BC_OP_SH2ADD_UW] = &&rv64i_op_sh2add_uw,
#endif // RISCV_64I

#ifdef RISCV_EXT_COMPRESSED
		[RV32C_BC_ADDI]     = &&rv32c_addi,
		[RV32C_BC_LI]       = &&rv32c_addi,
		[RV32C_BC_MV]       = &&rv32c_mv,
		[RV32C_BC_SLLI]     = &&rv32c_slli,
		[RV32C_BC_BEQZ]     = &&rv32c_beqz,
		[RV32C_BC_BNEZ]     = &&rv32c_bnez,
		[RV32C_BC_JMP]      = &&rv32c_jmp,
		[RV32C_BC_JR]       = &&rv32c_jr,
		[RV32C_BC_JAL_ADDIW]= &&rv32c_jal_addiw,
		[RV32C_BC_JALR]     = &&rv32c_jalr,
		[RV32C_BC_LDD]      = &&rv32c_ldd,
		[RV32C_BC_STD]      = &&rv32c_std,
		[RV32C_BC_FUNCTION] = &&rv32c_func,
		[RV32C_BC_JUMPFUNC] = &&rv32c_jumpfunc,
#endif

		[RV32I_BC_SYSCALL] = &&rv32i_syscall,
		[RV32I_BC_STOP]    = &&rv32i_stop,
		[RV32I_BC_NOP]     = &&rv32i_nop,

		[RV32F_BC_FLW]     = &&rv32i_flw,
		[RV32F_BC_FLD]     = &&rv32i_fld,
		[RV32F_BC_FSW]     = &&rv32i_fsw,
		[RV32F_BC_FSD]     = &&rv32i_fsd,
		[RV32F_BC_FADD]    = &&rv32f_fadd,
		[RV32F_BC_FSUB]    = &&rv32f_fsub,
		[RV32F_BC_FMUL]    = &&rv32f_fmul,
		[RV32F_BC_FDIV]    = &&rv32f_fdiv,
		[RV32F_BC_FMADD]   = &&rv32f_fmadd,
#ifdef RISCV_EXT_VECTOR
		[RV32V_BC_VLE32]   = &&rv32v_vle32,
		[RV32V_BC_VSE32]   = &&rv32v_vse32,
		[RV32V_BC_VFADD_VV] = &&rv32v_vfadd_vv,
		[RV32V_BC_VFMUL_VF] = &&rv32v_vfmul_vf,
#endif
		[RV32I_BC_FUNCTION] = &&execute_decoded_function,
		[RV32I_BC_FUNCBLOCK] = &&execute_function_block,
#ifdef RISCV_BINARY_TRANSLATION
		[RV32I_BC_TRANSLATOR] = &&translated_function,
#endif
		[RV32I_BC_SYSTEM]  = &&rv32i_system,
	};
#endif

	DecodedExecuteSegment<W>* exec = this->m_exec;
	address_t current_begin = exec->exec_begin();
	address_t current_end   = exec->exec_end();

	DecoderData<W>* exec_decoder = exec->decoder_cache();
	DecoderData<W>* decoder;

	InstrCounter counter{inscounter, maxcounter};

	// We need an execute segment matching current PC
	if (UNLIKELY(!(pc >= current_begin && pc < current_end)))
		goto new_execute_segment;

continue_segment:
	decoder = &exec_decoder[pc >> DecoderCache<W>::SHIFT];
	pc += decoder->block_bytes();
	counter.increment_counter(decoder->instruction_count());

#ifdef DISPATCH_MODE_SWITCH_BASED

while (true) {
	switch (decoder->get_bytecode()) {
	#define INSTRUCTION(bc, lbl) case bc:

#else

	goto *computed_opcode[decoder->get_bytecode()];
	#define INSTRUCTION(bc, lbl) lbl:

#endif

#define DECODER()   (*decoder)
#define CPU()       (*this)
#define REG(x)      registers().get()[x]
#define REGISTERS() registers()
#define VECTORS()   registers().rvv()
#define MACHINE()   machine()

	/** Instruction handlers **/

#include "bytecode_impl.cpp"

INSTRUCTION(RV32I_BC_SYSTEM, rv32i_system) {
	VIEW_INSTR();
	// Make the current PC visible
	REGISTERS().pc = pc;
	// Make the instruction counters visible
	counter.apply(MACHINE());
	// Invoke SYSTEM
	MACHINE().system(instr);
	// Restore counters
	counter.retrieve_max_counter(MACHINE());
	// Overflow-check, next block
	NEXT_BLOCK(4, true);
}

#ifdef RISCV_BINARY_TRANSLATION
INSTRUCTION(RV32I_BC_TRANSLATOR, translated_function) {
	VIEW_INSTR();
	// Invoke translated code
	auto bintr_results = 
		exec->unchecked_mapping_at(instr.whole)(*this, counter.value()-1, counter.max(), pc);
	pc = REGISTERS().pc;
	counter.set_counters(bintr_results.counter, bintr_results.max_counter);
	if (LIKELY(!counter.overflowed() && (pc - current_begin < current_end - current_begin)))
		goto continue_segment;
	goto check_jump;
}
#endif // RISCV_BINARY_TRANSLATION

INSTRUCTION(RV32I_BC_SYSCALL, rv32i_syscall) {
	// Make the current PC visible
	REGISTERS().pc = pc;
	// Make the instruction counter(s) visible
	counter.apply(MACHINE());
	// Invoke system call
	MACHINE().system_call(REG(REG_ECALL));
	// Restore max counter
	counter.retrieve_max_counter(MACHINE());
	if (UNLIKELY(counter.overflowed() || pc != REGISTERS().pc))
	{
		// System calls are always full-length instructions
		if constexpr (VERBOSE_JUMPS) {
			if (pc != REGISTERS().pc)
			fprintf(stderr, "SYSCALL jump from 0x%lX to 0x%lX\n",
				long(pc), long(REGISTERS().pc + 4));
		}
		pc = REGISTERS().pc + 4;
		goto check_jump;
	}
	NEXT_BLOCK(4, false);
}

INSTRUCTION(RV32I_BC_STOP, rv32i_stop) {
	REGISTERS().pc = pc + 4;
	MACHINE().set_instruction_counter(counter.value());
	return true;
}

#ifdef DISPATCH_MODE_SWITCH_BASED
	default:
		goto execute_invalid;
	} // switch case
} // while loop

#endif

check_jump:
	if (UNLIKELY(counter.overflowed()))
		goto counter_overflow;

	if (LIKELY(pc - current_begin < current_end - current_begin))
		goto continue_segment;
	else
		goto new_execute_segment;

counter_overflow:
	registers().pc = pc;
	MACHINE().set_instruction_counter(counter.value());

	// Machine stopped normally?
	return counter.max() == 0;

	// Change to a new execute segment
new_execute_segment: {
		auto new_values = this->next_execute_segment(pc);
		exec = new_values.exec;
		pc   = new_values.pc;
		current_begin = exec->exec_begin();
		current_end   = exec->exec_end();
		exec_decoder  = exec->decoder_cache();
	}
	goto continue_segment;

execute_invalid:
	MACHINE().set_instruction_counter(counter.value());
	trigger_exception(ILLEGAL_OPCODE, decoder->instr);

} // CPU::simulate_XXX()

} // riscv
