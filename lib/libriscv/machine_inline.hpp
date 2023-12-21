
template <int W>
inline void Machine<W>::stop() noexcept {
	m_max_counter = 0;
}
template <int W>
inline bool Machine<W>::stopped() const noexcept {
	return m_counter >= m_max_counter;
}
template <int W>
inline bool Machine<W>::instruction_limit_reached() const noexcept {
	return m_counter >= m_max_counter && m_max_counter != 0;
}

template <int W>
void Machine<W>::penalize(uint64_t val) noexcept
{
	if (m_max_counter - val > m_max_counter)
		m_max_counter = 0;
	else
		m_max_counter -= val;
}

template <int W>
template <bool Throw>
inline bool Machine<W>::simulate(uint64_t max_instr, uint64_t counter)
{
	const auto new_max = cpu.simulate(counter, max_instr);
	if constexpr (Throw) {
		// The simulation ends normally (new_max == 0), or it throws an exception
		// Storing to m_max_counter is not helpful here.
		if (UNLIKELY(new_max))
			timeout_exception(max_instr);
		return false;
	} else {
		// Here m_max_counter is useful for instruction_limit_reached() and stopped().
		this->m_max_counter = new_max;
		return new_max == 0;
	}
}

template <int W>
template <bool Throw>
inline bool Machine<W>::resume(uint64_t max_instr)
{
	return this->simulate<Throw>(this->instruction_counter() + max_instr, this->instruction_counter());
}

template <int W>
inline void Machine<W>::reset()
{
	cpu.reset();
	memory.reset();
}

template <int W>
inline void Machine<W>::print(const char* buffer, size_t len) const
{
	this->m_printer(*this, buffer, len);
}
template <int W>
inline long Machine<W>::stdin_read(char* buffer, size_t len) const
{
	return this->m_stdin(*this, buffer, len);
}
template <int W>
inline void Machine<W>::debug_print(const char* buffer, size_t len) const
{
	this->m_debug_printer(*this, buffer, len);
}

template <int W> inline
void Machine<W>::install_syscall_handler(size_t sysn, syscall_t handler)
{
	syscall_handlers.at(sysn) = handler;
}
template <int W> inline
void Machine<W>::install_syscall_handlers(std::initializer_list<std::pair<size_t, syscall_t>> syscalls)
{
	for (auto& scall : syscalls)
		install_syscall_handler(scall.first, scall.second);
}

template <int W>
inline void Machine<W>::system_call(size_t sysnum)
{
	if (LIKELY(sysnum < syscall_handlers.size())) {
		Machine::syscall_handlers[RISCV_SPECSAFE(sysnum)](*this);
	} else {
		on_unhandled_syscall(*this, sysnum);
	}
}

template <int W>
template <typename T>
inline T Machine<W>::sysarg(int idx) const
{
	if constexpr (std::is_integral_v<T>) {
		// 64-bit integers on 32-bit uses 2 registers
		if constexpr (sizeof(T) > W) {
			return static_cast<T> (cpu.reg(REG_ARG0 + idx))
				| static_cast<T> (cpu.reg(REG_ARG0 + idx + 1)) << 32;
		}
		return static_cast<T> (cpu.reg(REG_ARG0 + idx));
	}
	else if constexpr (std::is_same_v<T, float>)
		return cpu.registers().getfl(REG_FA0 + idx).f32[0];
	else if constexpr (std::is_same_v<T, double>)
		return cpu.registers().getfl(REG_FA0 + idx).f64;
	else if constexpr (std::is_same_v<T, riscv::Buffer>)
		return memory.rvbuffer(
			cpu.reg(REG_ARG0 + idx), cpu.reg(REG_ARG0 + idx + 1));
	else if constexpr (std::is_same_v<T, std::basic_string_view<char>>)
		return memory.rvbuffer(
			cpu.reg(REG_ARG0 + idx), cpu.reg(REG_ARG0 + idx + 1)).strview();
	else if constexpr (is_stdstring<T>::value)
		return memory.memstring(cpu.reg(REG_ARG0 + idx));
	else if constexpr (std::is_standard_layout_v<remove_cvref<T>> && std::is_trivial_v<remove_cvref<T>>) {
		T value;
		memory.memcpy_out(&value, cpu.reg(REG_ARG0 + idx), sizeof(T));
		return value;
	} else
		static_assert(always_false<T>, "Unknown type");
}

template <int W>
template<typename... Args, std::size_t... Indices>
inline auto Machine<W>::resolve_args(std::index_sequence<Indices...>) const
{
	std::tuple<std::decay_t<Args>...> retval;
	size_t i = 0;
	size_t f = 0;
	([&] {
		if constexpr (std::is_integral_v<Args>) {
			std::get<Indices>(retval) = sysarg<Args>(i++);
			if constexpr (sizeof(Args) > W) i++; // uses 2 registers
		}
		else if constexpr (std::is_floating_point_v<Args>)
			std::get<Indices>(retval) = sysarg<Args>(f++);
		else if constexpr (std::is_same_v<Args, riscv::Buffer>) {
			std::get<Indices>(retval) = std::move(sysarg<Args>(i)); i += 2; // ptr, len
		}
		else if constexpr (std::is_same_v<Args, std::basic_string_view<char>>) {
			std::get<Indices>(retval) = sysarg<Args>(i); i+= 2;
		}
		else if constexpr (is_stdstring<Args>::value)
			std::get<Indices>(retval) = sysarg<Args>(i++);
		else if constexpr (std::is_standard_layout_v<remove_cvref<Args>> && std::is_trivial_v<remove_cvref<Args>>)
			std::get<Indices>(retval) = sysarg<Args>(i++);
		else
			static_assert(always_false<Args>, "Unknown type");
	}(), ...);
	return retval;
}

template <int W>
template<typename... Args>
inline auto Machine<W>::sysargs() const {
	return resolve_args<Args...>(std::index_sequence_for<Args...>{});
}

template <int W>
template <typename... Args>
inline void Machine<W>::set_result(Args... args) {
	size_t i = 0;
	size_t f = 0;
	([&] {
		if constexpr (std::is_integral_v<Args>) {
			cpu.registers().at(REG_ARG0 + i++) = args;
		}
		else if constexpr (std::is_same_v<Args, float>)
			cpu.registers().getfl(REG_FA0 + f++).set_float(args);
		else if constexpr (std::is_same_v<Args, double>)
			cpu.registers().getfl(REG_FA0 + f++).set_double(args);
		else
			static_assert(always_false<Args>, "Unknown type");
	}(), ...);
}

template <int W> inline
void Machine<W>::ebreak()
{
	// its simpler and more flexible to just call a user-provided function
	this->system_call(riscv::SYSCALL_EBREAK);
}

template <int W> inline
void Machine<W>::copy_to_guest(address_t dst, const void* buf, size_t len)
{
	memory.memcpy_unsafe(dst, buf, len);
}

template <int W> inline
void Machine<W>::copy_from_guest(void* dst, address_t buf, size_t len)
{
	memory.memcpy_out(dst, buf, len);
}

template <int W> inline
address_type<W> Machine<W>::address_of(std::string_view name) const {
	return memory.resolve_address(name);
}

template <int W>
address_type<W> Machine<W>::stack_push(const void* data, size_t length)
{
	auto& sp = cpu.reg(REG_SP);
	sp = (sp - length) & ~(address_t) (W-1); // maintain word alignment
	this->copy_to_guest(sp, data, length);
	return sp;
}
template <int W> inline
address_type<W> Machine<W>::stack_push(const std::string& string)
{
	return stack_push(string.data(), string.size()+1); /* zero */
}
template <int W>
template <typename T> inline
address_type<W> Machine<W>::stack_push(const T& type)
{
	static_assert(std::is_standard_layout_v<T>, "Must be a POD type");
	return stack_push(&type, sizeof(T));
}

template <int W> inline
void Machine<W>::realign_stack()
{
	// the RISC-V calling convention mandates a 16-byte alignment
	cpu.reg(REG_SP) &= ~address_t{0xF};
}

template <int W> inline
const MultiThreading<W>& Machine<W>::threads() const
{
	if (UNLIKELY(m_mt == nullptr))
		throw MachineException(FEATURE_DISABLED, "Threads are not initialized");
	return *m_mt;
}
template <int W> inline
MultiThreading<W>& Machine<W>::threads()
{
	if (UNLIKELY(m_mt == nullptr))
		throw MachineException(FEATURE_DISABLED, "Threads are not initialized");
	return *m_mt;
}

template <int W> inline
const FileDescriptors& Machine<W>::fds() const
{
	if (m_fds != nullptr) return *m_fds;
	throw MachineException(ILLEGAL_OPERATION, "No access to files or sockets", 0);
}
template <int W> inline
FileDescriptors& Machine<W>::fds()
{
	if (m_fds != nullptr) return *m_fds;
	throw MachineException(ILLEGAL_OPERATION, "No access to files or sockets", 0);
}

template <int W> inline
Signals<W>& Machine<W>::signals() {
	if (m_signals == nullptr) m_signals.reset(new Signals<W>);
	return *m_signals;
}

#include "machine_vmcall.hpp"
