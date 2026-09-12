#pragma once
#include <memory>
#include "types.hpp"
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_set>

namespace riscv
{
	template<int W> struct DecoderData;

	// A fully decoded execute segment
	template <int W>
	struct DecodedExecuteSegment
	{
		using address_t = address_type<W>;

		bool is_within(address_t addr, size_t len = 2) const noexcept {
			address_t addr_end;
#ifdef _MSC_VER
			addr_end = addr + len;
			return addr >= m_vaddr_begin && addr_end <= m_vaddr_end && (addr_end > addr);
#else
			if (!__builtin_add_overflow(addr, len, &addr_end))
				return addr >= m_vaddr_begin && addr_end <= m_vaddr_end;
#endif
			return false;
		}

		auto* exec_data(address_t pc = 0) const noexcept {
			return m_exec_pagedata.get() + 4 + (ptrdiff_t(pc) - ptrdiff_t(m_vaddr_begin));
		}

		address_t exec_begin() const noexcept { return m_vaddr_begin; }
		address_t exec_end() const noexcept { return m_vaddr_end; }

		auto* decoder_cache() noexcept { return m_exec_decoder; }
		auto* decoder_cache() const noexcept { return m_exec_decoder; }
		auto* decoder_cache_base() const noexcept { return m_decoder_cache.get(); }
		size_t decoder_cache_size() const noexcept { return m_decoder_cache_size; }

		auto* create_decoder_cache(DecoderData<W>* cache, size_t size) {
			m_decoder_cache.reset(cache);
			m_decoder_cache_size = size;
			return m_decoder_cache.get();
		}
		void set_decoder(DecoderData<W>* dec) { m_exec_decoder = dec; }

		size_t size_bytes() const noexcept {
			return sizeof(*this) + (m_vaddr_end - m_vaddr_begin) + m_decoder_cache_size * 4;
		}
		bool empty() const noexcept { return m_exec_pagedata == nullptr; }

		DecodedExecuteSegment() = default;
		DecodedExecuteSegment(address_t vaddr, size_t exlen);
		DecodedExecuteSegment(DecodedExecuteSegment&&);
		~DecodedExecuteSegment();

		size_t threaded_rewrite(size_t bytecode, address_t pc, rv32i_instruction& instr);

		uint32_t crc32c_hash() const noexcept { return m_crc32c_hash; }
		void set_crc32c_hash(uint32_t hash) { m_crc32c_hash = hash; }

#ifdef RISCV_BINARY_TRANSLATION
		bool is_binary_translated() const noexcept { return !m_translator_mappings.empty(); }
		bool is_libtcc() const noexcept { return m_is_libtcc; }
		void* binary_translation_so() const { return m_bintr_dl; }
		void set_binary_translated(void* dl, bool is_libtcc) const { m_bintr_dl = dl; m_is_libtcc = is_libtcc; }
		uint32_t translation_hash() const { return m_bintr_hash; }
		void set_translation_hash(uint32_t hash) { m_bintr_hash = hash; }
		auto& create_mappings(size_t mappings) { m_translator_mappings.resize(mappings); return m_translator_mappings; }
		void set_mapping(unsigned i, bintr_block_func<W> handler) { m_translator_mappings.at(i) = handler; }
		bintr_block_func<W> mapping_at(unsigned i) const { return m_translator_mappings.at(i); }
		bintr_block_func<W> unchecked_mapping_at(unsigned i) const { return m_translator_mappings[i]; }
		size_t translator_mappings() const noexcept { return m_translator_mappings.size(); }
		void set_record_slowpaths(bool do_record) { m_do_record_slowpaths = do_record; }
		bool is_recording_slowpaths() const noexcept { return m_do_record_slowpaths; }
#ifdef RISCV_DEBUG
		void insert_slowpath_address(address_t addr) { m_slowpath_addresses.insert(addr); }
		auto& slowpath_addresses() const noexcept { return m_slowpath_addresses; }
#endif
#else
		bool is_binary_translated() const noexcept { return false; }
		bool is_libtcc() const noexcept { return false; }
#endif

#ifdef RISCV_ASMJIT
		bool is_asmjit_translated() const noexcept { return !m_asmjit_mappings.empty(); }
		auto& create_asmjit_mappings(size_t n) { m_asmjit_mappings.resize(n); return m_asmjit_mappings; }
		void set_asmjit_mapping(unsigned i, aj_block_func<W> f) { m_asmjit_mappings.at(i) = f; }
		aj_block_func<W> unchecked_asmjit_mapping_at(unsigned i) const { return m_asmjit_mappings[i]; }
		size_t asmjit_mappings() const noexcept { return m_asmjit_mappings.size(); }
		void set_asmjit_code(std::shared_ptr<AjCode> code) { m_asmjit_code = std::move(code); }
#else
		bool is_asmjit_translated() const noexcept { return false; }
#endif

#if defined(RISCV_BINARY_TRANSLATION) || defined(RISCV_ASMJIT)
		// Binary translation or asmjit executable code that was produced in the
		// background cannot patch the live decoder cache. See livepatch.hpp.
		auto* patched_decoder_cache() noexcept { return m_patched_exec_decoder; }
		void set_patched_decoder_cache(std::unique_ptr<DecoderData<W>[]> cache, DecoderData<W>* dec)
			{ m_patched_decoder_cache = std::move(cache); m_patched_exec_decoder = dec; }

		void wait_for_compilation_complete() {
			// Fast path: avoid taking the lock when nothing is compiling. The
			// flag is atomic, so this is safe even while a background thread
			// is finishing up.
			if (!m_is_background_compiling)
				return;
			std::unique_lock<std::mutex> lock(m_background_compilation_mutex);
			m_background_compilation_cv.wait(lock, [this]{ return !m_is_background_compiling.load(); });
		}
		bool is_background_compiling() const noexcept { return m_is_background_compiling; }
		void set_background_compiling(bool is_bg) {
			std::lock_guard<std::mutex> lock(m_background_compilation_mutex);
			const bool was_compiling = m_is_background_compiling;
			m_is_background_compiling = is_bg;
			if (was_compiling && !is_bg) {
				m_background_compilation_cv.notify_all();
			}
		}
		// The decoder cache is only fully generated *after* binary translation
		// has been kicked off, which means a background compilation must not
		// touch the decoder cache until the owning thread says it is complete.
		bool is_decoder_cache_ready() const noexcept { return m_decoder_cache_ready; }
		void set_decoder_cache_generator(std::thread::id id) { m_decoder_cache_generator = id; }
		void set_decoder_cache_ready() {
			std::lock_guard<std::mutex> lock(m_background_compilation_mutex);
			m_decoder_cache_ready = true;
			m_background_compilation_cv.notify_all();
		}
		void wait_for_decoder_cache_ready() {
			if (m_decoder_cache_ready)
				return;
			// A user-provided background callback is free to invoke the compilation
			// step synchronously, in which case we *are* the thread generating the
			// decoder cache, and waiting for ourselves would deadlock.
			if (std::this_thread::get_id() == m_decoder_cache_generator.load())
				return;
			std::unique_lock<std::mutex> lock(m_background_compilation_mutex);
			m_background_compilation_cv.wait(lock, [this]{ return m_decoder_cache_ready.load(); });
		}
#endif // RISCV_BINARY_TRANSLATION || RISCV_ASMJIT

		bool is_execute_only() const noexcept { return m_is_execute_only; }
		void set_execute_only(bool is_xo) { m_is_execute_only = is_xo; }

		bool is_likely_jit() const noexcept { return m_is_likely_jit; }
		void set_likely_jit(bool is_jit) { m_is_likely_jit = is_jit; }

		bool is_stale() const noexcept { return m_is_stale; }
		// XO-segments *must not be rebuilt*, so they should never be stale.
		void set_stale(bool is_stale) { m_is_stale = is_stale && !m_is_execute_only; }

		// Part of the shared-segment key: unchecked code must not be handed to a sandboxed Machine.
		bool is_unchecked_memory() const noexcept { return m_unchecked_memory; }
		void set_unchecked_memory(bool unchecked) { m_unchecked_memory = unchecked; }

	private:
		address_t m_vaddr_begin = 0;
		address_t m_vaddr_end   = 0;
		DecoderData<W>* m_exec_decoder = nullptr;

		// The flat execute segment is used to execute
		// the CPU::simulate_precise function in order to
		// support debugging, as well as when producing
		// the decoder cache
		std::unique_ptr<uint8_t[]> m_exec_pagedata = nullptr;

		// Decoder cache is used to run bytecode simulation at a high speed
		size_t          m_decoder_cache_size = 0;
		std::unique_ptr<DecoderData<W>[]> m_decoder_cache = nullptr;

#ifdef RISCV_BINARY_TRANSLATION
		std::vector<bintr_block_func<W>> m_translator_mappings;
		mutable void* m_bintr_dl = nullptr;
#ifdef RISCV_DEBUG
		std::unordered_set<address_t> m_slowpath_addresses;
#endif
		uint32_t m_bintr_hash = 0x0; // CRC32-C of the execute segment + compiler options
#endif
#ifdef RISCV_ASMJIT
		std::vector<aj_block_func<W>> m_asmjit_mappings;
		// Releases the JitRuntime (and every function in it) when the last
		// execute segment referencing it goes away.
		std::shared_ptr<AjCode> m_asmjit_code;
#endif
		uint32_t m_crc32c_hash = 0x0; // CRC32-C of the execute segment
		bool m_is_execute_only = false;
		bool m_unchecked_memory = false;
#ifdef RISCV_BINARY_TRANSLATION
		bool m_do_record_slowpaths = false;
		mutable bool m_is_libtcc = false;
#endif
#if defined(RISCV_BINARY_TRANSLATION) || defined(RISCV_ASMJIT)
		std::unique_ptr<DecoderData<W>[]> m_patched_decoder_cache = nullptr;
		DecoderData<W>* m_patched_exec_decoder = nullptr;
		std::atomic<bool> m_is_background_compiling { false };
		std::atomic<bool> m_decoder_cache_ready { false };
		std::atomic<std::thread::id> m_decoder_cache_generator {};
		mutable std::mutex m_background_compilation_mutex;
		std::condition_variable m_background_compilation_cv;
#endif
		// High-memory execute segments are likely to be JIT'd, and needs to
		// be nuked when attempting to re-use the segment
		bool m_is_likely_jit = false;
		bool m_is_stale = false;
	};

	template <int W>
	inline DecodedExecuteSegment<W>::DecodedExecuteSegment(
		address_t exaddr, size_t exlen)
	{
		m_vaddr_begin = exaddr;
		m_vaddr_end   = exaddr + exlen;
		// Allocate with 4 zero bytes before and after for sandbox hardening
		// Don't allocate for empty segments (exlen == 0), so that empty() remains correct
		if (exlen > 0)
			m_exec_pagedata.reset(new uint8_t[exlen + 8]);
	}

	template <int W>
	inline DecodedExecuteSegment<W>::DecodedExecuteSegment(DecodedExecuteSegment&& other)
	{
		m_vaddr_begin = other.m_vaddr_begin;
		m_vaddr_end   = other.m_vaddr_end;
		m_exec_decoder = other.m_exec_decoder;
		other.m_exec_decoder = nullptr;

		m_exec_pagedata = std::move(other.m_exec_pagedata);

		m_decoder_cache_size = other.m_decoder_cache_size;
		m_decoder_cache = std::move(other.m_decoder_cache);

#ifdef RISCV_BINARY_TRANSLATION
		m_translator_mappings = std::move(other.m_translator_mappings);
		m_bintr_dl = other.m_bintr_dl;
		other.m_bintr_dl = nullptr;
		m_bintr_hash = other.m_bintr_hash;
		m_is_libtcc = other.m_is_libtcc;
#endif
#ifdef RISCV_ASMJIT
		m_asmjit_mappings = std::move(other.m_asmjit_mappings);
		m_asmjit_code     = std::move(other.m_asmjit_code);
#endif
#if defined(RISCV_BINARY_TRANSLATION) || defined(RISCV_ASMJIT)
		m_patched_decoder_cache = std::move(other.m_patched_decoder_cache);
		m_patched_exec_decoder = other.m_patched_exec_decoder;
#endif
	}

	template <int W>
	inline DecodedExecuteSegment<W>::~DecodedExecuteSegment()
	{
#if defined(RISCV_BINARY_TRANSLATION) || defined(RISCV_ASMJIT)
		wait_for_compilation_complete();
#endif
#ifdef RISCV_BINARY_TRANSLATION
		extern void  dylib_close(void* dylib, bool is_libtcc);
		if (m_bintr_dl)
			dylib_close(m_bintr_dl, m_is_libtcc);
		m_bintr_dl = nullptr;
#endif
	}

} // riscv
