#pragma once
#include "elf.hpp"
#include "page.hpp"
#include <cassert>
#include <cstring>
#include <map>
#include "util/buffer.hpp" // <string>

namespace riscv
{
	template<int W> struct Machine;
	template<int W> struct DecoderCache;

	template<int W>
	struct Memory
	{
		using address_t = address_type<W>;
		using mmio_cb_t = Page::mmio_cb_t;
		using page_fault_cb_t = Function<Page&(Memory&, size_t)>;
		using page_readf_cb_t = Function<const Page&(const Memory&, size_t)>;
		using page_write_cb_t = Function<void(Memory&, Page&)>;

		template <typename T>
		T read(address_t src);

		template <typename T>
		T& writable_read(address_t src);

		template <typename T>
		void write(address_t dst, T value);

		void memset(address_t dst, uint8_t value, size_t len);
		void memcpy(address_t dst, const void* src, size_t);
		void memcpy(address_t dst, Machine<W>& srcm, address_t src, address_t len);
		void memcpy_out(void* dst, address_t src, size_t) const;
		void memcpy_unsafe(address_t dst, const void* src, size_t); // No page protections
		// Gives a chunk-wise view of the data at address, with a callback
		// invocation at each page boundary. @offs is the current byte offset.
		void foreach(address_t addr, size_t len,
			Function<void(Memory&, address_t offs, const uint8_t*, size_t)> callback);
		void foreach(address_t addr, size_t len,
			Function<void(const Memory&, address_t offs, const uint8_t*, size_t)>) const;
		// Gives a sequential view of the data at address, with the possibility
		// of optimizing away a copy if the data crosses no page-boundaries.
		void memview(address_t addr, size_t len,
			Function<void(Memory&, const uint8_t*, size_t)> callback);
		void memview(address_t addr, size_t len,
			Function<void(const Memory&, const uint8_t*, size_t)> callback) const;
		// Gives const-ref access to pod-type T viewed as sequential memory. (See above)
		template <typename T>
		void memview(address_t addr, Function<void(const T&)> callback) const;
		// Compare bounded memory
		int memcmp(address_t p1, address_t p2, size_t len) const;
		int memcmp(const void* p1, address_t p2, size_t len) const;
		// Gather fragmented virtual memory into an array of buffers
		riscv::Buffer rvbuffer(address_t addr, size_t len, size_t maxlen = 1 << 24) const;
		// Read a zero-terminated string directly from guests memory
		std::string memstring(address_t addr, size_t maxlen = 1024) const;
		size_t strlen(address_t addr, size_t maxlen = 4096) const;

		address_t start_address() const noexcept { return this->m_start_address; }
		address_t stack_initial() const noexcept { return this->m_stack_address; }
		void set_stack_initial(address_t addr) { this->m_stack_address = addr; }

		auto& machine() { return this->m_machine; }
		const auto& machine() const { return this->m_machine; }

		// Call interface
		address_t resolve_address(const std::string& sym) const;
		address_t resolve_section(const char* name) const;
		address_t exit_address() const noexcept;
		void      set_exit_address(address_t new_exit);
		// Basic backtraces and symbol lookups
		struct Callsite {
			std::string name = "(null)";
			address_t   address = 0x0;
			uint32_t    offset  = 0x0;
			size_t      size    = 0;
		};
		Callsite lookup(address_t) const;
		void print_backtrace(void(*print_function)(std::string_view));

		// Helpers for memory usage
		size_t pages_active() const noexcept { return m_pages.size(); }
		size_t owned_pages_active() const noexcept;
		// Page handling
		const auto& pages() const noexcept { return m_pages; }
		auto& pages() noexcept { return m_pages; }
		const Page& get_page(address_t) const noexcept;
		const Page& get_exec_pageno(address_t npage) const; // throws
		const Page& get_pageno(address_t npage) const noexcept;
		Page& create_page(address_t npage);
		void  set_page_attr(address_t, size_t len, PageAttributes);
		std::string get_page_info(address_t addr) const;
		// Page creation & destruction
		template <typename... Args>
		Page& allocate_page(size_t page, Args&& ...);
		void  invalidate_page(address_t pageno, Page&);
		void  free_pages(address_t, size_t len);
		// Page fault when writing to unused memory
		void set_page_fault_handler(page_fault_cb_t h) { this->m_page_fault_handler = h; }
#ifdef RISCV_SHARED_PAGETABLES
		// Page fault when reading unused memory. Primarily used for
		// pagetable sharing across machines, enabled with RISCV_SHARED_PT.
		void set_page_readf_handler(page_readf_cb_t h) { this->m_page_readf_handler = h; }
		void reset_page_readf_handler() { this->m_page_readf_handler = default_page_read; }
#endif
		// Page write on copy-on-write page
		void set_page_write_handler(page_write_cb_t h) { this->m_page_write_handler = h; }
		static void default_page_write(Memory&, Page& page);
		static const Page& default_page_read(const Memory&, size_t);
		// NOTE: use print_and_pause() to immediately break!
		void trap(address_t page_addr, mmio_cb_t callback);
		// Shared pages (regular pages will have priority!)
		Page&  install_shared_page(address_t pageno, const Page&);
		// Create pages for non-owned (shared) memory with given attributes
		void insert_non_owned_memory(
			address_t dst, void* src, size_t size, PageAttributes = {});
		// Advanced functions that faults in and gives you the page for an address
		const Page& get_readable_page(address_t);
		Page& get_writable_page(address_t);

		// Returns true if the address is inside the executable code segment
		bool is_executable(address_t addr);

#ifdef RISCV_INSTR_CACHE
		void generate_decoder_cache(const MachineOptions<W>&, address_t pbase, address_t va, size_t len);
		auto* get_decoder_cache() const { return m_exec_decoder; }
#endif

		const auto& binary() const noexcept { return m_binary; }
		void reset();

		bool is_binary_translated() const { return m_bintr_dl != nullptr; }
		void set_binary_translated(void* dl) const { m_bintr_dl = dl; }

		// serializes all the machine state + a tiny header to @vec
		void serialize_to(std::vector<uint8_t>& vec);
		// returns the machine to a previously stored state
		void deserialize_from(const std::vector<uint8_t>&, const SerializedMachine<W>&);

		Memory(Machine<W>&, std::string_view, MachineOptions<W>);
		Memory(Machine<W>&, const Machine<W>&, MachineOptions<W>);
		~Memory();
	private:
		struct MemoryArea {
			address_t begin = 0;
			address_t end = 0;
			std::unique_ptr<Page[]> pages = nullptr;
			std::unique_ptr<uint8_t[]> data = nullptr;
			bool contains(address_t pg) const noexcept { return pg >= begin && pg < end; }
		};
		inline auto& create_attr(const address_t address);
		static inline address_t page_number(const address_t address) {
			return address >> Page::SHIFT;
		}
		void clear_all_pages();
		void initial_paging();
		[[noreturn]] static void protection_fault(address_t);
		const Page& get_pageno_slowpath(address_t) const noexcept;
		// Helpers
		template <typename T>
		static void foreach_helper(T& mem, address_t addr, size_t len,
			Function<void(T&, address_t, const uint8_t*, size_t)> callback);
		template <typename T>
		static void memview_helper(T& mem, address_t addr, size_t len,
			Function<void(T&, const uint8_t*, size_t)> callback);
		// ELF stuff
		using Ehdr = typename Elf<W>::Ehdr;
		using Phdr = typename Elf<W>::Phdr;
		using Shdr = typename Elf<W>::Shdr;
		template <typename T> T* elf_offset(intptr_t ofs) const {
			return (T*) &m_binary.at(ofs);
		}
		inline const auto* elf_header() const noexcept {
			return elf_offset<const Ehdr> (0);
		}
		const Shdr* section_by_name(const char* name) const;
		void relocate_section(const char* section_name, const char* symtab);
		const typename Elf<W>::Sym* resolve_symbol(const char* name) const;
		const auto* elf_sym_index(const Shdr* shdr, uint32_t symidx) const {
			assert(symidx < shdr->sh_size / sizeof(typename Elf<W>::Sym));
			auto* symtab = elf_offset<typename Elf<W>::Sym>(shdr->sh_offset);
			return &symtab[symidx];
		}
		// ELF loader
		void binary_loader(const MachineOptions<W>&);
		void binary_load_ph(const MachineOptions<W>&, const Phdr*);
		void serialize_pages(MemoryArea&, address_t, const char*, size_t, size_t, PageAttributes);
		// Machine copy-on-write fork
		void machine_loader(const Machine<W>&, const MachineOptions<W>&);

		Machine<W>& m_machine;

		std::array<CachedPage<W, const Page>, RISCV_PAGE_CACHE> m_rd_cache;
		std::array<CachedPage<W, Page>, RISCV_PAGE_CACHE> m_wr_cache;

		std::map<address_t, Page> m_pages;

		page_fault_cb_t m_page_fault_handler = nullptr;
		page_write_cb_t m_page_write_handler = default_page_write;
#ifdef RISCV_SHARED_PAGETABLES
		page_readf_cb_t m_page_readf_handler = nullptr;
#endif

#ifdef RISCV_RODATA_SEGMENT_IS_SHARED
		MemoryArea m_ropages;
		MemoryArea m_rwpages;
#endif

		address_t m_start_address = 0;
		address_t m_stack_address = 0;
		address_t m_exit_address  = 0;
		const bool m_original_machine;

		const std::string_view m_binary;

		// ELF programs linear .text segment
		std::unique_ptr<uint8_t[]> m_exec_pagedata = nullptr;
		size_t    m_exec_pagedata_size = 0;
		address_t m_exec_pagedata_base = 0;
#ifdef RISCV_INSTR_CACHE
	#ifdef RISCV_DEBUG
		using dchandler_t = Instruction<W>;
	#else
		using dchandler_t = instruction_handler<W>;
	#endif
		dchandler_t* m_exec_decoder = nullptr;
		DecoderCache<W>* m_decoder_cache = nullptr;
#endif
		mutable void* m_bintr_dl = nullptr;
	};
#include "memory_inline.hpp"
#include "memory_helpers.hpp"
}
