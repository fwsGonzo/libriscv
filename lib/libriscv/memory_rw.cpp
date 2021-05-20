#include "machine.hpp"

namespace riscv
{
	template <int W>
	const Page& Memory<W>::get_readable_page(address_t address)
	{
		const auto pageno = page_number(address);
		auto& entry = m_rd_cache[pageno % m_rd_cache.size()];
		if (entry.pageno == pageno)
			return *entry.page;
		const auto& potential = get_pageno(pageno);
		if (UNLIKELY(!potential.attr.read)) {
			this->protection_fault(address);
		}
		entry = {pageno, &potential};
		return potential;
	}

	template <int W>
	Page& Memory<W>::get_writable_page(address_t address)
	{
		const auto pageno = page_number(address);
		auto& entry = m_wr_cache[pageno % m_wr_cache.size()];
		if (entry.pageno == pageno)
			return *entry.page;
		auto& potential = create_page(pageno);
		if (UNLIKELY(!potential.attr.write)) {
			this->protection_fault(address);
		}
		entry = {pageno, &potential};
		return potential;
	}

	template <int W>
	Page& Memory<W>::create_page(const address_t pageno)
	{
#ifdef RISCV_RODATA_SEGMENT_IS_SHARED
		if (m_rwpages.contains(pageno)) {
			return m_rwpages.pages[pageno - m_rwpages.begin];
		}
#endif
		auto it = m_pages.find(pageno);
		if (it != m_pages.end()) {
			Page& page = it->second;
			if (UNLIKELY(page.attr.is_cow)) {
				// don't enter page write handler with no-data page
				if (UNLIKELY(!page.has_data() || !page.attr.write))
					protection_fault(pageno * Page::size());
				m_page_write_handler(*this, page);
			}
			return page;
		}
#ifdef RISCV_RODATA_SEGMENT_IS_SHARED
		if (UNLIKELY(m_ropages.contains(pageno))) {
			this->protection_fault(pageno * Page::size());
		}
#endif
		// this callback must produce a new page, or throw
		return m_page_fault_handler(*this, pageno);
	}

	template <int W>
	const Page& Memory<W>::get_pageno_slowpath(address_t pageno) const noexcept
	{
#ifdef RISCV_SHARED_PAGETABLES
		// we can provide pages from even other machines
		// by mapping memory completely separately
		if (UNLIKELY(m_page_readf_handler != nullptr))
			return m_page_readf_handler(*this, pageno);
#endif
		(void) pageno;
		return Page::cow_page();
	}

	template <int W>
	void Memory<W>::free_pages(address_t dst, size_t len)
	{
		while (len > 0)
		{
			const size_t size = std::min(Page::size(), len);
			const address_t pageno = dst >> Page::SHIFT;
			auto& page = this->get_pageno(pageno);
			if (!page.is_cow_page()) {
				m_pages.erase(pageno);
			}
			dst += size;
			len -= size;
		}
	}

	template <int W>
	void Memory<W>::default_page_write(Memory<W>&, Page& page)
	{
		page.make_writable();
	}

	template <int W>
	const Page& Memory<W>::default_page_read(const Memory<W>&, size_t)
	{
		return Page::cow_page();
	}

	static const Page zeroed_page {
		PageAttributes {
			.read   = true,
			.write  = false,
			.exec   = false,
			.is_cow = true
		}
	};
	static const Page guarded_page {
		PageAttributes {
			.read   = false,
			.write  = false,
			.exec   = false,
			.is_cow = false,
			.non_owning = true
		}, nullptr
	};
	const Page& Page::cow_page() noexcept {
		return zeroed_page; // read-only, zeroed page
	}
	const Page& Page::guard_page() noexcept {
		return guarded_page; // inaccessible page
	}

	template <int W>
	Page& Memory<W>::install_shared_page(address_t pageno, const Page& shared_page)
	{
		auto& already_there = get_pageno(pageno);
		if (!already_there.is_cow_page() && !already_there.attr.non_owning)
			throw MachineException(ILLEGAL_OPERATION,
				"There was a page at the specified location already", pageno);
		if (shared_page.data() == nullptr && (
			shared_page.attr.write || shared_page.attr.read || shared_page.attr.exec))
			throw MachineException(ILLEGAL_OPERATION,
				"There was a RWX page with no allocated data", pageno);

		auto attr = shared_page.attr;
		attr.non_owning = true;
		// NOTE: If you insert a const Page, DON'T modify it! The machine
		// won't, unless system-calls do or manual intervention happens!
		auto res = m_pages.try_emplace(pageno, attr, const_cast<PageData*> (shared_page.m_page.get()));
		// try overwriting instead, if emplace failed
		if (res.second == false) {
			auto& page = res.first->second;
			new (&page) Page{attr, const_cast<PageData*> (shared_page.m_page.get())};
			return page;
		}
		return m_pages[pageno];
	}

	template <int W>
	void Memory<W>::insert_non_owned_memory(
		address_t dst, void* src, size_t size, PageAttributes attr)
	{
		assert(dst % Page::size() == 0);
		assert((dst + size) % Page::size() == 0);
		attr.non_owning = true;

		for (size_t i = 0; i < size; i += Page::size())
		{
			const auto pageno = (dst + i) >> Page::SHIFT;
			PageData* pdata = reinterpret_cast<PageData*> ((char*) src + i);
			m_pages.try_emplace(pageno, attr, pdata);
		}
	}

	template <int W> void
	Memory<W>::set_page_attr(address_t dst, size_t len, PageAttributes options)
	{
		const bool is_default = options.is_default();
		while (len > 0)
		{
			const size_t size = std::min(Page::size(), len);
			const address_t pageno = page_number(dst);
			// unfortunately, have to create pages for non-default attrs
			if (!is_default) {
				this->create_page(pageno).attr = options;
			} else {
				// set attr on non-COW pages only!
				const auto& page = this->get_pageno(pageno);
				if (page.attr.is_cow == false) {
					// this page has been written to, or had attrs set,
					// otherwise it would still be CoW.
					this->create_page(pageno).attr = options;
				}
			}

			dst += size;
			len -= size;
		}
	}

	template <int W>
	void Memory<W>::memcpy_unsafe(address_t dst, const void* vsrc, size_t len)
	{
		auto* src = (uint8_t*) vsrc;
		while (len != 0)
		{
			const size_t offset = dst & (Page::size()-1); // offset within page
			const size_t size = std::min(Page::size() - offset, len);
			auto& page = this->create_page(dst >> Page::SHIFT);
			if (UNLIKELY(!page.has_data()))
				protection_fault(dst);

			std::copy(src, src + size, page.data() + offset);

			dst += size;
			src += size;
			len -= size;
		}
	}

	template struct Memory<4>;
	template struct Memory<8>;
}
