#pragma once
#include <cstdint>

namespace riscv {
	template <int W>
	struct CallbackTable {
		uint8_t  (*mem_read8)(CPU<W>&, address_type<W> addr);
		uint16_t (*mem_read16)(CPU<W>&, address_type<W> addr);
		uint32_t (*mem_read32)(CPU<W>&, address_type<W> addr);
		uint64_t (*mem_read64)(CPU<W>&, address_type<W> addr);
		void (*mem_write8) (CPU<W>&, address_type<W> addr, uint8_t);
		void (*mem_write16)(CPU<W>&, address_type<W> addr, uint16_t);
		void (*mem_write32)(CPU<W>&, address_type<W> addr, uint32_t);
		void (*mem_write64)(CPU<W>&, address_type<W> addr, uint64_t);
		void* (*mem_getrptr)(CPU<W>&, address_type<W> addr);
		void* (*mem_getwptr)(CPU<W>&, address_type<W> addr);
		void (*jump)(CPU<W>&, address_type<W>, uint64_t);
		int  (*syscall)(CPU<W>&, address_type<W>, uint64_t);
		void (*stop)(CPU<W>&, uint64_t);
		void (*ebreak)(CPU<W>&, uint64_t);
		void (*system)(CPU<W>&, uint32_t);
		void (*trigger_exception)(CPU<W>&, int);
		float  (*sqrtf32)(float);
		double (*sqrtf64)(double);
	};
}
