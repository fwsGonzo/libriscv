#include "crc32.hpp"
#if defined(__x86_64__) && !defined(_MSC_VER)
#define CRC32C_INTRINSICS_ENABLED
#endif

inline bool ____is__aligned(const uint8_t* buffer, const int align) noexcept {
	return (((uintptr_t) buffer) & (align-1)) == 0;
}

#ifdef CRC32C_INTRINSICS_ENABLED
#include <immintrin.h>

__attribute__ ((target ("sse4.2")))
uint32_t crc32c_sse42(uint32_t hash, const uint8_t* buffer, size_t len)
{
	// 8-bits until 4-byte aligned
	while (!____is__aligned(buffer, 4) && len > 0) {
		hash = _mm_crc32_u8(hash, *buffer); buffer++; len--;
	}
	// 16 bytes at a time
	while (len >= 16) {
		hash = _mm_crc32_u32(hash, *(uint32_t*) (buffer +  0));
		hash = _mm_crc32_u32(hash, *(uint32_t*) (buffer +  4));
		hash = _mm_crc32_u32(hash, *(uint32_t*) (buffer +  8));
		hash = _mm_crc32_u32(hash, *(uint32_t*) (buffer + 12));
		buffer += 16; len -= 16;
	}
	// 4 bytes at a time
	while (len >= 4) {
		hash = _mm_crc32_u32(hash, *(uint32_t*) buffer);
		buffer += 4; len -= 4;
	}
	// remaining bytes
	if (len & 2) {
		hash = _mm_crc32_u16(hash, *(uint16_t*) buffer);
		buffer += 2;
	}
	if (len & 1) {
		hash = _mm_crc32_u8(hash, *buffer);
	}
	return hash;
}
#endif

namespace riscv
{
	uint32_t crc32c(uint32_t crc, const void* data, size_t len)
	{
	#ifdef CRC32C_INTRINSICS_ENABLED
		if (__builtin_cpu_supports ("sse4.2"))
		{
			return crc32c_sse42(crc, (const uint8_t*)data, len);
		}
	#endif
		return crc32<0x1EDC6F41>(crc, data, len);
	}

	uint32_t crc32c(const void* data, size_t len)
	{
	#ifdef CRC32C_INTRINSICS_ENABLED
		if (__builtin_cpu_supports ("sse4.2"))
		{
			return ~crc32c_sse42(0xFFFFFFFF, (const uint8_t *)data, len);
		}
	#endif
		return ~crc32<0x1EDC6F41>(0xFFFFFFFF, data, len);
	}
}
