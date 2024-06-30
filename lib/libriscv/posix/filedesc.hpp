#pragma once
#include <functional>
#include <string>
#include <map>
#include "../types.hpp"

#if defined(__APPLE__) || defined(__LINUX__)
#include <errno.h>
#endif

namespace riscv {

struct FileDescriptors
{
#ifdef _WIN32
    typedef uint64_t real_fd_type; // SOCKET is uint64_t
#else
    typedef int real_fd_type;
#endif

	// Insert and manage real FDs, return virtual FD
	int assign_file(real_fd_type fd) { return assign(fd, false); }
	int assign_pipe(real_fd_type fd);
	int assign_socket(real_fd_type fd) { return assign(fd, true); }
	int assign(real_fd_type fd, bool socket);
	// Get real FD from virtual FD
    real_fd_type get(int vfd);
    real_fd_type translate(int vfd);
	// Remove virtual FD and return real FD
    real_fd_type erase(int vfd);

	bool is_socket(int vfd) const { return (vfd & SOCKET_BIT) != 0; }
	bool is_pipe(int vfd) const { return (vfd & PIPE_BIT) != 0; }
	bool is_file(int vfd) const { return !is_socket(vfd) && !is_pipe(vfd); }
	bool permit_write(int vfd) {
		if (is_socket(vfd)) return true;
		else if (is_pipe(vfd)) return true;
		return permit_file_write;
	}

	~FileDescriptors();

    std::map<int, real_fd_type> translation;

	// Default working directory (fake root)
	std::string cwd = "/home";

	static constexpr int FILE_BASE = 0x1000;
	static constexpr int FILE_BIT   = 0x0;
	static constexpr int PIPE_BIT   = 0x10000000;
	static constexpr int SOCKET_BIT = 0x40000000;
	int file_counter   = FILE_BASE;
	int socket_counter = FILE_BASE;

	bool permit_filesystem = false;
	bool permit_file_write = false;
	bool permit_sockets = false;

	std::function<bool(void*, std::string&)> filter_open = nullptr; /* NOTE: Can modify path */
	std::function<bool(void*, std::string&)> filter_readlink = nullptr; /* NOTE: Can modify path */
	std::function<bool(void*, const std::string&)> filter_stat = nullptr;
	std::function<bool(void*, uint64_t)> filter_ioctl = nullptr;
};

inline int FileDescriptors::assign(FileDescriptors::real_fd_type real_fd, bool socket)
{
	int virtfd = file_counter++;
	if (socket) virtfd |= SOCKET_BIT;
	else virtfd |= FILE_BIT;

	translation.emplace(virtfd, real_fd);
	return virtfd;
}
inline int FileDescriptors::assign_pipe(FileDescriptors::real_fd_type real_fd)
{
	int virtfd = file_counter++;
	virtfd |= PIPE_BIT;

	translation.emplace(virtfd, real_fd);
	return virtfd;
}
inline FileDescriptors::real_fd_type FileDescriptors::get(int virtfd)
{
	auto it = translation.find(virtfd);
	if (it != translation.end()) return it->second;
	return -EBADF;
}
inline FileDescriptors::real_fd_type FileDescriptors::translate(int virtfd)
{
	auto it = translation.find(virtfd);
	if (it != translation.end()) return it->second;
	// Only allow direct access to standard pipes and errors
	return (virtfd <= 2) ? virtfd : -1;
}
inline FileDescriptors::real_fd_type FileDescriptors::erase(int virtfd)
{
	auto it = translation.find(virtfd);
	if (it != translation.end()) {
        FileDescriptors::real_fd_type real_fd = it->second;
		// Remove the virt FD
		translation.erase(it);
		return real_fd;
	}
	return -EBADF;
}

} // riscv
