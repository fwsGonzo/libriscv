#include <libriscv/machine.hpp>
#include <libriscv/debug.hpp>
#include <libriscv/rsp_server.hpp>
#include <inttypes.h>
#include <chrono>
#include "settings.hpp"
static inline std::vector<uint8_t> load_file(const std::string&);
// On 32-bit systems, max 2GB. NodeJS uses a lot of virtual allocations, so max 12GB on 64-bit.
#if defined(__LP64__) || defined(_WIN64)
static constexpr uint64_t MAX_MEMORY = 12000ULL << 20; // 12GB
#else
static constexpr uint64_t MAX_MEMORY = 2000ULL << 20;  // 2GB
#endif
static const std::string DYNAMIC_LINKER = "/usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1";

struct Arguments {
	bool verbose = false;
	bool accurate = false;
	bool debug = false;
	bool singlestep = false;
	bool gdb = false;
	bool silent = false;
	bool timing = false;
	bool trace = false;
	bool no_translate = false;
	bool mingw = false;
	bool from_start = false;
	bool sandbox = false;
	bool ignore_text = false;
	uint64_t fuel = UINT64_MAX;
	std::string output_file;
	std::string call_function;
};

#ifdef HAVE_GETOPT_LONG
#include <getopt.h>

static const struct option long_options[] = {
	{"help", no_argument, 0, 'h'},
	{"verbose", no_argument, 0, 'v'},
	{"accurate", no_argument, 0, 'a'},
	{"debug", no_argument, 0, 'd'},
	{"single-step", no_argument, 0, '1'},
	{"fuel", required_argument, 0, 'f'},
	{"gdb", no_argument, 0, 'g'},
	{"silent", no_argument, 0, 's'},
	{"timing", no_argument, 0, 't'},
	{"trace", no_argument, 0, 'T'},
	{"no-translate", no_argument, 0, 'n'},
	{"mingw", no_argument, 0, 'm'},
	{"output", required_argument, 0, 'o'},
	{"from-start", no_argument, 0, 'F'},
	{"sandbox", no_argument, 0, 'S'},
	{"ignore-text", no_argument, 0, 'I'},
	{"call", required_argument, 0, 'c'},
	{0, 0, 0, 0}
};

static void print_help(const char* name)
{
	printf("Usage: %s [options] <program> [args]\n", name);
	printf("Options:\n"
		"  -h, --help         Print this help message\n"
		"  -v, --verbose      Enable verbose loader output\n"
		"  -a, --accurate     Accurate instruction counting\n"
		"  -d, --debug        Enable CLI debugger\n"
		"  -1, --single-step  One instruction at a time, enabling exact exceptions\n"
		"  -f, --fuel amt     Set max instructions until program halts\n"
		"  -g, --gdb          Start GDB server on port 2159\n"
		"  -s, --silent       Suppress program completion information\n"
		"  -t, --timing       Enable timing information in binary translator\n"
		"  -T, --trace        Enable tracing in binary translator\n"
		"  -n, --no-translate Disable binary translation\n"
		"  -m, --mingw        Cross-compile for Windows (MinGW)\n"
		"  -o, --output file  Output embeddable binary translated code (C99)\n"
		"  -F, --from-start   Start debugger from the beginning (_start)\n"
		"  -S  --sandbox      Enable strict sandbox\n"
		"  -I, --ignore-text  Ignore .text section, and use segments only\n"
		"  -c, --call func    Call a function after loading the program\n"
		"\n"
	);
	printf("libriscv is compiled with:\n"
#ifdef RISCV_32I
		"-  32-bit RISC-V support (RV32GB)\n"
#endif
#ifdef RISCV_64I
		"-  64-bit RISC-V support (RV64GB)\n"
#endif
#ifdef RISCV_128I
		"-  128-bit RISC-V support (RV128G)\n"
#endif
#ifdef RISCV_EXT_A
		"-  A: Atomic extension is enabled\n"
#endif
#ifdef RISCV_EXT_C
		"-  C: Compressed extension is enabled\n"
#endif
#ifdef RISCV_EXT_V
		"-  V: Vector extension is enabled\n"
#endif
#if defined(RISCV_BINARY_TRANSLATION) && defined(RISCV_LIBTCC)
		"-  Binary translation is enabled (libtcc)\n"
#elif defined(RISCV_BINARY_TRANSLATION)
		"-  Binary translation is enabled\n"
#endif
#ifdef RISCV_DEBUG
		"-  Extra debugging features are enabled\n"
#endif
#ifdef RISCV_FLAT_RW_ARENA
		"-  Flat sequential memory arena is enabled\n"
#endif
#ifdef RISCV_ENCOMPASSING_ARENA
#define _STR(x) #x
#define STR(x) _STR(x)
		"-  Fixed N-bit address space is enabled (" STR(RISCV_ENCOMPASSING_ARENA_BITS) " bits)\n"
#endif
#ifdef RISCV_TIMED_VMCALLS
		"-  Timed VM calls are enabled\n"
#endif
		"\n"
	);
}

static int parse_arguments(int argc, const char** argv, Arguments& args)
{
	int c;
	while ((c = getopt_long(argc, (char**)argv, "hvad1f:gstTnmo:FSIc:", long_options, nullptr)) != -1)
	{
		switch (c)
		{
			case 'h': print_help(argv[0]); return 0;
			case 'v': args.verbose = true; break;
			case 'a': args.accurate = true; break;
			case 'd': args.debug = true; break;
			case '1': args.singlestep = true; break;
			case 'f': break;
			case 'g': args.gdb = true; break;
			case 's': args.silent = true; break;
			case 't': args.timing = true; break;
			case 'T': args.trace = true; break;
			case 'n': args.no_translate = true; break;
			case 'm': args.mingw = true; break;
			case 'o': break;
			case 'F': args.from_start = true; break;
			case 'S': args.sandbox = true; break;
			case 'I': args.ignore_text = true; break;
			case 'c': break;
			default:
				fprintf(stderr, "Unknown option: %c\n", c);
				return -1;
		}

		if (c == 'f') {
			char* endptr;
			args.fuel = strtoull(optarg, &endptr, 10);
			if (*endptr != '\0') {
				fprintf(stderr, "Invalid number: %s\n", optarg);
				return -1;
			}
			if (args.fuel == 0) {
				args.fuel = UINT64_MAX;
			}
			if (args.verbose) {
				printf("* Fuel set to %" PRIu64 "\n", args.fuel);
			}
		} else if (c == 'o') {
			args.output_file = optarg;
			if (args.verbose) {
				printf("* Output file prefix set to %s\n", args.output_file.c_str());
			}
		} else if (c == 'c') {
			args.call_function = optarg;
			if (args.verbose) {
				printf("* Function to VMCall: %s\n", args.call_function.c_str());
			}
		}
	}

	if (optind >= argc) {
		print_help(argv[0]);
		return -1;
	}

	return optind;
}

#endif

template <int W>
static void run_sighandler(riscv::Machine<W>&);

template <int W>
static void run_program(
	const Arguments& cli_args,
	const std::vector<uint8_t>& binary,
	const bool is_dynamic,
	const std::vector<std::string>& args)
{
	if (cli_args.mingw && (!riscv::binary_translation_enabled || riscv::libtcc_enabled)) {
		fprintf(stderr, "Error: Full binary translation must be enabled for MinGW cross-compilation\n");
		exit(1);
	}

	std::vector<riscv::MachineTranslationOptions> cc;
	if (cli_args.mingw) {
		cc.push_back(riscv::MachineTranslationCrossOptions{});
	}
	if (!cli_args.output_file.empty()) {
		cc.push_back(riscv::MachineTranslationEmbeddableCodeOptions{cli_args.output_file});
	}

	// Create a RISC-V machine with the binary as input program
	riscv::Machine<W> machine { binary, {
		.memory_max = MAX_MEMORY,
		.ignore_text_section = cli_args.ignore_text,
		.verbose_loader = cli_args.verbose,
		.use_shared_execute_segments = false, // We are only creating one machine, disabling this can enable some optimizations
#ifdef RISCV_BINARY_TRANSLATION
		.translate_enabled = !cli_args.no_translate,
		.translate_trace = cli_args.trace,
		.translate_timing = cli_args.timing,
		.translate_ignore_instruction_limit = !cli_args.accurate, // Press Ctrl+C to stop
#ifdef _WIN32
		.translation_prefix = "translations/rvbintr-",
		.translation_suffix = ".dll",
#else
		.cross_compile = cc,
#endif
#endif
	}};

	// A helper system call to ask for symbols that is possibly only known at runtime
	// Used by testing executables
	riscv::address_type<W> symbol_function = 0;
	machine.set_userdata(&symbol_function);
	machine.install_syscall_handler(500,
		[] (auto& machine) {
			auto [addr] = machine.template sysargs<riscv::address_type<W>>();
			auto& symfunc = *machine.template get_userdata<decltype(symbol_function)>();
			symfunc = addr;
			printf("Introduced to symbol function: 0x%" PRIX64 "\n", uint64_t(addr));
		});

	if constexpr (full_linux_guest)
	{
		std::vector<std::string> env = {
			"LC_CTYPE=C", "LC_ALL=C", "RUST_BACKTRACE=full"
		};
		machine.setup_linux(args, env);
		// Linux system to open files and access internet
		machine.setup_linux_syscalls();
		machine.fds().permit_filesystem = !cli_args.sandbox;
		machine.fds().permit_sockets    = !cli_args.sandbox;
		// Rewrite certain links to masquerade and simplify some interactions (eg. /proc/self/exe)
		machine.fds().filter_readlink = [&] (void* user, std::string& path) {
			if (path == "/proc/self/exe") {
				path = machine.fds().cwd + "/program";
				return true;
			}
			fprintf(stderr, "Guest wanted to readlink: %s (denied)\n", path.c_str());
			return false;
		};
		// Only allow opening certain file paths. The void* argument is
		// the user-provided pointer set in the RISC-V machine.
		machine.fds().filter_open = [=] (void* user, std::string& path) {
			(void) user;
			if (path == "/etc/hostname"
				|| path == "/etc/hosts"
				|| path == "/etc/nsswitch.conf"
				|| path == "/etc/host.conf"
				|| path == "/etc/resolv.conf")
				return true;
			if (path == "/dev/urandom")
				return true;
			if (path == "/program") { // Fake program path
				path = args.at(0); // Sneakily open the real program instead
				return true;
			}
			if (path == "/etc/ssl/certs/ca-certificates.crt")
				return true;
			// ld-linux
			if (path == "/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1") {
				path = DYNAMIC_LINKER;
				return true;
			}

			// Paths that are allowed to be opened
			static const std::string sandbox_libdir  = "/lib/riscv64-linux-gnu/";
			// The real path to the libraries (on the host system)
			static const std::string real_libdir = "/usr/riscv64-linux-gnu/lib/";
			// The dynamic linker and libraries we allow
			static const std::vector<std::string> libs = {
				"libdl.so.2", "libm.so.6", "libgcc_s.so.1", "libc.so.6",
				"libstdc++.so.6", "libresolv.so.2", "libnss_dns.so.2", "libnss_files.so.2"
			};

			if (path.find(sandbox_libdir) == 0) {
				// Find the library name
				auto lib = path.substr(sandbox_libdir.size());
				if (std::find(libs.begin(), libs.end(), lib) == libs.end()) {
					if (cli_args.verbose) {
						fprintf(stderr, "Guest wanted to open: %s (denied)\n", path.c_str());
					}
					return false;
				} else if (cli_args.verbose) {
					fprintf(stderr, "Guest wanted to open: %s (allowed)\n", path.c_str());
				}
				// Construct new path
				path = real_libdir + path.substr(sandbox_libdir.size());
				return true;
			}

			if (is_dynamic && args.size() > 1 && path == args.at(1)) {
				return true;
			}
			if (cli_args.verbose) {
				fprintf(stderr, "Guest wanted to open: %s (denied)\n", path.c_str());
			}
			return false;
		};
		// multi-threading
		machine.setup_posix_threads();
	}
	else if constexpr (newlib_mini_guest)
	{
		// the minimum number of syscalls needed for malloc and C++ exceptions
		machine.setup_newlib_syscalls();
		machine.setup_argv(args);
	}
	else if constexpr (micro_guest)
	{
		// This guest has accelerated libc functions, which
		// are provided as system calls
		// See: tests/unit/native.cpp and tests/unit/include/native_libc.h
		constexpr size_t heap_size = 6ULL << 20; // 6MB
		auto heap = machine.memory.mmap_allocate(heap_size);

		machine.setup_native_heap(470, heap, heap_size);
		machine.setup_native_memory(475);
		machine.setup_native_threads(490);

		machine.setup_newlib_syscalls();
		machine.setup_argv(args);
	}
	else {
		fprintf(stderr, "Unknown emulation mode! Exiting...\n");
		exit(1);
	}

	// A CLI debugger used with --debug or DEBUG=1
	riscv::DebugMachine debug { machine };

	if (cli_args.debug)
	{
		// Print all instructions by default
		const bool vi = true;
		// With --verbose we also print register values after
		// every instruction.
		const bool vr = cli_args.verbose;

		auto main_address = machine.address_of("main");
		if (cli_args.from_start || main_address == 0x0) {
			debug.verbose_instructions = vi;
			debug.verbose_registers = vr;
			// Without main() this is a custom or stripped program,
			// so we break immediately.
			debug.print_and_pause();
		} else {
			// Automatic breakpoint at main() to help debug certain programs
			debug.breakpoint(main_address,
			[vi, vr] (auto& debug) {
				auto& cpu = debug.machine.cpu;
				// Remove the breakpoint to speed up debugging
				debug.erase_breakpoint(cpu.pc());
				debug.verbose_instructions = vi;
				debug.verbose_registers = vr;
				printf("\n*\n* Entered main() @ 0x%" PRIX64 "\n*\n", uint64_t(cpu.pc()));
				debug.print_and_pause();
			});
		}
	}

	auto t0 = std::chrono::high_resolution_clock::now();
	try {
		// If you run the emulator with --gdb or GDB=1, you can connect
		// with gdb-multiarch using target remote localhost:2159.
		if (cli_args.gdb) {
			printf("GDB server is listening on localhost:2159\n");
			riscv::RSP<W> server { machine, 2159 };
			auto client = server.accept();
			if (client != nullptr) {
				printf("GDB is connected\n");
				while (client->process_one());
			}
			if (!machine.stopped()) {
				// Run remainder of program
				machine.simulate(cli_args.fuel);
			}
		} else if (cli_args.debug) {
			// CLI debug simulation
			debug.simulate();
		} else if (cli_args.singlestep) {
			// Single-step precise simulation
			machine.set_max_instructions(~0ULL);
			machine.cpu.simulate_precise();
		} else {
#ifdef RISCV_TIMED_VMCALLS
			// Simulation with experimental timeout
			machine.execute_with_timeout(30.0f, ~0ULL, 0U, machine.cpu.pc());
#else
			// Normal RISC-V simulation
			machine.simulate(cli_args.fuel);
#endif
		}
	} catch (riscv::MachineException& me) {
		printf("%s\n", machine.cpu.current_instruction_to_string().c_str());
		printf(">>> Machine exception %d: %s (data: 0x%" PRIX64 ")\n",
				me.type(), me.what(), me.data());
		printf("%s\n", machine.cpu.registers().to_string().c_str());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
		if (me.type() == riscv::UNIMPLEMENTED_INSTRUCTION || me.type() == riscv::MISALIGNED_INSTRUCTION) {
			printf(">>> Is an instruction extension disabled?\n");
			printf(">>> A-extension: %d  C-extension: %d  V-extension: %d\n",
				riscv::atomics_enabled, riscv::compressed_enabled, riscv::vector_extension);
		}
		if (cli_args.debug)
			debug.print_and_pause();
		else
			run_sighandler(machine);
	} catch (std::exception& e) {
		printf(">>> Exception: %s\n", e.what());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
		if (cli_args.debug)
			debug.print_and_pause();
		else
			run_sighandler(machine);
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> runtime = t1 - t0;

	if (!cli_args.silent) {
		const auto retval = machine.return_value();
		printf(">>> Program exited, exit code = %" PRId64 " (0x%" PRIX64 ")\n",
			int64_t(retval), uint64_t(retval));
		if (cli_args.accurate || !riscv::binary_translation_enabled)
		printf("Instructions executed: %" PRIu64 "  Runtime: %.3fms  Insn/s: %.0fmi/s\n",
			machine.instruction_counter(), runtime.count()*1000.0,
			machine.instruction_counter() / (runtime.count() * 1e6));
		else
		printf("Runtime: %.3fms   (Use --accurate for instruction counting)\n",
			runtime.count()*1000.0);
		printf("Pages in use: %zu (%" PRIu64 " kB virtual memory, total %" PRIu64 " kB)\n",
			machine.memory.pages_active(),
			machine.memory.pages_active() * riscv::Page::size() / uint64_t(1024),
			machine.memory.memory_usage_total() / uint64_t(1024));
	}

	if (!cli_args.call_function.empty())
	{
		auto addr = machine.address_of(cli_args.call_function);
		if (addr == 0 && symbol_function != 0) {
			addr = machine.vmcall(symbol_function, cli_args.call_function);
		}
		if (addr != 0) {
			printf("Calling function %s @ 0x%lX\n", cli_args.call_function.c_str(), long(addr));
#ifdef RISCV_TIMED_VMCALLS
			machine.timed_vmcall(addr, 30.0f);
#else
			machine.vmcall(addr);
#endif
		} else {
			printf("Error: Function %s not found, not able to call\n", cli_args.call_function.c_str());
		}
	}
}

int main(int argc, const char** argv)
{
	Arguments cli_args;
#ifdef HAVE_GETOPT_LONG
	const int optind = parse_arguments(argc, argv, cli_args);
	if (optind < 0)
		return 1;
	else if (optind == 0)
		return 0;
	// Skip over the parsed arguments
	argc -= optind;
	argv += optind;
#else
	if (argc < 2) {
		fprintf(stderr, "Provide RISC-V binary as argument!\n");
		exit(1);
	}
	// Skip over the program name
	argc -= 1;
	argv += 1;

	// Environment variables can be used to control the emulator
	cli_args.verbose = getenv("VERBOSE") != nullptr;
	cli_args.debug = getenv("DEBUG") != nullptr;
	cli_args.gdb = getenv("GDB") != nullptr;
	cli_args.silent = getenv("SILENT") != nullptr;
	cli_args.timing = getenv("TIMING") != nullptr;
	cli_args.trace = getenv("TRACE") != nullptr;
	cli_args.no_translate = getenv("NO_TRANSLATE") != nullptr;
	cli_args.mingw = getenv("MINGW") != nullptr;
	cli_args.from_start = getenv("FROM_START") != nullptr;

#endif

	std::vector<std::string> args;
	for (int i = 0; i < argc; i++) {
		args.push_back(argv[i]);
	}
	const std::string& filename = args.front();

	using ElfHeader = typename riscv::Elf<4>::Header;

	try {
		auto binary = load_file(filename);
		if (binary.size() < sizeof(ElfHeader)) {
			fprintf(stderr, "ELF binary was too small to be usable!\n");
			exit(1);
		}

		const bool is_dynamic = ((ElfHeader *)binary.data())->e_type == ElfHeader::ET_DYN;

		if (binary[4] == riscv::ELFCLASS64 && is_dynamic) {
			// Load the dynamic linker shared object
			binary = load_file(DYNAMIC_LINKER);
			// Insert program name as argv[1]
			args.insert(args.begin() + 1, args.at(0));
			// Set dynamic linker to argv[0]
			args.at(0) = DYNAMIC_LINKER;
		}

		if (binary[4] == riscv::ELFCLASS64)
#ifdef RISCV_64I
			run_program<riscv::RISCV64> (cli_args, binary, is_dynamic, args);
#else
			throw riscv::MachineException(riscv::FEATURE_DISABLED, "64-bit not currently enabled");
#endif
		else if (binary[4] == riscv::ELFCLASS32)
#ifdef RISCV_32I
			run_program<riscv::RISCV32> (cli_args, binary, is_dynamic, args);
#else
			throw riscv::MachineException(riscv::FEATURE_DISABLED, "32-bit not currently enabled");
#endif
		else
			throw riscv::MachineException(riscv::INVALID_PROGRAM, "Unknown ELF class", binary[4]);
	} catch (const std::exception& e) {
		printf("Exception: %s\n", e.what());
	}

	return 0;
}

template <int W>
void run_sighandler(riscv::Machine<W>& machine)
{
	constexpr int SIG_SEGV = 11;
	auto& action = machine.sigaction(SIG_SEGV);
	if (action.is_unset())
		return;

	auto handler = action.handler;
	action.handler = 0x0; // Avoid re-triggering(?)

	machine.stack_push(machine.cpu.reg(riscv::REG_RA));
	machine.cpu.reg(riscv::REG_RA) = machine.cpu.pc();
	machine.cpu.reg(riscv::REG_ARG0) = 11; /* SIGSEGV */
	try {
		machine.cpu.jump(handler);
		machine.simulate(60'000);
	} catch (...) {}

	action.handler = handler;
}

#include <stdexcept>
#include <unistd.h>
std::vector<uint8_t> load_file(const std::string& filename)
{
    size_t size = 0;
    FILE* f = fopen(filename.c_str(), "rb");
    if (f == NULL) throw std::runtime_error("Could not open file: " + filename);

    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> result(size);
    if (size != fread(result.data(), 1, size, f))
    {
        fclose(f);
        throw std::runtime_error("Error when reading from file: " + filename);
    }
    fclose(f);
    return result;
}
