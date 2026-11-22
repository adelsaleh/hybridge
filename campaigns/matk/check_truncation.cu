// Direct native Truncate qualification; no solver and no pyamgx upload path.
// See AMGX/docs/matk_truncation_launch_fix_design.txt for acceptance criteria.
#include <amgx_c.h>
#include <dlfcn.h>
#include <link.h>
#include <stdexcept>
#include <matrix.h>
#include <truncate.h>
#include <truncate_max_elements_oracle.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace amgx
{

// A non-PIE function pointer can designate an executable PLT entry. Query the
// already loaded DSO explicitly so dladdr does not mistake that stub for the
// implementation. NOLOAD also prevents this diagnostic from loading a second
// AMGX library that the test is not otherwise using.
void report_library_provenance(void *import_address = NULL)
{
    struct LoadedLibrary
    {
        void *handle;
        LoadedLibrary() : handle(dlopen("libamgxsh.so", RTLD_NOLOAD | RTLD_NOW))
        {
            if (handle == NULL)
            {
                const char *error = dlerror();
                throw std::runtime_error(std::string("Cannot locate loaded libamgxsh.so: ") +
                                         (error == NULL ? "unknown loader error" : error));
            }
        }
        ~LoadedLibrary() { dlclose(handle); }
    } library;
    struct link_map *map = NULL;
    if (dlinfo(library.handle, RTLD_DI_LINKMAP, &map) != 0 ||
        map == NULL || map->l_name == NULL || map->l_name[0] == '\0')
    {
        throw std::runtime_error("Cannot identify the loaded AMGX library link map");
    }
    if (import_address == NULL)
    {
        // Every mode, including header-defined complex rejections, records the
        // selected library. This does not imply that complex truncation exists.
        std::cout << "AMGX library: " << map->l_name << std::endl;
        return;
    }
    Dl_info imported = {};
    if (dladdr(import_address, &imported) == 0 || imported.dli_sname == NULL)
    {
        throw std::runtime_error("Cannot identify imported Truncate symbol");
    }
    dlerror();
    void *implementation = dlsym(library.handle, imported.dli_sname);
    const char *symbol_error = dlerror();
    if (symbol_error != NULL || implementation == NULL)
    {
        throw std::runtime_error(std::string("Cannot resolve Truncate in loaded AMGX: ") +
                                 (symbol_error == NULL ? "null symbol" : symbol_error));
    }
    Dl_info resolved = {};
    if (dladdr(implementation, &resolved) == 0 || resolved.dli_fname == NULL ||
        reinterpret_cast<std::uintptr_t>(resolved.dli_fbase) !=
        static_cast<std::uintptr_t>(map->l_addr))
    {
        throw std::runtime_error("Truncate resolved outside the selected AMGX library");
    }
    std::cout << "Truncate implementation: " << resolved.dli_fname << std::endl;
}

template<class TConfig>
void rejection_contract()
{
    Matrix<TConfig> matrix(1, 1, 0, CSR);
    const bool host = TConfig::memSpace == AMGX_host;
    const bool complex = TConfig::vecPrec == AMGX_vecComplex || TConfig::vecPrec == AMGX_vecDoubleComplex;
    const int expected = host ? AMGX_ERR_NOT_IMPLEMENTED :
                         complex ? AMGX_ERR_NOT_SUPPORTED_TARGET : AMGX_ERR_BAD_PARAMETERS;
    try
    {
        Truncate<TConfig>::truncateByMaxElements(matrix, host || complex ? 32 : 33);
    }
    catch (amgx_exception &error)
    {
        if (error.reason() != expected) { throw; }
        std::cout << "Truncate rejection contract passed: error=" << expected << std::endl;
        return;
    }
    throw std::runtime_error("Expected typed rejection was not raised");
}

template<AMGX_Mode mode>
void run_real(const std::string &test)
{
    typedef typename TemplateMode<mode>::Type Config;
    report_library_provenance(reinterpret_cast<void *>(&Truncate<Config>::truncateByMaxElements));
    if (test == "provenance") { return; }
    if (test == "oracle")
    {
        TruncateMaxElementsOracle<Config> oracle;
        oracle.run();
    }
    else if (test == "boundaries")
    {
        TruncateMaxElementsOracle<Config> oracle;
        const int cap = oracle.integer_option("AMGX_TRUNCATION_TEST_CAP", 32);
        const int counts[] = {1, 31, 32, 33, 63, 64, 65, 127, 128, 129};
        for (int rows : counts)
        {
            oracle.check_fixture(rows, cap, "unique");
            oracle.check_fixture(rows, cap, "mixed");
        }
    }
    else { rejection_contract<Config>(); }
}

void dispatch(const std::string &mode, const std::string &test)
{
    if (test != "oracle" && test != "boundaries" && test != "rejections" && test != "provenance")
    { throw std::runtime_error("Test must be oracle, boundaries, rejections or provenance"); }
    if (mode == "dDDI") { run_real<AMGX_mode_dDDI>(test); }
    else if (mode == "dDFI") { run_real<AMGX_mode_dDFI>(test); }
    else if (mode == "dFFI") { run_real<AMGX_mode_dFFI>(test); }
    else if (test == "provenance" && (mode == "dZZI" || mode == "dZCI" || mode == "dCCI" ||
                                     mode == "hDDI" || mode == "hDFI" || mode == "hFFI")) { return; }
    else if (test != "rejections") { throw std::runtime_error("Oracle supports real device modes only"); }
    else if (mode == "dZZI") { rejection_contract<TemplateMode<AMGX_mode_dZZI>::Type>(); }
    else if (mode == "dZCI") { rejection_contract<TemplateMode<AMGX_mode_dZCI>::Type>(); }
    else if (mode == "dCCI") { rejection_contract<TemplateMode<AMGX_mode_dCCI>::Type>(); }
    else if (mode == "hDDI") { rejection_contract<TemplateMode<AMGX_mode_hDDI>::Type>(); }
    else if (mode == "hDFI") { rejection_contract<TemplateMode<AMGX_mode_hDFI>::Type>(); }
    else if (mode == "hFFI") { rejection_contract<TemplateMode<AMGX_mode_hFFI>::Type>(); }
    else { throw std::runtime_error("Unknown mode"); }
}
} // namespace amgx

int main(int argc, char **argv)
{
    if (argc > 3 || (argc > 1 && std::string(argv[1]) == "--help"))
    {
        std::cout << "Usage: check_truncation [dDDI|dDFI|dFFI|dZZI|dZCI|dCCI|hDDI|hDFI|hFFI] [oracle|boundaries|rejections|provenance]\n"
                  << "Env: AMGX_TRUNCATION_TEST_CAP=32 AMGX_TRUNCATION_TEST_ROWS=257 AMGX_TRUNCATION_TEST_CASE=all\n";
        return argc > 3 ? 2 : 0;
    }
    const std::string mode = argc > 1 ? argv[1] : "dDDI";
    const std::string test = argc > 2 ? argv[2] : "oracle";
    try
    {
        amgx::report_library_provenance();
        if (test == "provenance")
        {
            // Loader-only verification: no AMGX_initialize, matrices or CUDA
            // API calls, so the provenance repair can be checked on a CPU host.
            amgx::dispatch(mode, test);
            std::cout << "PASS provenance" << std::endl;
            return 0;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL provenance: " << error.what() << std::endl;
        return 2;
    }
    const AMGX_RC init_status = AMGX_initialize();
    if (init_status != AMGX_RC_OK) { std::cerr << "AMGX_initialize failed: " << init_status << std::endl; return 2; }
    int status = 0;
    try
    {
        std::cout << "Truncate qualification mode=" << mode << " test=" << test << std::endl;
        amgx::dispatch(mode, test);
        std::cout << "PASS" << std::endl;
    }
    catch (const amgx::amgx_exception &error)
    {
        std::cerr << "FAIL AMGX reason=" << error.reason() << ": " << error.what() << std::endl;
        status = 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << std::endl;
        status = 1;
    }
    const AMGX_RC final_status = AMGX_finalize();
    if (final_status != AMGX_RC_OK) { std::cerr << "AMGX_finalize failed: " << final_status << std::endl; status = 1; }
    return status;
}
