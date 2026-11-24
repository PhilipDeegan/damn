// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=1 profile=bootstrap
// a CUDA device, through the driver loaded at run time (no CUDA toolkit needed to build this; its
// own type names: a CUDA build (clang -x cuda) pre-includes cuda.h):
// passes with at least one device, prints each one's name and compute capability
#include <dlfcn.h>

#include <cstdio>

using DriverResult = int;
using DriverDevice = int;

int main()
{
    void* lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib)
    {
        std::printf("driver=none\n");
        return 1;
    }
    auto const sym = [&](char const* name) { return dlsym(lib, name); };
    auto init      = reinterpret_cast<DriverResult (*)(unsigned)>(sym("cuInit"));
    auto count     = reinterpret_cast<DriverResult (*)(int*)>(sym("cuDeviceGetCount"));
    auto get       = reinterpret_cast<DriverResult (*)(DriverDevice*, int)>(sym("cuDeviceGet"));
    auto name
        = reinterpret_cast<DriverResult (*)(char*, int, DriverDevice)>(sym("cuDeviceGetName"));
    auto attr
        = reinterpret_cast<DriverResult (*)(int*, int, DriverDevice)>(sym("cuDeviceGetAttribute"));
    int n     = 0;
    if (!init or !count or init(0) != 0 or count(&n) != 0)
    {
        std::printf("driver=unusable\n");
        return 1;
    }
    std::printf("devices=%d\n", n);
    for (int i = 0; i < n; ++i)
    {
        DriverDevice d = 0;
        char buf[256] = {};
        int major = 0, minor = 0;
        get(&d, i);
        name(buf, sizeof buf, d);
        attr(&major, 75, d); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        attr(&minor, 76, d); // ..._MINOR
        std::printf("device%d=%s sm_%d%d\n", i, buf, major, minor);
    }
    return n > 0 ? 0 : 1;
}
