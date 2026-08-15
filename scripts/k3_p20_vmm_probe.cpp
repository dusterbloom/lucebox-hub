#include <cuda.h>

#include <cstdint>
#include <cstdio>

int main() {
    CUresult status = cuInit(0);
    if (status != CUDA_SUCCESS) {
        std::printf("{\"supported\":false,\"cu_init\":%d}\n", status);
        return 1;
    }
    CUdevice device = 0;
    status = cuDeviceGet(&device, 0);
    if (status != CUDA_SUCCESS) {
        std::printf("{\"supported\":false,\"cu_device_get\":%d}\n", status);
        return 1;
    }
    int vmm = 0;
    int driver = 0;
    char name[256]{};
    cuDriverGetVersion(&driver);
    cuDeviceGetName(name, sizeof(name), device);
    status = cuDeviceGetAttribute(
        &vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        device);
    if (status != CUDA_SUCCESS || vmm == 0) {
        std::printf(
            "{\"supported\":false,\"device\":\"%s\","
            "\"driver_api\":%d,\"attribute_status\":%d}\n",
            name, driver, status);
        return 0;
    }
    CUmemAllocationProp property{};
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device;
    size_t minimum = 0;
    size_t recommended = 0;
    const CUresult minimum_status = cuMemGetAllocationGranularity(
        &minimum, &property, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    const CUresult recommended_status = cuMemGetAllocationGranularity(
        &recommended, &property, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    std::printf(
        "{\"supported\":%s,\"device\":\"%s\","
        "\"driver_api\":%d,\"minimum_granularity\":%llu,"
        "\"recommended_granularity\":%llu,"
        "\"minimum_status\":%d,\"recommended_status\":%d}\n",
        minimum_status == CUDA_SUCCESS ? "true" : "false", name, driver,
        static_cast<unsigned long long>(minimum),
        static_cast<unsigned long long>(recommended), minimum_status,
        recommended_status);
    return minimum_status == CUDA_SUCCESS ? 0 : 1;
}
