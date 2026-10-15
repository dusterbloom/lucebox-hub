// pagebw: sequence-driven GTT page-quality probe for the loader's allocation order.
// ops: alloc:<GiB>:<tag>  free:<tag>  readfile:<path> (mmap, touch every 4 KiB, like the loader)
//      bw:<tag>[:<sub-GiB-window>]  (read bandwidth: best of 5 passes per 1 GiB window, prints each)
//      drop:<path> (posix_fadvise DONTNEED)  touch:<tag> (hipMemset the buffer)  sleep:<s>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <chrono>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e)); exit(1);} } while (0)
__global__ void rd(const uint4 * __restrict__ p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}
// mmvq-like: one warp per 2720-byte row (Q8_0 row of 2560), 8 rows per block, rows round-robin over blocks.
__global__ void rdrows(const unsigned char * __restrict__ p, size_t rows, size_t row_bytes, unsigned * out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned acc = 0;
    for (size_t r = blockIdx.x * 8 + warp; r < rows; r += (size_t) gridDim.x * 8) {
        const uint4 * row = (const uint4 *) (p + r * row_bytes);
        for (size_t i = lane; i < row_bytes / 16; i += 32) { uint4 v = row[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    }
    if (acc == 0x12345678u) out[0] = acc;
}
static double bwrows_gbs(const void * base, size_t bytes, unsigned * out) {
    const size_t row_bytes = 2720, rows = bytes / row_bytes;
    double best = 0;
    for (int it = 0; it < 5; ++it) {
        hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
        CK(hipEventRecord(a));
        rdrows<<<2048, 256>>>((const unsigned char *) base, rows, row_bytes, out);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b));
        double g = rows * row_bytes / 1e9 / (ms / 1e3); if (g > best) best = g;
        CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
    }
    return best;
}
static double bw_gbs(const void * base, size_t bytes, unsigned * out) {
    double best = 0;
    for (int it = 0; it < 5; ++it) {
        hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
        CK(hipEventRecord(a));
        rd<<<4096, 256>>>((const uint4 *) base, bytes / 16, out);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b));
        double g = bytes / 1e9 / (ms / 1e3); if (g > best) best = g;
        CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
    }
    return best;
}
int main(int argc, char ** argv) {
    std::map<std::string, std::pair<void *, size_t>> bufs;
    unsigned * out; CK(hipMalloc(&out, 64));
    const size_t GiB = 1ull << 30;
    for (int i = 1; i < argc; ++i) {
        std::string op = argv[i];
        std::vector<std::string> f; size_t s = 0;
        for (;;) { size_t c = op.find(':', s); f.push_back(op.substr(s, c == std::string::npos ? c : c - s)); if (c == std::string::npos) break; s = c + 1; }
        auto t0 = std::chrono::steady_clock::now();
        if (f[0] == "alloc") {
            size_t bytes = (size_t) (atof(f[1].c_str()) * GiB); void * p = nullptr;
            CK(hipMalloc(&p, bytes)); bufs[f[2]] = {p, bytes};
            printf("alloc %s %.2f GiB -> %p\n", f[2].c_str(), bytes / (double) GiB, p);
        } else if (f[0] == "touch") {
            auto & b = bufs[f[1]]; CK(hipMemset(b.first, 1, b.second)); CK(hipDeviceSynchronize());
        } else if (f[0] == "free") {
            auto & b = bufs[f[1]]; CK(hipFree(b.first)); bufs.erase(f[1]);
        } else if (f[0] == "readfile") {
            int fd = open(f[1].c_str(), O_RDONLY); if (fd < 0) { perror("open"); return 1; }
            struct stat st; fstat(fd, &st);
            const unsigned char * m = (const unsigned char *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) { perror("mmap"); return 1; }
            unsigned long long sum = 0;
            for (size_t o = 0; o < (size_t) st.st_size; o += 4096) sum += m[o];
            munmap((void *) m, st.st_size); close(fd);
            printf("readfile %s %.2f GiB sum %llu\n", f[1].c_str(), st.st_size / (double) GiB, sum);
        } else if (f[0] == "drop") {
            int fd = open(f[1].c_str(), O_RDONLY); if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
        } else if (f[0] == "sleep") {
            sleep(atoi(f[1].c_str()));
        } else if (f[0] == "bwr") {
            auto & b = bufs[f[1]]; size_t win = f.size() > 2 ? (size_t) (atof(f[2].c_str()) * GiB) : GiB;
            if (win > b.second) win = b.second;
            printf("bwr %s:", f[1].c_str());
            for (size_t o = 0; o + win <= b.second; o += win) printf(" %.0f", bwrows_gbs((char *) b.first + o, win, out));
            printf("  GB/s (rows)\n");
        } else if (f[0] == "bw") {
            auto & b = bufs[f[1]]; size_t win = f.size() > 2 ? (size_t) (atof(f[2].c_str()) * GiB) : GiB;
            if (win > b.second) win = b.second;
            printf("bw %s:", f[1].c_str());
            double whole = bw_gbs(b.first, b.second, out);
            for (size_t o = 0; o + win <= b.second; o += win) {
                double g = bw_gbs((char *) b.first + o, win, out);
                printf(" %.0f", g);
            }
            printf("  | whole %.1f GB/s\n", whole);
        }
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt > 0.5) printf("  (%s %.1fs)\n", f[0].c_str(), dt);
        fflush(stdout);
    }
    return 0;
}
