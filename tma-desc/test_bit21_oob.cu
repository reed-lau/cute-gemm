// test_bit21_oob.cu
//
// 验证 TMA descriptor control word bit[21] 对 OOB 保护的影响。
//
// bit21=0: TMA 前端检查 coord < globalDim, OOB 时本地 fill, 不发内存请求
// bit21=1: TMA 跳过检查, 直接计算地址并访存, 地址无效时 page fault
//
// 编译:
//   nvcc -gencode arch=compute_100a,code=sm_100a -o test_bit21_oob test_bit21_oob.cu -lcuda
//   nvcc -arch=sm_90a -o test_bit21_oob test_bit21_oob.cu -lcuda
//
// 运行 (A/B 需在独立进程中执行, 因为 A 会导致 GPU context 不可恢复):
//   ./test_bit21_oob A   # bit21=1 → illegal memory access
//   ./test_bit21_oob B   # bit21=0 → OK (OOB fill)

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ─── device helpers ─────────────────────────────────────────────────────────

__device__ __forceinline__ uint32_t get_smem_ptr(const void* ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(ptr));
  return addr;
}

// ─── kernel: tensormap.replace + cp_fenceproxy (warp-level) ─────────────────

__global__ void update_descriptor(const __grid_constant__ CUtensorMap tmap_in,
                                  CUtensorMap* gmem_tmap, int new_dim1,
                                  const void* new_addr) {
  __shared__ __align__(128) CUtensorMap smem_tmap;

  // lane 0: copy + replace address + replace dim[1]
  if (threadIdx.x % 32 == 0) {
    smem_tmap = tmap_in;
    uint32_t si = get_smem_ptr(&smem_tmap);
    uint64_t const si64 = 0;
    asm volatile("cvt.u64.u32 %0, %1;" ::"l"(si64), "r"(si));

    uint64_t addr = reinterpret_cast<uint64_t>(new_addr);
    asm volatile(
        "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
        ::"l"(si64), "l"(addr));
    asm volatile(
        "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;"
        ::"l"(si64), "r"(new_dim1));
  }
  __syncwarp();

  // full warp: cp_fenceproxy (.sync.aligned requires warp convergence)
  uint32_t si = get_smem_ptr(&smem_tmap);
  uint64_t gi = reinterpret_cast<uint64_t>(gmem_tmap);
  asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic"
      ".release.gpu.sync.aligned [%0], [%1], 128;"
      ::"l"(gi), "r"(si));
}

// ─── kernel: fence.acquire + TMA load ───────────────────────────────────────

__global__ void tma_load(CUtensorMap* gmem_tmap, int coord_y) {
  __shared__ __align__(128) uint8_t data[128];
  __shared__ __align__(8) uint64_t mbar;

  if (threadIdx.x != 0) return;

  uint64_t gi = reinterpret_cast<uint64_t>(gmem_tmap);
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
               ::"l"(gi) : "memory");

  uint32_t mp = get_smem_ptr(&mbar);
  uint32_t dp = get_smem_ptr(data);

  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mp));

  uint32_t bytes = 16 * 8;  // box = [16, 8], fp8
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               ::"r"(mp), "r"(bytes));

  int cx = 0;
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dp), "l"(gi), "r"(cx), "r"(coord_y), "r"(mp));

  asm volatile(
      "{.reg .pred p;\n"
      " WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "        @!p bra WAIT;}\n"
      ::"r"(mp));

  printf("  coord_y=%d → data = %02x %02x %02x %02x\n",
         coord_y, data[0], data[1], data[2], data[3]);
}

// ─── host helpers ───────────────────────────────────────────────────────────

static int get_bit21(const CUtensorMap* t) {
  uint32_t c;
  memcpy(&c, (const uint8_t*)t + 8, 4);
  return (c >> 21) & 1;
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s A|B\n", argv[0]);
    printf("  A: template [256,512] → bit21=1 → illegal memory access\n");
    printf("  B: template [256,511] → bit21=0 → safe OOB fill\n");
    return 1;
  }
  bool use_big_shape = (argv[1][0] == 'A' || argv[1][0] == 'a');

  cuInit(0);
  cudaSetDevice(0);

  // 精确映射: 1 page mapped, 前后 unmapped
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};

  size_t page_size = 0;
  cuMemGetAllocationGranularity(&page_size, &prop,
                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);

  CUmemGenericAllocationHandle handle;
  cuMemCreate(&handle, page_size, &prop, 0);

  CUdeviceptr va = 0;
  cuMemAddressReserve(&va, page_size * 3, page_size, 0, 0);

  CUdeviceptr mapped = va + page_size;
  cuMemMap(mapped, page_size, 0, handle, 0);

  CUmemAccessDesc acc = {{CU_MEM_LOCATION_TYPE_DEVICE, 0},
                         CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
  cuMemSetAccess(mapped, page_size, &acc, 1);

  // tensor: fp8, 100 rows × 256 cols, 放在 mapped 区域末尾
  int num_seq = 100, k = 256;
  size_t tensor_bytes = num_seq * k;
  CUdeviceptr tensor_base = mapped + page_size - tensor_bytes;
  cuMemsetD8(tensor_base, 0xAB, tensor_bytes);

  // 创建 TMA descriptor 模板
  CUtensorMap tmap;
  cuuint64_t strides[1] = {256};
  cuuint32_t box[2] = {16, 8}, es[2] = {1, 1};
  cuuint64_t dims[2] = {256, use_big_shape ? 512ULL : 511ULL};

  CUtensorMap* gmem_tmap;
  cudaMalloc(&gmem_tmap, 256);

  cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                         (void*)mapped, dims, strides, box, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_NONE,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  printf("template dims=[256, %llu], total=%llu bytes, bit21=%d\n",
         (unsigned long long)dims[1],
         (unsigned long long)(256 * dims[1]),
         get_bit21(&tmap));
  printf("replace: addr=%p, dim[1]=%d\n", (void*)tensor_base, num_seq);
  printf("OOB load: coord_y=128 (128 > %d)\n", num_seq);
  printf("OOB addr: %p → %s mapped boundary\n\n",
         (void*)(tensor_base + 128 * 256),
         (tensor_base + 128 * 256 > mapped + page_size) ? "beyond" : "within");

  // kernel 1: replace addr + dim, write to gmem
  update_descriptor<<<1, 32>>>(tmap, gmem_tmap, num_seq, (void*)tensor_base);
  cudaDeviceSynchronize();

  // kernel 2: TMA load at coord_y=128 (OOB)
  tma_load<<<1, 32>>>(gmem_tmap, 128);
  cudaError_t err = cudaDeviceSynchronize();

  printf("result: %s\n",
         err == cudaSuccess ? "OK (OOB fill)" : cudaGetErrorString(err));

  // cleanup
  cudaFree(gmem_tmap);
  cuMemUnmap(mapped, page_size);
  cuMemRelease(handle);
  cuMemAddressFree(va, page_size * 3);
  return err != cudaSuccess;
}
