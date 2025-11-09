#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

union mbarrier_t {
  uint64_t u64;
  struct {
    uint64_t reserved : 1;
    uint64_t expected_arrive_count : 20;
    uint64_t transaction_count : 21;
    uint64_t lock : 1;
    uint64_t arrive_count : 20;
    uint64_t phase : 1;
  };
};

__device__ __forceinline__ int get_smem_ptr(void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(void* ptr, int n) {
  int smem_desc = get_smem_ptr(ptr);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_desc),
               "r"(n));
}

__device__ __forceinline__ uint64_t mbarrier_get_val(uint64_t* ptr) {
  int int_desc = get_smem_ptr(ptr);
  // disable barrier cache
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(int_desc));
  return ptr[0];
}

__device__ __forceinline__ void mbarrier_arrive(void* ptr, int n) {
  int int_desc = get_smem_ptr(ptr);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"r"(int_desc),
               "r"(n));
}

__device__ __forceinline__ void mbarrier_arrive1(void* ptr) {
  int smem_desc = get_smem_ptr(ptr);
  asm volatile(
      "{\n"
      ".reg.b64 state;\n"
      "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
      "}\n" ::"r"(smem_desc));
}

__device__ __forceinline__ void mbarrier_cluster_arrive(void* ptr, int n) {
  int smem_desc = get_smem_ptr(ptr);
  asm volatile(
      "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n" ::"r"(smem_desc),
      "r"(n));
}

__device__ __forceinline__ void mbarrier_cluster_arrive1(void* ptr) {
  int smem_desc = get_smem_ptr(ptr);
  asm volatile(
      "mbarrier.arrive.shared::cluster.b64 _, [%0];\n" ::"r"(smem_desc));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(void* ptr,
                                                              int tx) {
  int int_desc = get_smem_ptr(ptr);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                   int_desc),
               "r"(tx));
}

__device__ __forceinline__ void mbarrier_complete_tx(void* ptr, int n) {
  int int_desc = get_smem_ptr(ptr);
  asm volatile(
      "mbarrier.complete_tx.shared::cta.b64 [%0], %1;\n" ::"r"(int_desc),
      "r"(n));
}

__device__ __forceinline__ void mbarrier_expect_tx(void* ptr, int n) {
  int int_desc = get_smem_ptr(ptr);
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(int_desc),
               "r"(n));
}

__device__ __forceinline__ int mbarrier_test_wait(void* ptr, int phase) {
  int int_desc = get_smem_ptr(ptr);
  int ok;
  asm volatile(
      "{\n"
      ".reg .pred complete;\n"
      "mbarrier.test_wait.parity.shared::cta.b64 complete, [%1], %2;\n"
      "selp.u32 %0, 1, 0, complete;\n"
      "}\n"
      : "=r"(ok)
      : "r"(int_desc), "r"(phase));

  return ok;
}

__device__ __forceinline__ void mbarrier_try_wait(void* ptr, int phase) {
  int int_desc = get_smem_ptr(ptr);
  asm volatile(
      "{\n"
      ".reg .pred complete;\n"
      "WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 complete, [%0], %1;\n"
      "@complete bra DONE;\n"
      "bra WAIT;\n"
      "DONE:\n"
      "}\n"
      :
      : "r"(int_desc), "r"(phase));
}

__device__ __forceinline__ void mbarrier_inval(void* ptr) {
  int int_desc = get_smem_ptr(ptr);
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(int_desc));
}

__device__ __forceinline__ void mbarrier_arrive_drop(void* ptr, int n) {
  int int_desc = get_smem_ptr(ptr);
  asm volatile(
      "mbarrier.arrive_drop.shared::cta.b64 _, [%0], %1;\n" ::"r"(int_desc),
      "r"(n));
}

template <int kBits, bool kIsSigned, typename T>
__device__ __forceinline__ void print_field(T x) {
  for (int i = 0; i < kBits; ++i) {
    bool bit = (x >> (kBits - 1 - i)) & 0x1;
    if (bit) {
      printf("1");
    } else {
      printf("0");
    }
  }

  int v = x;
  if (kIsSigned) {
    int shift = 32 - kBits;
    v = (v << shift) >> shift;  // keep sign bit
  }
  if (kBits > 10) {
    printf("(%5d) ", v);
  } else {
    printf("(%d) ", v);
  }
}

__device__ __forceinline__ void print(uint64_t* ptr) {
  auto x = mbarrier_get_val(ptr);

  mbarrier_t b;
  b.u64 = x;

  printf("|");

  print_field<1, false>(b.phase);
  print_field<20, true>(b.arrive_count);
  print_field<1, false>(b.lock);
  print_field<21, true>(b.transaction_count);
  print_field<20, true>(b.expected_arrive_count);
  print_field<1, false>(b.reserved);

  bool test_phase_0 = mbarrier_test_wait(ptr, 0);
  bool test_phase_1 = mbarrier_test_wait(ptr, 1);

  printf("| test(0) = %d, test(1) = %d | ", test_phase_0, test_phase_1);
}

__global__ void mbarrier_test_case1() {
  __shared__ uint64_t bar;

  printf(
      "| p        Arrive Count         L     Transaction Count       Expected "
      "Arrive Count    X   |   TEST(0)    TEST(1)     |\n");

  int n = 7;

  mbarrier_init(&bar, n);
  print(&bar);
  printf("init(count=%d)\n", n);

  int a = 1;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  a = 6;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  a = 3;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  a = 2;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  a = 2;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  a = 2;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  a = 5;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);
}

__global__ void mbarrier_test_case2() {
  __shared__ uint64_t bar;

  printf(
      "| p        Arrive Count         L     Transaction Count       Expected "
      "Arrive Count    X   |   TEST(0)    TEST(1)     |\n");

  int n = 3;

  mbarrier_init(&bar, n);
  print(&bar);
  printf("init(count=%d)\n", n);

  int bytes = 1024;
  mbarrier_expect_tx(&bar, bytes);
  print(&bar);
  printf("expect_tx(bytes=%d)\n", bytes);

  int a = 1;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  bytes = 256;
  mbarrier_complete_tx(&bar, bytes);
  print(&bar);
  printf("async_complete(count=%d) *(simulate with synchronous)\n", bytes);

  a = 2;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  bytes = 512;
  mbarrier_complete_tx(&bar, bytes);
  print(&bar);
  printf("async_complete(count=%d) *(simulate with synchronous)\n", bytes);

  bytes = 256;
  mbarrier_complete_tx(&bar, bytes);
  print(&bar);
  printf("async_complete(count=%d) *(simulate with synchronous)\n", bytes);

  bytes = 512;
  mbarrier_arrive_and_expect_tx(&bar, bytes);
  print(&bar);
  printf("async_complete(count=%d) *(simulate with synchronous)\n", bytes);

  a = 2;
  mbarrier_arrive(&bar, a);
  print(&bar);
  printf("arrive(%d)\n", a);

  bytes = 512;
  mbarrier_complete_tx(&bar, bytes);
  print(&bar);
  printf("async_complete(count=%d) *(simulate with synchronous)\n", bytes);
}

int main() {
  {
    mbarrier_test_case1<<<1, 1>>>();
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  }

  {
    mbarrier_test_case2<<<1, 1>>>();
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  }
}
