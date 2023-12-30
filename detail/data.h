#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T>
void cpu_rand_data(T *c);

template <typename T>
void cpu_const_data(T *c, float k);

template <typename T>
void cpu_gemm(T *c, const T &a, const T &b);

template <typename T>
void cpu_compare(const T &x, const T &y, float threshold = 1.E-1);

template <typename T>
void gpu_compare(const T *x, const T *y, int n, float threshold = 1.E-1);

void printf_fail(const char *fmt, ...) {
  int red = 31;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

void printf_ok(const char *fmt, ...) {
  int red = 32;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

template <typename T>
void cpu_rand_data(T *c) {
  auto t = *c;

  using ValueType = typename T::value_type;

  int n = size(t);
  for (int i = 0; i < n; ++i) {
    float v = ((rand() % 200) - 100.f) * 0.01f;
    // printf("v = %f\n", v);
    t(i) = ValueType(v);
  }
}

template <typename T>
void cpu_const_data(T *c, float k) {
  auto t = *c;

  int n = size(t);
  for (int i = 0; i < n; ++i) {
    t(i) = k;
  }
}

template <typename T>
void cpu_gemm(T *c, const T &a, const T &b) {
  using namespace cute;

  using ValueType = typename T::value_type;

  int m = size<0>(a);
  int n = size<0>(b);
  int k = size<1>(a);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float s = 0.f;

      for (int kk = 0; kk < k; ++kk) {
        float v1 = a(i, kk);
        float v2 = b(j, kk);
        s += v1 * v2;
      }

      (*c)(i, j) = ValueType(s);
    }
  }
}

template <typename T>
void cpu_compare(const T &x, const T &y, float threshold) {
  using namespace cute;

  if (size(x) != size(y)) {
    fprintf(stderr, "lenght not equal x = %d, y = %d\n", size(x), size(y));
    exit(9);
  }

  int n = size(x);
  float diff_max = 0;
  int diff_count = 0;
  for (int i = 0; i < n; ++i) {
    float v0 = x(i);
    float v1 = y(i);

    diff_max = max(diff_max, fabs(v0 - v1));

    if (fabs(v0 - v1) > threshold) {
      ++diff_count;
    }
  }
  if (diff_count > 0) {
    printf("check fail: max_diff = %f, diff_count = %d\n", diff_max,
           diff_count);
  } else {
    printf("cpu check ok\n");
  }
}

template <typename T>
__global__ void gpu_compare_kernel(const T *x, const T *y, int n,
                                   float threshold, int *count,
                                   float *max_error) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  float v0 = x[idx];
  float v1 = y[idx];

  float diff = fabs(v0 - v1);
  if (diff > threshold) {
    atomicAdd(count, 1);

    // for positive floating point, there int representation is in the same
    // order.
    int int_diff = *((int *)(&diff));
    atomicMax((int *)max_error, int_diff);
  }
}

template <typename T>
void gpu_compare(const T *x, const T *y, int n, float threshold) {
  int *num_count;
  float *max_error;
  cudaMalloc(&num_count, sizeof(int));
  cudaMalloc(&max_error, sizeof(float));
  cudaMemset(num_count, 0, sizeof(int));
  cudaMemset(max_error, 0, sizeof(float));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  gpu_compare_kernel<<<grid, block>>>(x, y, n, threshold, num_count, max_error);
  int num = 0;
  float error = 0;
  cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (num == 0) {
    printf_ok("check ok, max_error = %f\n", error);
  } else {
    float p = (100.f * num) / n;
    printf_fail("===============================\n");
    printf_fail("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, n,
                error);
    printf_fail("===============================\n");
  }
}

static bool split_key_and_val(std::string *key, std::string *val,
                              const std::string &kv_and_eq) {
  auto it = kv_and_eq.find('=');
  if (it == std::string::npos) {
    return false;
  }

  *key = kv_and_eq.substr(0, it);
  *val = kv_and_eq.substr(it + 1);
  return true;
}

void Parse(int *val_out, const char *key_s, int argc, char *argv[]) {
  const std::string target(key_s);
  bool ok = false;
  for (int i = 0; i < argc; ++i) {
    std::string s(argv[i]);

    std::string key, val;
    if (!split_key_and_val(&key, &val, s)) {
      continue;
    }

    if (key == target) {
      *val_out = std::stoi(val);
      ok = true;
    }
  }

  if (!ok) {
    printf("%s not found in program argv, set it to default %d\n",
           target.c_str(), *val_out);
  }
}
