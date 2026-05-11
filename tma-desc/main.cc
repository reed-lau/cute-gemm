/**
 * TMA Descriptor Encoder & Validator
 *
 * Compares NVIDIA's cuTensorMapEncodeTiled() output against our reverse-engineered
 * implementation, and pretty-prints both descriptors with decoded fields.
 *
 * The user provides shape/stride/box/elemStride in any consistent order.
 * The program internally sorts by stride (ascending) to determine TMA API order
 * (dim0 = smallest stride = contiguous innermost dimension).
 *
 * Usage:
 *   ./main <dtype> <swz> <interl> <l2> <oob> <ndim>
 *          <shape0> ... <shape_{ndim-1}>
 *          <stride0> ... <stride_{ndim-1}>      (bytes)
 *          <box0> ... <box_{ndim-1}>
 *          <es0> ... <es_{ndim-1}>
 *
 * Parameters:
 *   dtype:       0=U8 1=U16 2=U32 3=I32 4=U64 5=I64 6=F16 7=F32 8=F64 9=BF16
 *   swizzle:     0=NONE 1=32B 2=64B 3=128B
 *   interleave:  0=NONE 1=16B 2=32B
 *   l2:          0=NONE 1=64B 2=128B 3=256B
 *   oob:         0=ZERO 1=NAN
 *   ndim:        tensor rank (1-5)
 *   shape:       ndim dimension sizes (elements)
 *   stride:      ndim strides (bytes), order matches shape
 *   box:         ndim box dimensions (elements), order matches shape
 *   es:          ndim element strides (1-8), order matches shape
 *
 * Total args: 6 + 4*ndim
 *
 * The stride values encode the memory layout. The program sorts all arrays by
 * stride ascending so that dim0 in TMA = the contiguous (smallest stride) dimension.
 *
 * Batch usage:
 *   cat tests.txt | grep -v '^#' | grep -v '^$' | xargs -L1 ./main
 *
 * Exit code: 0 = PASS (or NVIDIA error skipped), 1 = MISMATCH
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#define __CUDA_API_VERSION 1
#include "cuTensorMapEncodeTiled_impl.h"

// ============================================================================
// Argument parsing helpers
// ============================================================================

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <dtype> <swz> <interl> <l2> <oob> <ndim>\n"
        "          <shape0> ... <shape_{ndim-1}>\n"
        "          <stride0> ... <stride_{ndim-1}>   (bytes)\n"
        "          <box0> ... <box_{ndim-1}>\n"
        "          <es0> ... <es_{ndim-1}>\n"
        "\n"
        "  dtype:   0=U8 1=U16 2=U32 3=I32 4=U64 5=I64 6=F16 7=F32 8=F64 9=BF16\n"
        "  swz:     0=NONE 1=32B 2=64B 3=128B\n"
        "  interl:  0=NONE 1=16B 2=32B\n"
        "  l2:      0=NONE 1=64B 2=128B 3=256B\n"
        "  oob:     0=ZERO 1=NAN\n"
        "  ndim:    1-5\n"
        "  shape:   dimension sizes (elements)\n"
        "  stride:  byte strides for each dimension\n"
        "  box:     box dimensions (elements)\n"
        "  es:      element strides (1-8)\n"
        "\n"
        "Total args = 6 + 4*ndim. Stride determines dimension ordering for TMA API.\n"
        "\n"
        "Examples:\n"
        "  # 2D f32 64x64 contiguous: shape=[64,64] stride=[256,4] box=[64,64] es=[1,1]\n"
        "  %s 7 0 0 0 0 2  64 64  256 4  64 64  1 1\n"
        "\n"
        "  # 2D f16 128x256, box=64x128, swz128:\n"
        "  %s 6 3 0 0 0 2  128 256  512 2  64 128  1 1\n"
        "\n"
        "Batch: cat tests.txt | grep -v '^#' | grep -v '^$' | xargs -L1 %s\n",
        prog, prog, prog, prog);
}

static CUtensorMapDataType parseDtype(unsigned v) {
    switch (v) {
        case 0: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        case 1: return CU_TENSOR_MAP_DATA_TYPE_UINT16;
        case 2: return CU_TENSOR_MAP_DATA_TYPE_UINT32;
        case 3: return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case 4: return CU_TENSOR_MAP_DATA_TYPE_UINT64;
        case 5: return CU_TENSOR_MAP_DATA_TYPE_INT64;
        case 6: return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
        case 7: return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case 8: return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
        case 9: return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        default: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    }
}

static CUtensorMapSwizzle parseSwizzle(unsigned v) {
    switch (v) {
        case 0: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 1: return CU_TENSOR_MAP_SWIZZLE_32B;
        case 2: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 3: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
}

static CUtensorMapInterleave parseInterleave(unsigned v) {
    switch (v) {
        case 0: return CU_TENSOR_MAP_INTERLEAVE_NONE;
        case 1: return CU_TENSOR_MAP_INTERLEAVE_16B;
        case 2: return CU_TENSOR_MAP_INTERLEAVE_32B;
        default: return CU_TENSOR_MAP_INTERLEAVE_NONE;
    }
}

static CUtensorMapL2promotion parseL2(unsigned v) {
    switch (v) {
        case 0: return CU_TENSOR_MAP_L2_PROMOTION_NONE;
        case 1: return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
        case 2: return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
        case 3: return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
        default: return CU_TENSOR_MAP_L2_PROMOTION_NONE;
    }
}

static CUtensorMapFloatOOBfill parseOOB(unsigned v) {
    switch (v) {
        case 0: return CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
        case 1: return CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
        default: return CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    }
}

static const char* dtypeName(unsigned v) {
    const char* names[] = {"U8","U16","U32","I32","U64","I64","F16","F32","F64","BF16"};
    return (v < 10) ? names[v] : "?";
}

// ============================================================================
// Comparison utilities
// ============================================================================

static int compareDescriptors(const CUtensorMap& ref, const CUtensorMap& impl) {
    const uint8_t* r = (const uint8_t*)&ref;
    const uint8_t* o = (const uint8_t*)&impl;
    int diffs = 0;
    for (int i = 0; i < 128; i++) {
        if (r[i] != o[i]) diffs++;
    }
    return diffs;
}

static void printDiffBytes(const CUtensorMap& ref, const CUtensorMap& impl) {
    const uint8_t* r = (const uint8_t*)&ref;
    const uint8_t* o = (const uint8_t*)&impl;

    printf("Byte-level diff (offset: REF != IMPL):\n");
    for (int i = 0; i < 128; i++) {
        if (r[i] != o[i]) {
            printf("  [%3d] 0x%02X != 0x%02X\n", i, r[i], o[i]);
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // First 6 fixed params: dtype swz interl l2 oob ndim
    if (argc < 7) {
        printUsage(argv[0]);
        return 1;
    }

    unsigned argDtype      = atoi(argv[1]);
    unsigned argSwizzle    = atoi(argv[2]);
    unsigned argInterleave = atoi(argv[3]);
    unsigned argL2         = atoi(argv[4]);
    unsigned argOOB        = atoi(argv[5]);
    int      ndim          = atoi(argv[6]);

    if (ndim < 1 || ndim > 5) {
        fprintf(stderr, "Error: ndim must be 1-5, got %d\n", ndim);
        return 1;
    }

    // Total args needed: 6 + 4*ndim (shape*ndim + stride*ndim + box*ndim + es*ndim)
    int requiredArgs = 6 + 4 * ndim;
    if (argc < 1 + requiredArgs) {
        fprintf(stderr, "Error: ndim=%d requires %d params (got %d).\n",
                ndim, requiredArgs, argc - 1);
        fprintf(stderr, "  Need: 6 fixed + %d shape + %d stride + %d box + %d es = %d\n",
                ndim, ndim, ndim, ndim, requiredArgs);
        printUsage(argv[0]);
        return 1;
    }

    // Parse user-order arrays (ndim elements each)
    uint64_t userShape[5]  = {1, 1, 1, 1, 1};
    uint64_t userStride[5] = {0, 0, 0, 0, 0};
    uint32_t userBox[5]    = {1, 1, 1, 1, 1};
    uint32_t userES[5]     = {1, 1, 1, 1, 1};

    int argIdx = 7;
    for (int i = 0; i < ndim; i++)
        userShape[i] = strtoull(argv[argIdx++], NULL, 0);
    for (int i = 0; i < ndim; i++)
        userStride[i] = strtoull(argv[argIdx++], NULL, 0);
    for (int i = 0; i < ndim; i++)
        userBox[i] = atoi(argv[argIdx++]);
    for (int i = 0; i < ndim; i++)
        userES[i] = atoi(argv[argIdx++]);

    // Convert to CUDA enum types
    CUtensorMapDataType    dtype  = parseDtype(argDtype);
    CUtensorMapSwizzle     swz    = parseSwizzle(argSwizzle);
    CUtensorMapInterleave  interl = parseInterleave(argInterleave);
    CUtensorMapL2promotion l2     = parseL2(argL2);
    CUtensorMapFloatOOBfill oob   = parseOOB(argOOB);
    uint32_t elemSize = getElementSizeBytes(dtype);

    // ----------------------------------------------------------------
    // Sort by stride ascending to get TMA order (dim0 = smallest stride)
    // ----------------------------------------------------------------
    int perm[5] = {0, 1, 2, 3, 4};
    std::sort(perm, perm + ndim, [&](int a, int b) {
        return userStride[a] < userStride[b];
    });

    // Reorder all arrays into TMA order
    cuuint64_t globalDim[5]      = {1, 1, 1, 1, 1};
    uint64_t   sortedStride[5]   = {0, 0, 0, 0, 0};
    cuuint32_t boxDim[5]         = {1, 1, 1, 1, 1};
    cuuint32_t elementStrides[5] = {1, 1, 1, 1, 1};

    for (int i = 0; i < ndim; i++) {
        globalDim[i]      = userShape[perm[i]];
        sortedStride[i]   = userStride[perm[i]];
        boxDim[i]         = userBox[perm[i]];
        elementStrides[i] = userES[perm[i]];
    }

    // ----------------------------------------------------------------
    // Validate
    // ----------------------------------------------------------------
    bool hasWarning = false;

    // Check innermost stride == elemSize
    if (sortedStride[0] != elemSize) {
        fprintf(stderr, "WARNING: innermost stride = %lu bytes, expected elemSize = %u bytes.\n"
                        "         (dimension '%d' in user order has smallest stride)\n",
                (unsigned long)sortedStride[0], elemSize, perm[0]);
        hasWarning = true;
    }

    // Check remaining strides are multiples of 16
    for (int i = 1; i < ndim; i++) {
        if (sortedStride[i] % 16 != 0) {
            fprintf(stderr, "WARNING: stride[%d] = %lu bytes is not a multiple of 16.\n"
                            "         (TMA requires globalStrides to be multiples of 16)\n",
                    perm[i], (unsigned long)sortedStride[i]);
            hasWarning = true;
        }
    }

    // ----------------------------------------------------------------
    // Compute globalStrides for TMA API: ndim-1 strides for dims 1..ndim-1
    // (dim0 stride is implicit = elemSize, not passed to API)
    // ----------------------------------------------------------------
    cuuint64_t globalStrides[4] = {0, 0, 0, 0};
    for (int i = 0; i < ndim - 1 && i < 4; i++) {
        globalStrides[i] = sortedStride[i + 1];
    }

    // ----------------------------------------------------------------
    // Print test parameters
    // ----------------------------------------------------------------
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("TEST: dtype=%s(%u) swz=%u interl=%u l2=%u oob=%u ndim=%d\n",
           dtypeName(argDtype), argDtype, argSwizzle, argInterleave, argL2, argOOB, ndim);
    printf("\n");
    printf("  User input (original order):\n");
    printf("    shape  = [");
    for (int i = 0; i < ndim; i++) printf("%s%lu", i?", ":"", (unsigned long)userShape[i]);
    printf("]\n");
    printf("    stride = [");
    for (int i = 0; i < ndim; i++) printf("%s%lu", i?", ":"", (unsigned long)userStride[i]);
    printf("] bytes\n");
    printf("    box    = [");
    for (int i = 0; i < ndim; i++) printf("%s%u", i?", ":"", userBox[i]);
    printf("]\n");
    printf("    es     = [");
    for (int i = 0; i < ndim; i++) printf("%s%u", i?", ":"", userES[i]);
    printf("]\n");
    printf("\n");
    printf("  Sorted by stride (TMA API order, dim0=innermost):\n");
    printf("    perm       = [");
    for (int i = 0; i < ndim; i++) printf("%s%d", i?", ":"", perm[i]);
    printf("]\n");
    printf("    globalDim  = [");
    for (int i = 0; i < ndim; i++) printf("%s%lu", i?", ":"", (unsigned long)globalDim[i]);
    printf("]\n");
    printf("    stride     = [");
    for (int i = 0; i < ndim; i++) printf("%s%lu", i?", ":"", (unsigned long)sortedStride[i]);
    printf("] bytes\n");
    printf("    API strides= [");
    for (int i = 0; i < ndim - 1; i++) printf("%s%lu", i?", ":"", (unsigned long)globalStrides[i]);
    printf("] bytes (innermost stride=%u implicit)\n", elemSize);
    printf("    boxDim     = [");
    for (int i = 0; i < ndim; i++) printf("%s%u", i?", ":"", boxDim[i]);
    printf("]\n");
    printf("    elemStride = [");
    for (int i = 0; i < ndim; i++) printf("%s%u", i?", ":"", elementStrides[i]);
    printf("]\n");

    if (hasWarning) printf("\n  ⚠ See warnings above.\n");
    printf("════════════════════════════════════════════════════════════════════\n\n");

    // ----------------------------------------------------------------
    // Allocate device memory for base address
    // ----------------------------------------------------------------
    void* devPtr = nullptr;
    cudaError_t cudaErr = cudaMalloc(&devPtr, 1024 * 1024);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // --- Run NVIDIA's implementation ---
    CUtensorMap refMap;
    memset(&refMap, 0, sizeof(refMap));
    CUresult refErr = cuTensorMapEncodeTiled(
        &refMap, dtype, ndim, devPtr,
        globalDim, globalStrides, boxDim, elementStrides,
        interl, swz, l2, oob);

    if (refErr != CUDA_SUCCESS) {
        const char* errStr = "unknown";
        cuGetErrorString(refErr, &errStr);
        printf("[SKIP] NVIDIA cuTensorMapEncodeTiled returned error %d (%s)\n", refErr, errStr);
        cudaFree(devPtr);
        return 0;  // Not a mismatch, just an invalid config
    }

    // --- Run our implementation ---
    CUtensorMap implMap;
    memset(&implMap, 0, sizeof(implMap));
    CUresult implErr = cuTensorMapEncodeTiled_impl(
        &implMap, dtype, ndim, devPtr,
        globalDim, globalStrides, boxDim, elementStrides,
        interl, swz, l2, oob);

    if (implErr != CUDA_SUCCESS) {
        printf("[FAIL] Our implementation returned error %d\n", implErr);
        printf("\nNVIDIA reference (succeeded):\n");
        cuTensorMapPrint(&refMap, "NVIDIA Reference");
        cudaFree(devPtr);
        return 1;
    }

    // --- Compare ---
    int diffs = compareDescriptors(refMap, implMap);

    if (diffs == 0) {
        printf("[PASS] Descriptors match byte-for-byte (128 bytes identical)\n\n");
        cuTensorMapPrint(&refMap, "NVIDIA Reference = Our Implementation");
    } else {
        printf("[FAIL] %d byte(s) differ!\n\n", diffs);

        cuTensorMapPrint(&refMap, "NVIDIA Reference");
        cuTensorMapPrint(&implMap, "Our Implementation");

        printf("────────────────────────────────────────────────────────────────────\n");
        printDiffBytes(refMap, implMap);
        printf("────────────────────────────────────────────────────────────────────\n");
    }

    cudaFree(devPtr);
    return (diffs == 0) ? 0 : 1;
}
