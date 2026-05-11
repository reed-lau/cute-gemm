/**
 * cuTensorMapEncodeTiled() reverse-engineered implementation.
 *
 * This file provides a software-only implementation of NVIDIA's cuTensorMapEncodeTiled()
 * function, which constructs a 128-byte TMA (Tensor Memory Access) descriptor for the
 * SM90+ (Hopper) and later GPU architectures.
 *
 * The TMA descriptor is an opaque 128-byte structure (CUtensorMap) that encodes:
 *   - Global memory base address
 *   - Tensor dimensions and strides
 *   - Bounding box dimensions for shared memory tile
 *   - Element strides (for strided access patterns)
 *   - Data type, swizzle mode, interleave mode, L2 promotion, and OOB fill
 *
 * Binary layout of the 128-byte descriptor (little-endian):
 * ============================================================================
 * Bytes  0- 7: globalAddress (64-bit pointer)
 * Bytes  8-11: Control word (bitfield encoding dtype, ndim, swizzle, etc.)
 * Bytes 12-15: globalStrides[0] / 16 (lower 32 bits)
 * Bytes 16-19: globalStrides[1] / 16 (lower 32 bits)
 * Bytes 20-23: globalStrides[2] / 16 (lower 32 bits)
 * Bytes 24-27: globalStrides[3] / 16 (lower 32 bits)
 * Bytes 28-29: stride upper bits (4 bits per stride, packed as nibbles)
 * Bytes 30-31: reserved (zero)
 * Bytes 32-35: globalDim[0] - 1  (32-bit)
 * Bytes 36-39: globalDim[1] - 1  (32-bit)
 * Bytes 40-43: globalDim[2] - 1  (32-bit)
 * Bytes 44-47: globalDim[3] - 1  (32-bit)
 * Bytes 48-51: globalDim[4] - 1  (32-bit)
 * Bytes 52-53: elementStrides (3 bits per dimension, packed)
 * Byte  54:    elementStrides continued (bits for dims 3-4)
 * Byte  55:    boxDim[0] - 1
 * Byte  56:    boxDim[1] - 1
 * Byte  57:    boxDim[2] - 1
 * Byte  58:    boxDim[3] - 1
 * Byte  59:    boxDim[4] - 1
 * Bytes 60-63: reserved (zero)
 * Bytes 64-67: total box bytes in smem (product of effective box dims * elemSize)
 * Bytes 68-71: reserved (zero)
 * Bytes 72-75: smem swizzle stride value
 * Bytes 76-127: reserved (zero)
 * ============================================================================
 *
 * Control word (bytes 8-11) bitfield layout:
 *   bits[ 3: 0] = 0 (tensor type: tiled)
 *   bits[ 6: 4] = tensorRank - 1
 *   bits[10: 7] = internal dtype code (4 bits)
 *   bits[12:11] = interleave (0=none, 1=16B, 2=32B)
 *   bits[14:13] = swizzle (0=none, 1=32B, 2=64B, 3=128B)
 *   bit [15]    = oobFill (0=zero, 1=NaN)
 *   bit [16]    = 0 (reserved)
 *   bits[18:17] = l2Promotion (0=none, 1=64B, 2=128B, 3=256B)
 *   bits[31:19] = 0 (reserved)
 */

#ifndef CU_TENSOR_MAP_ENCODE_TILED_IMPL_H
#define CU_TENSOR_MAP_ENCODE_TILED_IMPL_H

#include <cstdint>
#include <cstring>

// ============================================================================
// Type definitions (mirrors CUDA driver API types)
// ============================================================================

#ifndef CU_TENSOR_MAP_NUM_QWORDS
#define CU_TENSOR_MAP_NUM_QWORDS 16
#endif

// Forward declarations if not using CUDA headers
#ifndef __cuda_cuda_h__

typedef uint64_t cuuint64_t;
typedef uint32_t cuuint32_t;

struct alignas(128) CUtensorMap_st {
    cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
};
typedef CUtensorMap_st CUtensorMap;

enum CUtensorMapDataType {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16,
    CU_TENSOR_MAP_DATA_TYPE_UINT32,
    CU_TENSOR_MAP_DATA_TYPE_INT32,
    CU_TENSOR_MAP_DATA_TYPE_UINT64,
    CU_TENSOR_MAP_DATA_TYPE_INT64,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
};

enum CUtensorMapInterleave {
    CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
    CU_TENSOR_MAP_INTERLEAVE_16B,
    CU_TENSOR_MAP_INTERLEAVE_32B,
};

enum CUtensorMapSwizzle {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B,
    CU_TENSOR_MAP_SWIZZLE_64B,
    CU_TENSOR_MAP_SWIZZLE_128B,
};

enum CUtensorMapL2promotion {
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
};

enum CUtensorMapFloatOOBfill {
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
};

enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
};

#endif // __cuda_cuda_h__

// ============================================================================
// Helper: get element size in bytes from data type enum
// ============================================================================
static inline uint32_t getElementSizeBytes(CUtensorMapDataType dtype) {
    switch (dtype) {
        case CU_TENSOR_MAP_DATA_TYPE_UINT8:        return 1;
        case CU_TENSOR_MAP_DATA_TYPE_UINT16:       return 2;
        case CU_TENSOR_MAP_DATA_TYPE_UINT32:       return 4;
        case CU_TENSOR_MAP_DATA_TYPE_INT32:        return 4;
        case CU_TENSOR_MAP_DATA_TYPE_UINT64:       return 8;
        case CU_TENSOR_MAP_DATA_TYPE_INT64:        return 8;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:      return 2;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:      return 4;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:      return 8;
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:     return 2;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:  return 4;
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:     return 4;
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return 4;
        default: return 0;
    }
}

// ============================================================================
// Helper: map API dtype enum to internal 4-bit hardware encoding
// ============================================================================
// The hardware uses a different numbering than the CUDA API enum.
// API enum -> internal hardware code:
//   UINT8(0)       -> 0
//   UINT16(1)      -> 1
//   UINT32(2)      -> 2
//   INT32(3)       -> 3
//   UINT64(4)      -> 4
//   INT64(5)       -> 5
//   FLOAT16(6)     -> 6
//   FLOAT32(7)     -> 7
//   FLOAT64(8)     -> 9   (skips 8!)
//   BFLOAT16(9)    -> 10
//   FLOAT32_FTZ(10)-> 8
//   TFLOAT32(11)   -> 11
//   TFLOAT32_FTZ(12)-> 12
static inline uint32_t getInternalDtypeCode(CUtensorMapDataType dtype) {
    switch (dtype) {
        case CU_TENSOR_MAP_DATA_TYPE_UINT8:        return 0;
        case CU_TENSOR_MAP_DATA_TYPE_UINT16:       return 1;
        case CU_TENSOR_MAP_DATA_TYPE_UINT32:       return 2;
        case CU_TENSOR_MAP_DATA_TYPE_INT32:        return 3;
        case CU_TENSOR_MAP_DATA_TYPE_UINT64:       return 4;
        case CU_TENSOR_MAP_DATA_TYPE_INT64:        return 5;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:      return 6;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:      return 7;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:      return 9;
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:     return 10;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:  return 8;
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:     return 11;
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return 12;
        default: return 0;
    }
}

// ============================================================================
// Helper: get the shared memory swizzle stride value
// ============================================================================
// This value is stored at bytes[72..75] and represents the stride pattern
// used by the TMA hardware when writing to shared memory.
//   SWIZZLE_NONE:  16   (0x10)  - 16-byte natural alignment
//   SWIZZLE_32B:   256  (0x100) - 32B * 8 rows per swizzle group
//   SWIZZLE_64B:   512  (0x200) - 64B * 8 rows per swizzle group
//   SWIZZLE_128B:  1024 (0x400) - 128B * 8 rows per swizzle group
static inline uint32_t getSmemSwizzleStride(CUtensorMapSwizzle swizzle) {
    switch (swizzle) {
        case CU_TENSOR_MAP_SWIZZLE_NONE: return 16;
        case CU_TENSOR_MAP_SWIZZLE_32B:  return 256;
        case CU_TENSOR_MAP_SWIZZLE_64B:  return 512;
        case CU_TENSOR_MAP_SWIZZLE_128B: return 1024;
        default: return 16;
    }
}

// ============================================================================
// Main implementation: cuTensorMapEncodeTiled
// ============================================================================
/**
 * Software implementation of cuTensorMapEncodeTiled.
 * Constructs a 128-byte TMA descriptor for tiled memory access patterns.
 *
 * @param tensorMap       Output: 128-byte aligned descriptor (must be 128-byte aligned)
 * @param tensorDataType  Element data type
 * @param tensorRank      Number of dimensions (1-5)
 * @param globalAddress   Base address of the tensor in global memory (must be 16-byte aligned)
 * @param globalDim       Array of tensorRank dimension sizes (in elements)
 * @param globalStrides   Array of (tensorRank-1) byte strides for dims 1..N-1
 *                        (stride for dim 0 is implicit = elementSize)
 * @param boxDim          Array of tensorRank bounding box sizes (in elements)
 * @param elementStrides  Array of tensorRank element strides (1-8 per dim; dim0 is ignored by HW)
 * @param interleave      Interleaved layout mode
 * @param swizzle         Shared memory bank swizzle pattern
 * @param l2Promotion     L2 cache promotion granularity
 * @param oobFill         Out-of-bounds fill value
 *
 * @return CUDA_SUCCESS on success, CUDA_ERROR_INVALID_VALUE on invalid parameters
 */
static inline CUresult cuTensorMapEncodeTiled_impl(
    CUtensorMap*             tensorMap,
    CUtensorMapDataType      tensorDataType,
    cuuint32_t               tensorRank,
    void*                    globalAddress,
    const cuuint64_t*        globalDim,
    const cuuint64_t*        globalStrides,
    const cuuint32_t*        boxDim,
    const cuuint32_t*        elementStrides,
    CUtensorMapInterleave    interleave,
    CUtensorMapSwizzle       swizzle,
    CUtensorMapL2promotion   l2Promotion,
    CUtensorMapFloatOOBfill  oobFill)
{
    // ---- Parameter validation ----
    if (!tensorMap || !globalDim || !globalStrides || !boxDim || !elementStrides)
        return CUDA_ERROR_INVALID_VALUE;
    if (tensorRank < 1 || tensorRank > 5)
        return CUDA_ERROR_INVALID_VALUE;

    uint32_t elemSize = getElementSizeBytes(tensorDataType);
    if (elemSize == 0)
        return CUDA_ERROR_INVALID_VALUE;

    // ---- Zero out the descriptor ----
    uint8_t* desc = reinterpret_cast<uint8_t*>(tensorMap);
    memset(desc, 0, 128);

    // ================================================================
    // Bytes 0-7: Global address (64-bit, little-endian)
    // ================================================================
    uint64_t addr = reinterpret_cast<uint64_t>(globalAddress);
    memcpy(&desc[0], &addr, 8);

    // ================================================================
    // Bytes 8-11: Control word
    // ================================================================
    // Layout:
    //   bits[ 3: 0] = 0 (tiled mode)
    //   bits[ 6: 4] = tensorRank - 1
    //   bits[10: 7] = internal dtype code
    //   bits[12:11] = interleave mode
    //   bits[14:13] = swizzle mode
    //   bit [15]    = OOB fill
    //   bit [16]    = 0 (reserved)
    //   bits[18:17] = L2 promotion
    //   bits[31:19] = 0 (reserved)
    uint32_t control = 0;
    control |= ((tensorRank - 1) & 0x7) << 4;
    control |= (getInternalDtypeCode(tensorDataType) & 0xF) << 7;
    control |= ((uint32_t)interleave & 0x3) << 11;
    control |= ((uint32_t)swizzle & 0x3) << 13;
    control |= ((uint32_t)oobFill & 0x1) << 15;
    control |= ((uint32_t)l2Promotion & 0x3) << 17;
    
    // ================================================================
    // Bit 21: Hardware optimization flag
    // Set when total tensor data >= 128KB (131072 bytes).
    // Total tensor data = product(globalDim[0..rank-1]) * elemSize.
    // This enables a hardware path optimized for large tensors.
    // (Driver 580+; older drivers used a different heuristic.)
    // ================================================================
    uint64_t totalTensorBytes = (uint64_t)elemSize;
    for (cuuint32_t i = 0; i < tensorRank; ++i) {
        totalTensorBytes *= globalDim[i];
    }
    if (totalTensorBytes >= 131072) {
        control |= (1u << 21);
    }

    memcpy(&desc[8], &control, 4);

    // ================================================================
    // Bytes 12-27: Global strides (divided by 16)
    // Each stride is a 36-bit value: lower 32 bits at offset 12+i*4,
    // upper 4 bits packed into bytes 28-29 as nibbles.
    // ================================================================
    // The API provides (tensorRank - 1) strides for dimensions 1..N-1.
    // Strides are in bytes and must be multiples of 16.
    // Stride for dimension 0 is implicit (= element size).
    uint8_t stride_upper_byte28 = 0;
    uint8_t stride_upper_byte29 = 0;

    for (uint32_t i = 0; i < tensorRank - 1 && i < 4; i++) {
        uint64_t stride_div16 = globalStrides[i] >> 4;  // divide by 16
        uint32_t lower32 = (uint32_t)(stride_div16 & 0xFFFFFFFF);
        uint8_t  upper4  = (uint8_t)((stride_div16 >> 32) & 0xF);

        // Store lower 32 bits at bytes[12 + i*4]
        memcpy(&desc[12 + i * 4], &lower32, 4);

        // Pack upper 4 bits into bytes[28..29]
        if (i < 2) {
            stride_upper_byte28 |= (upper4 << (i * 4));
        } else {
            stride_upper_byte29 |= (upper4 << ((i - 2) * 4));
        }
    }
    desc[28] = stride_upper_byte28;
    desc[29] = stride_upper_byte29;

    // ================================================================
    // Bytes 32-51: Global dimensions (each stored as dim - 1, 32-bit LE)
    // ================================================================
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t dimVal = 0;
        if (i < tensorRank && globalDim[i] > 0) {
            dimVal = (uint32_t)(globalDim[i] - 1);
        }
        memcpy(&desc[32 + i * 4], &dimVal, 4);
    }

    // ================================================================
    // Bytes 52-54: Element strides (3 bits per dimension, packed)
    // Layout: bits[2:0]=es[0]-1, bits[5:3]=es[1]-1, bits[8:6]=es[2]-1,
    //         bits[11:9]=es[3]-1, bits[14:12]=es[4]-1
    // ================================================================
    uint32_t elemStrideBits = 0;
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t es = (i < tensorRank) ? elementStrides[i] : 1;
        if (es < 1) es = 1;
        if (es > 8) es = 8;
        elemStrideBits |= ((es - 1) & 0x7) << (i * 3);
    }
    desc[52] = (uint8_t)(elemStrideBits & 0xFF);
    desc[53] = (uint8_t)((elemStrideBits >> 8) & 0xFF);
    // Only bits[14:12] would go into bit 0-6 of desc[53] is wrong...
    // Actually 15 bits total: byte[52] has bits[7:0], byte[53] has bits[14:8] (7 bits)
    // Already handled above.

    // ================================================================
    // Bytes 55-59: Box dimensions (each stored as boxDim - 1, 8-bit)
    // Max boxDim = 256, stored as uint8_t (boxDim-1), so 0..255.
    // ================================================================
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t bd = (i < tensorRank) ? boxDim[i] : 1;
        if (bd < 1) bd = 1;
        desc[55 + i] = (uint8_t)(bd - 1);
    }

    // ================================================================
    // Bytes 64-67: Total box size in shared memory (bytes)
    // = product(boxDim[i] / elemStride[i]) * baseSize
    // where baseSize = interleave_bytes when interleave != NONE,
    //                  elemSize          when interleave == NONE.
    // Division is integer division for each dimension.
    // ================================================================
    uint32_t baseSize = elemSize;
    if (interleave == CU_TENSOR_MAP_INTERLEAVE_16B) {
        baseSize = 16;
    } else if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        baseSize = 32;
    }
    uint32_t totalBoxBytes = baseSize;
    for (uint32_t i = 0; i < tensorRank; i++) {
        uint32_t es = elementStrides[i];
        if (es < 1) es = 1;
        uint32_t effectiveDim = boxDim[i] / es;
        if (effectiveDim < 1) effectiveDim = 1;
        totalBoxBytes *= effectiveDim;
    }
    memcpy(&desc[64], &totalBoxBytes, 4);

    // ================================================================
    // Bytes 72-75: Shared memory swizzle stride
    // Depends only on the swizzle mode, not on tensor dimensions.
    // ================================================================
    uint32_t smemSwizzleStride = getSmemSwizzleStride(swizzle);
    memcpy(&desc[72], &smemSwizzleStride, 4);

    return CUDA_SUCCESS;
}

// ============================================================================
// Debug: pretty-print a TMA descriptor by decoding all fields
// ============================================================================

static inline const char* dtypeCodeToString(uint32_t code) {
    switch (code) {
        case 0:  return "UINT8";
        case 1:  return "UINT16";
        case 2:  return "UINT32";
        case 3:  return "INT32";
        case 4:  return "UINT64";
        case 5:  return "INT64";
        case 6:  return "FLOAT16";
        case 7:  return "FLOAT32";
        case 8:  return "FLOAT32_FTZ";
        case 9:  return "FLOAT64";
        case 10: return "BFLOAT16";
        case 11: return "TFLOAT32";
        case 12: return "TFLOAT32_FTZ";
        default: return "UNKNOWN";
    }
}

static inline uint32_t dtypeCodeToElemSize(uint32_t code) {
    switch (code) {
        case 0:  return 1;  // UINT8
        case 1:  return 2;  // UINT16
        case 2:  return 4;  // UINT32
        case 3:  return 4;  // INT32
        case 4:  return 8;  // UINT64
        case 5:  return 8;  // INT64
        case 6:  return 2;  // FLOAT16
        case 7:  return 4;  // FLOAT32
        case 8:  return 4;  // FLOAT32_FTZ
        case 9:  return 8;  // FLOAT64
        case 10: return 2;  // BFLOAT16
        case 11: return 4;  // TFLOAT32
        case 12: return 4;  // TFLOAT32_FTZ
        default: return 0;
    }
}

static inline const char* swizzleToString(uint32_t swz) {
    switch (swz) {
        case 0: return "NONE";
        case 1: return "32B";
        case 2: return "64B";
        case 3: return "128B";
        default: return "UNKNOWN";
    }
}

static inline const char* interleaveToString(uint32_t interl) {
    switch (interl) {
        case 0: return "NONE";
        case 1: return "16B";
        case 2: return "32B";
        default: return "UNKNOWN";
    }
}

static inline const char* l2promoToString(uint32_t l2) {
    switch (l2) {
        case 0: return "NONE";
        case 1: return "64B";
        case 2: return "128B";
        case 3: return "256B";
        default: return "UNKNOWN";
    }
}

static inline const char* oobFillToString(uint32_t oob) {
    switch (oob) {
        case 0: return "ZERO";
        case 1: return "NAN_REQUEST_ZERO_FMA";
        default: return "UNKNOWN";
    }
}

/**
 * Pretty-print a TMA descriptor by decoding all encoded fields.
 *
 * @param tensorMap  Pointer to a 128-byte TMA descriptor (CUtensorMap)
 * @param label      Optional label to print as a header (can be NULL)
 */
static inline void cuTensorMapPrint(const CUtensorMap* tensorMap, const char* label = nullptr) {
    const uint8_t* desc = reinterpret_cast<const uint8_t*>(tensorMap);

    // --- Raw hex dump ---
    if (label) {
        printf("=== TMA Descriptor: %s ===\n", label);
    } else {
        printf("=== TMA Descriptor ===\n");
    }

    printf("Raw bytes:\n");
    for (int i = 0; i < 128; i++) {
        if (i % 16 == 0) printf("  [%3d] ", i);
        printf("%02X ", desc[i]);
        if (i % 16 == 15) printf("\n");
    }
    printf("\n");

    // --- Decode globalAddress (bytes 0-7) ---
    uint64_t globalAddr;
    memcpy(&globalAddr, &desc[0], 8);
    printf("Global Address:     0x%016lX\n", (unsigned long)globalAddr);

    // --- Decode control word (bytes 8-11) ---
    uint32_t control;
    memcpy(&control, &desc[8], 4);

    uint32_t tensorType  = (control >> 0) & 0xF;
    uint32_t ndim        = ((control >> 4) & 0x7) + 1;
    uint32_t dtypeCode   = (control >> 7) & 0xF;
    uint32_t interleave  = (control >> 11) & 0x3;
    uint32_t swizzle     = (control >> 13) & 0x3;
    uint32_t oobFill     = (control >> 15) & 0x1;
    uint32_t l2Promo     = (control >> 17) & 0x3;
    uint32_t bit21       = (control >> 21) & 0x1;
    uint32_t elemSize    = dtypeCodeToElemSize(dtypeCode);

    printf("\nControl Word:       0x%08X\n", control);
    printf("  Tensor Type:      %u (0=tiled)\n", tensorType);
    printf("  Rank (ndim):      %u\n", ndim);
    printf("  Data Type:        %s (internal code=%u, elemSize=%u bytes)\n",
           dtypeCodeToString(dtypeCode), dtypeCode, elemSize);
    printf("  Interleave:       %s (code=%u)\n", interleaveToString(interleave), interleave);
    printf("  Swizzle:          %s (code=%u)\n", swizzleToString(swizzle), swizzle);
    printf("  OOB Fill:         %s (code=%u)\n", oobFillToString(oobFill), oobFill);
    printf("  L2 Promotion:     %s (code=%u)\n", l2promoToString(l2Promo), l2Promo);
    printf("  bit[21] (OOB ctrl): %u", bit21);
    if (bit21)
        printf(" → OOB protection DISABLED (tensor ≥ 128KB)");
    else
        printf(" → OOB protection ENABLED (tensor < 128KB)");
    printf("\n");

    // --- Decode global strides (bytes 12-29) ---
    // Lower 32 bits at bytes[12 + i*4], upper 4 bits in bytes[28..29] nibbles
    printf("\nGlobal Strides (bytes, stored as stride/16):\n");
    for (uint32_t i = 0; i < 4 && i < ndim - 1; i++) {
        uint32_t lower32;
        memcpy(&lower32, &desc[12 + i * 4], 4);

        uint8_t upperByte = (i < 2) ? desc[28] : desc[29];
        uint8_t upper4 = (i % 2 == 0) ? (upperByte & 0xF) : ((upperByte >> 4) & 0xF);

        uint64_t stride_div16 = ((uint64_t)upper4 << 32) | (uint64_t)lower32;
        uint64_t stride_bytes = stride_div16 << 4;  // multiply by 16

        printf("  stride[%u]:        %lu bytes (stored: %lu = 0x%lX, raw_hi4=0x%X)\n",
               i, (unsigned long)stride_bytes, (unsigned long)stride_div16,
               (unsigned long)stride_div16, upper4);
    }

    // --- Decode global dimensions (bytes 32-51) ---
    printf("\nGlobal Dimensions (stored as dim-1):\n");
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t dimMinusOne;
        memcpy(&dimMinusOne, &desc[32 + i * 4], 4);
        uint32_t dim = dimMinusOne + 1;
        if (i < ndim) {
            printf("  globalDim[%u]:     %u (stored: %u)\n", i, dim, dimMinusOne);
        } else if (dimMinusOne != 0) {
            printf("  globalDim[%u]:     %u (stored: %u) [UNUSED but non-zero!]\n", i, dim, dimMinusOne);
        }
    }

    // --- Decode element strides (bytes 52-53, 15 bits) ---
    uint32_t elemStrideBits = (uint32_t)desc[52] | ((uint32_t)desc[53] << 8);
    printf("\nElement Strides (stored as elemStride-1, 3 bits each):\n");
    for (uint32_t i = 0; i < 5 && i < ndim; i++) {
        uint32_t es = ((elemStrideBits >> (i * 3)) & 0x7) + 1;
        printf("  elemStride[%u]:    %u%s\n", i, es,
               (i == 0) ? " (dim0 stride ignored by HW for addressing)" : "");
    }

    // --- Decode box dimensions (bytes 55-59) ---
    printf("\nBox Dimensions (stored as boxDim-1, 8 bits each):\n");
    for (uint32_t i = 0; i < 5 && i < ndim; i++) {
        uint32_t bd = (uint32_t)desc[55 + i] + 1;
        printf("  boxDim[%u]:        %u (stored: %u)\n", i, bd, desc[55 + i]);
    }

    // --- Decode total box bytes (bytes 64-67) ---
    uint32_t totalBoxBytes;
    memcpy(&totalBoxBytes, &desc[64], 4);
    printf("\nShared Memory Box:\n");
    printf("  Total Box Bytes:  %u (0x%X)\n", totalBoxBytes, totalBoxBytes);

    // --- Decode smem swizzle stride (bytes 72-75) ---
    uint32_t smemSwzStride;
    memcpy(&smemSwzStride, &desc[72], 4);
    printf("  Swizzle Stride:   %u (0x%X)", smemSwzStride, smemSwzStride);
    if (smemSwzStride == 16) printf(" [no swizzle, 16B atom]");
    else if (smemSwzStride == 256) printf(" [32B swizzle, 8 rows]");
    else if (smemSwzStride == 512) printf(" [64B swizzle, 8 rows]");
    else if (smemSwzStride == 1024) printf(" [128B swizzle, 8 rows]");
    printf("\n");

    // --- Derived info ---
    printf("\nDerived Info:\n");
    if (elemSize > 0) {
        printf("  boxDim[0] * elemSize = %u bytes", (uint32_t)(desc[55] + 1) * elemSize);
        if (swizzle > 0) {
            uint32_t swzBytes = (swizzle == 1) ? 32 : (swizzle == 2) ? 64 : 128;
            uint32_t innerBytes = (uint32_t)(desc[55] + 1) * elemSize;
            if (innerBytes <= swzBytes)
                printf(" <= %uB swizzle ✓", swzBytes);
            else
                printf(" > %uB swizzle ✗ INVALID!", swzBytes);
        }
        printf("\n");

        // Compute total tensor bytes and verify bit21 consistency
        uint64_t totalBytes = (uint64_t)elemSize;
        for (uint32_t i = 0; i < ndim; i++) {
            uint32_t dimMinusOne;
            memcpy(&dimMinusOne, &desc[32 + i * 4], 4);
            totalBytes *= (uint64_t)(dimMinusOne + 1);
        }
        uint32_t expectedBit21 = (totalBytes >= 131072) ? 1 : 0;
        printf("  Total tensor data: %lu bytes (%s 128KB)\n",
               (unsigned long)totalBytes, totalBytes >= 131072 ? ">=" : "<");
        printf("  bit21 expected:    %u  actual: %u", expectedBit21, bit21);
        if (expectedBit21 != bit21)
            printf("  ⚠ INCONSISTENT (OOB protection may not match tensor size!)");
        printf("\n");
    }

    // Check for non-zero bytes in reserved regions
    bool hasReserved = false;
    for (int i = 76; i < 128; i++) {
        if (desc[i] != 0) { hasReserved = true; break; }
    }
    if (hasReserved) {
        printf("  WARNING: non-zero bytes in reserved region [76..127]!\n");
    }

    printf("\n");
}

// ============================================================================
// Device-side: pretty-print a TMA descriptor from GPU kernel
// ============================================================================
// Usage:
//   __global__ void debug_kernel(const CUtensorMap* tmap) {
//       if (threadIdx.x == 0) cuTensorMapPrint_device(tmap);
//   }
//
// Note: printf from device has limited buffer (~1MB per kernel launch).
//       Call from a single thread only.

#ifdef __CUDACC__

__device__ inline const char* dtypeCodeToString_d(uint32_t code) {
    switch (code) {
        case 0:  return "UINT8";
        case 1:  return "UINT16";
        case 2:  return "UINT32";
        case 3:  return "INT32";
        case 4:  return "UINT64";
        case 5:  return "INT64";
        case 6:  return "FLOAT16";
        case 7:  return "FLOAT32";
        case 8:  return "FLOAT32_FTZ";
        case 9:  return "FLOAT64";
        case 10: return "BFLOAT16";
        case 11: return "TFLOAT32";
        case 12: return "TFLOAT32_FTZ";
        default: return "UNKNOWN";
    }
}

__device__ inline uint32_t dtypeCodeToElemSize_d(uint32_t code) {
    switch (code) {
        case 0:  return 1;
        case 1:  return 2;
        case 2:  return 4;
        case 3:  return 4;
        case 4:  return 8;
        case 5:  return 8;
        case 6:  return 2;
        case 7:  return 4;
        case 8:  return 4;
        case 9:  return 8;
        case 10: return 2;
        case 11: return 4;
        case 12: return 4;
        default: return 0;
    }
}

__device__ inline const char* swizzleToString_d(uint32_t swz) {
    switch (swz) {
        case 0: return "NONE";
        case 1: return "32B";
        case 2: return "64B";
        case 3: return "128B";
        default: return "?";
    }
}

__device__ inline const char* interleaveToString_d(uint32_t v) {
    switch (v) {
        case 0: return "NONE";
        case 1: return "16B";
        case 2: return "32B";
        default: return "?";
    }
}

__device__ inline const char* l2promoToString_d(uint32_t v) {
    switch (v) {
        case 0: return "NONE";
        case 1: return "64B";
        case 2: return "128B";
        case 3: return "256B";
        default: return "?";
    }
}

/**
 * Device-side TMA descriptor printer.
 * Decodes and prints all fields of a 128-byte CUtensorMap descriptor.
 *
 * Must be called from a single thread (e.g., threadIdx.x == 0).
 * The descriptor can reside in global memory, shared memory, or
 * be passed via __grid_constant__.
 *
 * @param tensorMap  Pointer to descriptor (device-accessible)
 * @param label      Optional label string (device-side string literal)
 */
__device__ inline void cuTensorMapPrint_device(const CUtensorMap* tensorMap,
                                               const char* label = nullptr) {
    const uint8_t* desc = reinterpret_cast<const uint8_t*>(tensorMap);

    if (label)
        printf("=== TMA Descriptor [device]: %s ===\n", label);
    else
        printf("=== TMA Descriptor [device] ===\n");

    // --- Raw hex dump (first 76 meaningful bytes) ---
    printf("Raw bytes:\n");
    for (int i = 0; i < 80; i++) {
        if (i % 16 == 0) printf("  [%3d] ", i);
        printf("%02X ", desc[i]);
        if (i % 16 == 15) printf("\n");
    }
    printf("\n");

    // --- Global address (bytes 0-7) ---
    uint64_t globalAddr;
    memcpy(&globalAddr, &desc[0], 8);
    printf("Global Address:     0x%016llX\n", (unsigned long long)globalAddr);

    // --- Control word (bytes 8-11) ---
    uint32_t control;
    memcpy(&control, &desc[8], 4);

    uint32_t tensorType = (control >> 0) & 0xF;
    uint32_t ndim       = ((control >> 4) & 0x7) + 1;
    uint32_t dtypeCode  = (control >> 7) & 0xF;
    uint32_t interleave = (control >> 11) & 0x3;
    uint32_t swizzle    = (control >> 13) & 0x3;
    uint32_t oobFill    = (control >> 15) & 0x1;
    uint32_t l2Promo    = (control >> 17) & 0x3;
    uint32_t bit21      = (control >> 21) & 0x1;
    uint32_t elemSize   = dtypeCodeToElemSize_d(dtypeCode);

    printf("\nControl Word:       0x%08X\n", control);
    printf("  Type:   %u (0=tiled)  Rank: %u\n", tensorType, ndim);
    printf("  Dtype:  %s (code=%u, %uB)\n", dtypeCodeToString_d(dtypeCode), dtypeCode, elemSize);
    printf("  Interleave: %s  Swizzle: %s\n", interleaveToString_d(interleave), swizzleToString_d(swizzle));
    printf("  OOB Fill: %s  L2 Promo: %s\n",
           oobFill ? "NaN" : "ZERO", l2promoToString_d(l2Promo));
    printf("  bit[21]: %u → OOB protection %s\n", bit21, bit21 ? "DISABLED" : "ENABLED");

    // --- Global strides (bytes 12-29) ---
    printf("\nStrides (bytes):\n");
    for (uint32_t i = 0; i < 4 && i < ndim - 1; i++) {
        uint32_t lower32;
        memcpy(&lower32, &desc[12 + i * 4], 4);
        uint8_t upperByte = (i < 2) ? desc[28] : desc[29];
        uint8_t upper4 = (i % 2 == 0) ? (upperByte & 0xF) : ((upperByte >> 4) & 0xF);
        uint64_t stride_bytes = (((uint64_t)upper4 << 32) | (uint64_t)lower32) << 4;
        printf("  stride[%u]: %llu\n", i, (unsigned long long)stride_bytes);
    }

    // --- Global dimensions (bytes 32-51) ---
    printf("\nDimensions:\n");
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dimM1;
        memcpy(&dimM1, &desc[32 + i * 4], 4);
        printf("  dim[%u]: %u\n", i, dimM1 + 1);
    }

    // --- Element strides (bytes 52-53) ---
    uint32_t esBits = (uint32_t)desc[52] | ((uint32_t)desc[53] << 8);
    printf("\nElement Strides:\n");
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t es = ((esBits >> (i * 3)) & 0x7) + 1;
        printf("  elemStride[%u]: %u\n", i, es);
    }

    // --- Box dimensions (bytes 55-59) ---
    printf("\nBox Dimensions:\n");
    for (uint32_t i = 0; i < ndim; i++) {
        printf("  box[%u]: %u\n", i, (uint32_t)desc[55 + i] + 1);
    }

    // --- Total box bytes (bytes 64-67) ---
    uint32_t totalBoxBytes;
    memcpy(&totalBoxBytes, &desc[64], 4);
    printf("\nBox Total Bytes: %u\n", totalBoxBytes);

    // --- Swizzle stride (bytes 72-75) ---
    uint32_t swzStride;
    memcpy(&swzStride, &desc[72], 4);
    printf("Swizzle Stride:  %u\n", swzStride);

    // --- bit21 consistency check ---
    uint64_t totalBytes = (uint64_t)elemSize;
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dimM1;
        memcpy(&dimM1, &desc[32 + i * 4], 4);
        totalBytes *= (uint64_t)(dimM1 + 1);
    }
    uint32_t expectedBit21 = (totalBytes >= 131072) ? 1 : 0;
    printf("\nDerived:\n");
    printf("  Total tensor: %llu bytes (%s 128KB)\n",
           (unsigned long long)totalBytes, totalBytes >= 131072 ? ">=" : "<");
    if (expectedBit21 != bit21) {
        printf("  WARNING: bit21 INCONSISTENT! expected=%u actual=%u\n", expectedBit21, bit21);
        printf("           (tensormap.replace modified dims without updating bit21)\n");
    } else {
        printf("  bit21 consistent ✓\n");
    }
    printf("\n");
}

#endif // __CUDACC__

#endif // CU_TENSOR_MAP_ENCODE_TILED_IMPL_H
