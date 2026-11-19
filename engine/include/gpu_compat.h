/**
 * gpu_compat.h - CUDA and AMD HIP portability layer for model_cuda.cu.
 *
 * When VITNA_HIP is defined (built with hipcc via `make VITNA_HIP=1` or
 * `cmake -DVITNA_HIP=ON`), maps the CUDA runtime types, constants, functions,
 * and 32-lane warp shuffle intrinsics used by model_cuda.cu onto their ROCm
 * HIP equivalents, and disables NVIDIA-only nvcuda::wmma tensor-core paths.
 *
 * When VITNA_HIP is not defined, includes CUDA headers directly with zero
 * macro remapping so NVIDIA builds remain unchanged.
 */

#ifndef VITNA_GPU_COMPAT_H
#define VITNA_GPU_COMPAT_H

#if defined(VITNA_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define VITNA_GPU_BACKEND_NAME "hip"
#define VITNA_HAS_WMMA 0

/* Runtime types */
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemcpyKind hipMemcpyKind

/* Status codes */
#define cudaSuccess hipSuccess
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaErrorNoKernelImageForDevice hipErrorNoBinaryForGpu
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorUnsupportedPtxVersion hipErrorInvalidImage

/* Flags and attributes */
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaDevAttrHostRegisterReadOnlySupported hipDeviceAttributeHostRegisterSupported
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

/* Device, memory, stream, event, and graph APIs */
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaMemGetInfo hipMemGetInfo
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventQuery hipEventQuery
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaFuncSetAttribute hipFuncSetAttribute

/* hipGraphInstantiate takes (exec, graph, errNode, logBuf, bufSize) across ROCm versions. */
static inline hipError_t vitna_hip_graph_instantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    (void)flags;
    return hipGraphInstantiate(exec, graph, NULL, NULL, 0);
}
#define cudaGraphInstantiate vitna_hip_graph_instantiate

/* 32-lane sub-warp shuffles matching CUDA's 32-thread warp reduction tree. */
#define __shfl_xor_sync(mask, val, lane_mask) __shfl_xor((val), (lane_mask), 32)
#define __shfl_sync(mask, val, src_lane) __shfl((val), (src_lane), 32)

/* Portable IEEE-754 float16 bit-cast widening without PTX inline asm. */
__device__ __forceinline__ float vitna_half_bits_to_float(unsigned short h) {
    return __half2float(__ushort_as_half(h));
}

#else /* NVIDIA CUDA */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define VITNA_GPU_BACKEND_NAME "cuda"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define VITNA_HAS_WMMA 1
#else
#define VITNA_HAS_WMMA 0
#endif

__device__ __forceinline__ float vitna_half_bits_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

#endif /* VITNA_HIP */

#endif /* VITNA_GPU_COMPAT_H */
