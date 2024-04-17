#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "nccl.h"
#include "timer.h"

#define CUDACHECK(cmd)                                           \
    do                                                           \
    {                                                            \
        cudaError_t err = cmd;                                   \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("Failed: Cuda error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

#define NCCLCHECK(cmd)                                           \
    do                                                           \
    {                                                            \
        ncclResult_t res = cmd;                                  \
        if (res != ncclSuccess)                                  \
        {                                                        \
            printf("Failed, NCCL error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(int argc, char *argv[])
{
    ncclComm_t comms[4];

    // managing 4 devices
    int nDev = 4;
    size_t size = 16 * 1024 * 1024;
    int devs[4] = {0, 1, 2, 3};

    // allocating and initializing device buffers
    half **sendbuff = (half **)malloc(nDev * sizeof(half *));
    half **recvbuff = (half **)malloc(nDev * sizeof(half *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void **)sendbuff + i, size * sizeof(half)));
        CUDACHECK(cudaMalloc((void **)recvbuff + i, size * sizeof(half)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(half)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(half)));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    for (int i = 0; i < 5; i++)
    {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nDev; ++i)
            NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat16, ncclSum,
                                    comms[i], s[i]));
        NCCLCHECK(ncclGroupEnd());
    }
    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }


    timer t;
    for (int i = 0; i < 10; i++)
    {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nDev; ++i)
            NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat16, ncclSum,
                                    comms[i], s[i]));
        NCCLCHECK(ncclGroupEnd());
    }
    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }
    double latency = t.elapsed();


    // free device buffers
    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);

    printf("Success: %lfs, bandwidth: %lfGB/s\n", latency / 10, double(size) * sizeof(half) /1.0E9/latency * 10);
    return 0;
}