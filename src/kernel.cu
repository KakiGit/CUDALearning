
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "functions.h"
#include <stdio.h>


int main()
{

    cudaError_t cudaStatus;

    //addWithCuda();
    //matrixMulWithCuda();
    //runGraphs();
    //runCooperativeGroups();
    //cudaImageConvolution();
    cudaTrainNeuralNetwork();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}