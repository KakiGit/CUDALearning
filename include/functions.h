#pragma once
#include "cuda_runtime.h"

cudaError_t addWithCuda();
cudaError_t matrixMulWithCuda();
cudaError_t runGraphs();
cudaError_t runCooperativeGroups();
