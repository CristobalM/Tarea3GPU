//
// Created by cristobal, 2018
//

#ifndef TAREA3GPU_DEFS_H
#define TAREA3GPU_DEFS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <memory>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

using uchar = unsigned char;
using uint = unsigned int;


struct Params{
  uint sizeX;
  uint sizeY;
  uint blockNum;
  uint threadNum;
  uint iterationsCount;
  uint iterationStartMeasuring;
  uint maxBytesPerThread;
};

struct SimulationResult{
  std::unique_ptr<uchar> finalConfiguration;
  uint finalConfigurationSize;
  uint timeInMs;
};


#endif //TAREA3GPU_DEFS_H
