//
// Created by cristobal, 2018
//

#include "GameOfLife.h"

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include "defs.h"
#include <chrono>



__global__ void simulationStep(uchar* data, uchar* data_work, uint worldWidth, uint worldHeight, uint worldSize){
  uint bId = blockIdx.x;
  uint bWidth = blockDim.x;
  uint xRel = threadIdx.x;
  uint bNum = gridDim.x;

  for(uint cId = bId * bWidth + xRel; cId < worldSize; cId += bWidth * bNum){
    uint x = cId % worldWidth;
    uint y = cId - x;
    uint xLeft = (x - 1 + worldWidth) % worldWidth;
    uint xRight = (x + 1) % worldWidth;
    uint yUp = (y - worldWidth + worldSize) % worldSize;
    uint yDown = (y + worldWidth) % worldSize;
    uint aliveCount =
            data[y + xLeft] +
            data[yUp + xLeft] + data[yUp + x] + data[yUp + xRight] +
            data[y + xRight] +
            data[yDown + xRight] + data[yDown + x] + data[yDown + xLeft];

    uchar alive = data[y + x];
    //data_work[y + x] = (uchar)((aliveCount == 3 || (aliveCount == 2 && alive)) ? 1 : 0); //game of life
    int result = ((aliveCount == 3 || (aliveCount == 2 && alive) || (aliveCount == 6 && !alive)) ? 1 : 0); // highlife
    data_work[y + x] = (uchar)result;
  }
}


__host__ std::unique_ptr<SimulationResult> startSimulation(uchar* initial_positions, Params &p) {
  const uint worldSize = p.sizeX * p.sizeY;

  std::unique_ptr<uchar> h_data(new uchar[worldSize]);
  uchar *d_data, *d_data_work;

  const size_t worldSizeInBytes = worldSize * sizeof(uchar);


  gpuErrchk(cudaMalloc((void **)&d_data, worldSizeInBytes));
  gpuErrchk(cudaMalloc((void **)&d_data_work, worldSizeInBytes));
  gpuErrchk(cudaMemcpy(d_data_work, initial_positions, worldSizeInBytes, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_data, d_data_work, worldSizeInBytes, cudaMemcpyDeviceToDevice));

  std::chrono::time_point<std::chrono::high_resolution_clock> timeStart;
  uint totalTime = 0;

  for(int i = 0; i < p.iterationsCount; i++){
    if(i == p.iterationStartMeasuring){
      timeStart = std::chrono::high_resolution_clock::now();
    }
    simulationStep<<<p.blockNum, p.threadNum>>>(d_data, d_data_work, p.sizeX, p.sizeY, worldSize);
    cudaDeviceSynchronize();


    std::swap(d_data, d_data_work);
  }

  totalTime = (uint)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timeStart).count();

  if(p.iterationsCount == 0)
    totalTime = 0;

  gpuErrchk(cudaMemcpy(h_data.get(), d_data, worldSizeInBytes, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_data_work));

  //return h_data;

  SimulationResult result = {
          .finalConfiguration = std::move(h_data),
          .finalConfigurationSize = worldSize,
          .timeInMs = totalTime,
  };

  return std::unique_ptr<SimulationResult>(new SimulationResult(std::move(result)));
}