//
// Created by cristobal, 2018
//

#include "GameOfLife.h"

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include "defs.h"
#include <chrono>
#include <iostream>

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


__global__ void simulationStepIfElse(uchar* data, uchar* data_work, uint worldWidth, uint worldHeight, uint worldSize){
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
    int result;
    if(aliveCount == 3 || (aliveCount == 2 && alive) || (aliveCount == 6 && !alive)){
      result = 1;
    }
    else{
      result = 0;
    }

    data_work[y + x] = (uchar)result;
  }
}


__global__ void simulationStepBits(uchar* data, uchar* data_work, uint worldWidth, uint worldHeight, uint worldSize,
        uint bytesPerThread){
  uint bId = blockIdx.x;
  uint bWidth = blockDim.x;
  uint xRel = threadIdx.x;
  uint bNum = gridDim.x;

  for(uint cId = (bId * bWidth + xRel) * bytesPerThread; cId < worldSize; cId += bWidth * bNum * bytesPerThread){
    uint x = (cId + worldWidth - 1) % worldWidth;
    uint y = (cId / worldWidth) * worldWidth;
    uint yUp = (y - worldWidth + worldSize) % worldSize;
    uint yDown = (y + worldWidth) % worldSize;

    uint data0 = (uint) data[yUp + x] << 16;
    uint data1 = (uint) data[y + x] << 16;
    uint data2 = (uint) data[yDown + x] << 16;

    x = (x + 1) % worldWidth;
    data0 |= (uint) data[yUp + x] << 8;
    data1 |= (uint) data[y + x] << 8;
    data2 |= (uint) data[yDown + x] << 8;

    for(uint i = 0; i < bytesPerThread; i++){
      uint xCenter  = x;
      x = (x + 1) % worldWidth;
      data0 |= (uint) data[yUp + x];
      data1 |= (uint) data[y + x];
      data2 |= (uint) data[yDown + x];

      uint result = 0;
      for(uint j = 0; j < 8; j++){
        uint aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
        aliveCells >>= 14;
        aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) + ((data0 >> 15) & 0x1u) + ((data2 >> 15) & 0x1u);
        result = (result << 1) | (aliveCells == 3  || (aliveCells == 2 && (data1 & 0x8000u)) ||
                (aliveCells == 6 && !(data1 & 0x8000u)) ? 1u : 0u);

        data0 <<= 1;
        data1 <<= 1;
        data2 <<= 1;
      }
      data_work[y + xCenter] = (uchar)result;
    }
  }
}


__host__ std::unique_ptr<SimulationResult> startSimulation(uchar* initial_positions, Params &p) {
  return startSimulationGeneric(initial_positions, p, simulationStep);
}

__host__ std::unique_ptr<SimulationResult> startSimulationIfElse(uchar* initial_positions, Params &p){
  return startSimulationGeneric(initial_positions, p, simulationStepIfElse);
}

__host__ std::unique_ptr<SimulationResult> startSimulationGeneric(uchar* initial_positions, Params &p,
        void (*kernel)(uchar*, uchar*, uint, uint, uint)) {
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
    if(i == p.iterationStartMeasuring)
      timeStart = std::chrono::high_resolution_clock::now();

    kernel<<<p.blockNum, p.threadNum>>>(d_data, d_data_work, p.sizeX, p.sizeY, worldSize);
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



__host__ std::unique_ptr<SimulationResult> startSimulationBits(uchar* initial_positions, Params &p) {
  const uint worldSize = p.sizeX * p.sizeY;
  const uint worldWidth = p.sizeX / 8 + (p.sizeX % 8 == 0 ? 0 : 1);
  //const uint worldSizeBits = worldSize / 8  + (worldSize % 8 == 0 ? 0 : 1);
  const uint worldSizeBits = worldWidth * p.sizeY;
  const auto bytesPerThread = std::min<uint>(p.maxBytesPerThread, std::max<uint>(p.sizeX / 8, 1));



  std::unique_ptr<uchar> h_data(new uchar[worldSizeBits]);
  uchar *d_data, *d_data_work;

  const size_t worldSizeInBytes = worldSizeBits * sizeof(uchar);

  auto initial_positions_bits = convertToBitsMode(initial_positions, p.sizeX, p.sizeY, worldSizeBits);

  gpuErrchk(cudaMalloc((void **)&d_data, worldSizeInBytes));
  gpuErrchk(cudaMalloc((void **)&d_data_work, worldSizeInBytes));
  gpuErrchk(cudaMemcpy(d_data_work, initial_positions_bits.get(), worldSizeInBytes, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_data, d_data_work, worldSizeInBytes, cudaMemcpyDeviceToDevice));

  std::chrono::time_point<std::chrono::high_resolution_clock> timeStart;
  uint totalTime = 0;

  for(int i = 0; i < p.iterationsCount; i++){
    if(i == p.iterationStartMeasuring)
      timeStart = std::chrono::high_resolution_clock::now();

    simulationStepBits<<<p.blockNum, p.threadNum>>>(d_data, d_data_work, worldWidth, p.sizeY, worldSizeBits, bytesPerThread);
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

  auto final_positions = convertFromBitsMode(h_data.get(), p.sizeX, p.sizeY, worldSizeBits);


  SimulationResult result = {
          .finalConfiguration = std::move(final_positions),
          .finalConfigurationSize = worldSize,
          .timeInMs = totalTime,
  };

  return std::unique_ptr<SimulationResult>(new SimulationResult(std::move(result)));
}


__host__ std::unique_ptr<uchar> convertToBitsMode(uchar* positions, uint sizeX, uint sizeY, uint worldSizeBits){

  std::unique_ptr<uchar> result(new uchar[worldSizeBits]);

  auto sizeXBits = sizeX / 8 + (sizeX % 8 == 0 ? 0 : 1);

  for(auto i = 0u; i < sizeY; i++){
    for(auto j = 0u; j < sizeX; j += 8){
      const auto id = i * sizeX + j;
      uchar resultByte = 0;
      for(auto k = 0; k < 8; k++) {
        const auto currentBitId = id + k;
        resultByte <<= 1;
        resultByte |= positions[currentBitId];
      }
      const auto bitId  = i * sizeXBits + j/8;
      result.get()[bitId] = resultByte;
    }
  }
  return result;
}


__host__ std::unique_ptr<uchar> convertFromBitsMode(uchar* positions, uint sizeX, uint sizeY, uint worldSizeBits) {
  std::unique_ptr<uchar> result(new uchar[sizeX * sizeY]);

  auto sizeXBits = sizeX / 8 + (sizeX % 8 == 0 ? 0 : 1);

  for(auto i = 0u; i < sizeY; i++){
    for(auto j = 0u; j < sizeXBits; j++){
      const auto id = i * sizeXBits + j;
      const auto baseIdResult = i * sizeX + j*8;
      for(auto k = 7; k >= 0; k--){
        const auto currentBit = (uchar)((positions[id] >> k) & 1);
        result.get()[baseIdResult + 7 - k] = currentBit;
      }
    }
  }

  return result;
}