//
// Created by cristobal, 2018
//

#include <chrono>
#include <cstring>

#include "SequentialGameOfLife.h"


std::unique_ptr<SimulationResult> sequentialGameOfLifeRun(uchar* initialPositions, Params& p){
  const uint worldSize = p.sizeX * p.sizeY;

  std::unique_ptr<uchar> data(new uchar[worldSize]);
  std::unique_ptr<uchar> data_work(new uchar[worldSize]);
  const size_t worldSizeInBytes = worldSize * sizeof(uchar);

  memcpy(data.get(), initialPositions, worldSizeInBytes);
  //memcpy(data_work.get(), initialPositions, worldSizeInBytes);

  std::chrono::time_point<std::chrono::high_resolution_clock> timeStart;
  uint totalTime = 0;

  for(int i = 0; i < p.iterationsCount; i++){
    if(i == p.iterationStartMeasuring){
      timeStart = std::chrono::high_resolution_clock::now();
    }
    simulationStepSequential(data.get(), data_work.get(), p.sizeX, p.sizeY, worldSize);

    std::swap(data, data_work);
  }

  totalTime = (uint)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timeStart).count();


  SimulationResult result = {
          .finalConfiguration = std::move(data),
          .finalConfigurationSize = worldSize,
          .timeInMs = totalTime,
  };

  return std::unique_ptr<SimulationResult>(new SimulationResult(std::move(result)));
}

void simulationStepSequential(uchar* data, uchar* data_work, uint sizeX, uint sizeY, uint size){
  const auto worldWidth = sizeX;
  const auto worldSize = size;
  for(uint i = 0; i < sizeY; i++){
    for(uint j = 0; j < sizeX; j++){
      uint cId = i*sizeX + j;
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

}