#include <iostream>
#include "GameOfLife.h"
#include "defs.h"
#include <algorithm>
#include <chrono>
#include <thread>

#include "SequentialGameOfLife.h"

void printTable(uchar* table, uint sizeX, uint sizeY);
uint runExperimentCuda(uint sizeX, uint sizeY);
void runExperimentsCuda_1(uint repeat = 5);
std::unique_ptr<uchar> createInitConfig(uint sizeX, uint sizeY, uint discount = 2);
void runExperimentsSeq_1(uint repeat);
void runExperiments_1(uint repeat, uint (*runAExperiment)(uint, uint));
uint runExperimentSequential(uint sizeX, uint sizeY);
uint runExperiment(uint sizeX, uint sizeY, std::unique_ptr<SimulationResult> (*simulationStarter)(uchar*, Params&));

int main() {
  runExperimentsCuda_1(1);
  std::cout << "Sequential experiments" << std::endl;
  runExperimentsSeq_1(1);
  return 0;
}


uint runExperiment(const uint sizeX, const uint sizeY, std::unique_ptr<SimulationResult> (*simulationStarter)(uchar*, Params&)){
  const auto size = sizeX * sizeY;

  auto initial_positions = createInitConfig(sizeX, sizeY);

  const uint threadNum = 1024;
  const uint blockNum = std::min<uint>(32768, std::max<uint>(size/threadNum, 1));
  const uint iterationsCount = 10000;

  Params params = {
          .sizeX = sizeX,
          .sizeY = sizeY,
          .blockNum = blockNum,
          .threadNum = threadNum,
          .iterationsCount = iterationsCount,
          .iterationStartMeasuring = iterationsCount/10
  };

  auto simulationResult = simulationStarter(initial_positions.get(), params);
  auto timeElapsed = simulationResult->timeInMs;


  return timeElapsed;
}

uint runExperimentCuda(const uint sizeX, const uint sizeY) {
  return runExperiment(sizeX, sizeY, startSimulation);
}
uint runExperimentSequential(const uint sizeX, const uint sizeY) {
  return runExperiment(sizeX, sizeY, sequentialGameOfLifeRun);
}

void runExperiments_1(uint repeat, uint (*runAExperiment)(uint, uint)){
  std::cout << "Width x Height,Elapsed Time(ms)" << std::endl;
  uint sumTimes = 0;
  for(uint s = 2; s <= 10; s++){
    auto oneDimSize = (uint)1 << s;
    for(uint k = 0; k < repeat; k++)
      sumTimes += runAExperiment(oneDimSize, oneDimSize);

    auto timeElapsedMean = sumTimes / repeat;
    std::cout << oneDimSize << "x" << oneDimSize << "," << timeElapsedMean << std::endl;
  }
}

void runExperimentsCuda_1(uint repeat){
  runExperiments_1(repeat, runExperimentCuda);
};
void runExperimentsSeq_1(uint repeat){
  runExperiments_1(repeat, runExperimentSequential);
};

std::unique_ptr<uchar> createInitConfig(uint sizeX, uint sizeY, uint discount){
  uint size = sizeX * sizeY;

  std::unique_ptr<uchar> initial_positions(new uchar[size]);

  for(uint i = 0; i < sizeY; i++){
    for(uint j = 0; j < sizeX; j++){
      initial_positions.get()[i * sizeX + j] = (uchar)(( i > discount-1 && j > discount-1 && i <sizeY-(discount) && j < sizeX -(discount) ) ? 1 : 0);
    }
  }

  return initial_positions;
}
// debug
void printTable(uchar* table, uint sizeX, uint sizeY){
  for(uint i = 0; i < sizeY; i++){
    for(uint j = 0; j < sizeX; j++){
      auto value = (uint)table[i * sizeX + j];
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }
}