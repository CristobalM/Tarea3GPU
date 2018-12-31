#include <iostream>
#include "GameOfLife.h"
#include "defs.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

#include "SequentialGameOfLife.h"


static const uint DEFAULT_THREADS_PER_BLOCK = 1024;

std::unique_ptr<uchar> createInitConfig(uint sizeX, uint sizeY, uint discount = 2);

uint runExperimentCuda(uint sizeX, uint sizeY, uint threadsPerBlock);
uint runExperimentCudaIfElse(uint sizeX, uint , uint threadsPerBlock=DEFAULT_THREADS_PER_BLOCK);
uint runExperimentBitCuda(uint sizeX, uint sizeY, uint threadsPerBlock=DEFAULT_THREADS_PER_BLOCK);
uint runExperimentSequential(uint sizeX, uint sizeY, uint threadsPerBlock=DEFAULT_THREADS_PER_BLOCK);

void runExperimentsCuda_1(uint repeat, uint threadsPerBlock=DEFAULT_THREADS_PER_BLOCK);
void runExperimentsCudaIfElse_1(uint repeat, uint threadsPerBlock=DEFAULT_THREADS_PER_BLOCK);
void runExperimentsSeq_1(uint repeat, uint threadsPerBlock=DEFAULT_THREADS_PER_BLOCK);
void runExperimentsBitCuda_1(uint repeat, uint threadsPerBlock=DEFAULT_THREADS_PER_BLOCK);

void runExperiments_1(uint repeat, uint (*runAExperiment)(uint, uint, uint), uint maxPowerOf2=10,
        uint threadsPerBlock=DEFAULT_THREADS_PER_BLOCK);
uint runExperiment(uint sizeX, uint sizeY, std::unique_ptr<SimulationResult> (*simulationStarter)(uchar*, Params&),
        uint threadsPerBlock);

void blockSizeExperimentsMain();

//debug
void testBitConversion();
void printTable(uchar* table, uint sizeX, uint sizeY);
//</debug

int main() {

  blockSizeExperimentsMain();

  //std::cout << "Cuda experiments 1D Array If..Else" << std::endl;
  //runExperimentsCudaIfElse_1(1);
  std::cout << "Cuda experiments 1D Array" << std::endl;
  runExperimentsCuda_1(1);

  std::cout << "Cuda experiments 1D Array Bit Implementation" << std::endl;
  runExperimentsBitCuda_1(1);

  std::cout << std::endl << "Sequential experiments" << std::endl;
  runExperimentsSeq_1(1);

  std::cout << std::endl << std::endl;

  //testBitConversion();

  return 0;
}

void blockSizeExperimentsMain(){
  std::cout << "Block sizes experiment" << std::endl;
  std::vector<uint> blockSizes{32, 320, 512, 800, 1024};
  for (auto blockSize : blockSizes) {
    std::cout << "Block size = " << blockSize << std::endl;
    runExperimentsCuda_1(1, blockSize);
    std::cout << "Block size = " << blockSize +1 << std::endl;
    runExperimentsCuda_1(1, blockSize +1);
  }
}


uint runExperiment(const uint sizeX, const uint sizeY,
        std::unique_ptr<SimulationResult> (*simulationStarter)(uchar*, Params&), const uint threadsPerBlock){
  const auto size = sizeX * sizeY;

  auto initial_positions = createInitConfig(sizeX, sizeY);

  const uint threadNum = threadsPerBlock;
  const uint blockNum = std::min<uint>(32768, std::max<uint>(size/threadNum, 1));
  const uint iterationsCount = 10000;

  Params params = {
          .sizeX = sizeX,
          .sizeY = sizeY,
          .blockNum = blockNum,
          .threadNum = threadNum,
          .iterationsCount = iterationsCount,
          .iterationStartMeasuring = iterationsCount/10,
          .maxBytesPerThread = 4
  };

  auto simulationResult = simulationStarter(initial_positions.get(), params);
  auto timeElapsed = simulationResult->timeInMs;
  return timeElapsed;
}

uint runExperimentCuda(const uint sizeX, const uint sizeY, const uint threadsPerBlock) {
  return runExperiment(sizeX, sizeY, startSimulation, threadsPerBlock);
}

uint runExperimentCudaIfElse(const uint sizeX, const uint sizeY, const uint threadsPerBlock){
  return runExperiment(sizeX, sizeY, startSimulationIfElse, threadsPerBlock);
}


uint runExperimentSequential(const uint sizeX, const uint sizeY, const uint threadsPerBlock) {
  return runExperiment(sizeX, sizeY, sequentialGameOfLifeRun, threadsPerBlock);
}

uint runExperimentBitCuda(const uint sizeX, const uint sizeY, const uint threadsPerBlock){
  return runExperiment(sizeX, sizeY, startSimulationBits, threadsPerBlock);
}

void runExperiments_1(const uint repeat, uint (*runAExperiment)(uint, uint, uint), uint maxPowerOf2, uint threadsPerBlock){
  std::cout << "Width x Height,Elapsed Time(ms)" << std::endl;
  uint sumTimes = 0;
  for(uint s = 2; s <= maxPowerOf2; s++){
    auto oneDimSize = (uint)1 << s;
    for(uint k = 0; k < repeat; k++)
      sumTimes += runAExperiment(oneDimSize, oneDimSize, threadsPerBlock);

    auto timeElapsedMean = sumTimes / repeat;
    std::cout << oneDimSize << "x" << oneDimSize << "," << timeElapsedMean << std::endl;
  }
}

void runExperimentsCuda_1(const uint repeat, const uint threadsPerBlock){
  runExperiments_1(repeat, runExperimentCuda, 13, threadsPerBlock);
};

void runExperimentsCudaIfElse_1(const uint repeat, const uint threadsPerBlock){
  runExperiments_1(repeat, runExperimentCudaIfElse, 13, threadsPerBlock);
}

void runExperimentsSeq_1(const uint repeat, const uint threadsPerBlock){
  runExperiments_1(repeat, runExperimentSequential, 10, threadsPerBlock);
};

void runExperimentsBitCuda_1(const uint repeat, const uint threadsPerBlock){
  runExperiments_1(repeat, runExperimentBitCuda, 13, threadsPerBlock);
};

std::unique_ptr<uchar> createInitConfig(const uint sizeX, const uint sizeY, const uint discount){
  uint size = sizeX * sizeY;

  std::unique_ptr<uchar> initial_positions(new uchar[size]);

  for(uint i = 0; i < sizeY; i++){
    for(uint j = 0; j < sizeX; j++){
      uchar result;
      if(discount == 0)
        result = 1;
      else
        result = (uchar)(( i > discount-1 && j > discount-1 && i <sizeY-(discount) && j < sizeX -(discount) ) ? 1 : 0);
      initial_positions.get()[i * sizeX + j] = result;
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

void testBitConversion(){
  auto init_config = createInitConfig(9, 9, 0);
  for(int j = 0; j < 9; j++){
    init_config.get()[9 + j] = 0;
  }
  init_config.get()[18 + 8] = 0;
  printTable(init_config.get(), 9, 9);
  auto binit = convertToBitsMode(init_config.get(), 9, 9, 11);
  std::cout << "bits" << std::endl;
  //binit.get()[0] = 45;
  printTable(binit.get(), 2, 9);
  auto bback = convertFromBitsMode(binit.get(), 9, 9, 11);
  std::cout << "back" << std::endl;
  printTable(bback.get(), 9, 9);
}

