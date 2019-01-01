//
// Created by cristobal, 2018
//

#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include "GameOfLife.h"
#include "defs.h"


static const uint DEFAULT_THREADS_PER_BLOCK = 32;

std::unique_ptr<uchar> createInitConfig(uint sizeX, uint sizeY, uint discount = 2);

uint runExperiment(uint sizeX, uint sizeY, std::unique_ptr<SimulationResult> (*simulationStarter)(uchar*, Params&),
                   uint threadsPerBlock, uint discount);


//debug
void printTable(uchar* table, uint sizeX, uint sizeY);
//</debug

int main() {

  std::cout << "Simulación en GPU/CUDA con implementacion de bits" << std::endl << std::endl;
  //runExperimentsBitCuda_1(1);
  for(uint i = 1; i < 5; i++){
    std::cout << "Experimento " << i << std::endl << std::endl;
    runExperiment(10u, 10u, startSimulation, DEFAULT_THREADS_PER_BLOCK, i);
  }

  return 0;
}



uint runExperiment(const uint sizeX, const uint sizeY,
                   std::unique_ptr<SimulationResult> (*simulationStarter)(uchar*, Params&), const uint threadsPerBlock, uint discount){
  const auto size = sizeX * sizeY;

  auto initial_positions = createInitConfig(sizeX, sizeY, discount);

  const uint threadNum = threadsPerBlock;
  const uint blockNum = std::min<uint>(32768, std::max<uint>(size/threadNum, 1));
  const uint iterationsCount = 1000;

  Params params = {
          .sizeX = sizeX,
          .sizeY = sizeY,
          .blockNum = blockNum,
          .threadNum = threadNum,
          .iterationsCount = iterationsCount,
          .iterationStartMeasuring = iterationsCount/10,
          .maxBytesPerThread = 4
  };

  std::cout << "Configuración Inicial:" << std::endl;

  printTable(initial_positions.get(), sizeX, sizeY);


  auto simulationResult = simulationStarter(initial_positions.get(), params);
  auto timeElapsed = simulationResult->timeInMs;
  auto final_positions = simulationResult->finalConfiguration.get();

  std::cout << std::endl << "Configuración Final despues de " << iterationsCount << " iteraciones:" << std::endl << std::endl;
  printTable(final_positions, sizeX, sizeY);

  return timeElapsed;
}



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
