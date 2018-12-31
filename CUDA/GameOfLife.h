//
// Created by cristobal, 2018
//

#ifndef TAREA3GPU_GAMEOFLIFE_H
#define TAREA3GPU_GAMEOFLIFE_H

#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "defs.h"


__host__ std::unique_ptr<SimulationResult> startSimulation(uchar* initial_positions, Params &p);
__host__ std::unique_ptr<SimulationResult> startSimulationBits(uchar* initial_positions, Params &p);
__host__ std::unique_ptr<uchar> convertToBitsMode(uchar* positions, uint sizeX, uint sizeY, uint worldSizeBits);
__host__ std::unique_ptr<uchar> convertFromBitsMode(uchar* positions, uint sizeX, uint sizeY, uint worldSizeBits);
__host__ std::unique_ptr<SimulationResult> startSimulationGeneric(uchar* initial_positions, Params &p,
                                                                  void (*kernel)(uchar*, uchar*, uint, uint, uint));
__host__ std::unique_ptr<SimulationResult> startSimulationIfElse(uchar* initial_positions, Params &p);


#endif //TAREA3GPU_GAMEOFLIFE_H
