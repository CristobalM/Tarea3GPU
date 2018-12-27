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



#endif //TAREA3GPU_GAMEOFLIFE_H
