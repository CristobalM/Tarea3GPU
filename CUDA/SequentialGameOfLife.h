//
// Created by cristobal, 2018
//

#ifndef TAREA3GPU_SEQUENTIALGAMEOFLIFE_H
#define TAREA3GPU_SEQUENTIALGAMEOFLIFE_H

#include <memory>

#include "defs.h"

void simulationStepSequential(uchar* data, uchar* data_work, uint sizeX, uint sizeY, uint size);
std::unique_ptr<SimulationResult> sequentialGameOfLifeRun(uchar* initialPositions, Params& p);

#endif //TAREA3GPU_SEQUENTIALGAMEOFLIFE_H
