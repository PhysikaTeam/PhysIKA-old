#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_FUILD_MODEL_SIMULATORI_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_FUILD_MODEL_SIMULATORI_H_
#include "SimulationBase.h"
namespace Physika{
class SimulatorI:public SimulationBase{
public:
SimulatorI();
~SimulatorI();
void init(int argc, char** argv);
void run();
void run_cuda (int frame);
void output_obj_cuda(int frame);
void post_data ();
void clear();
}
}
#endif;
