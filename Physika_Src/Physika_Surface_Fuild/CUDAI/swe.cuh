#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/mymesh.h"
#include "Surface_Fuild_Model/SimulatorI.h"
#include "Physika_Surface_Fuild/fluid_defs.h"
//#include "fluid_system_host.cuh"
//#include "fluid_system_kern.cuh"

void cudaInit ( int argc, char **argv );
void cudaExit ( int argc, char **argv );
void cudaFree ();
void setupCuda(Simulator const &sim);
void copyToCuda(Simulator const &sim);
void copyFromCuda(Simulator const &sim);
void processCuda(Simulator &sim);

void updateSimParams ();

#ifdef HYBRID


//To SPH
int AddParticlesFromSWE(int currentPnum);
//From SPH
void CollectLabelParticlesSWE();
void ShowBug();
#endif
