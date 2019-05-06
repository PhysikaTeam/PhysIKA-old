#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_SOLVER_SOLVERBASE_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_SOLVER_SOLVERBASE_H_
#include "Physika_Surface_Fuild/Surface_Model/simulator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/indexonvertex.h"
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/indexofmesh.h"
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/cvfem.h"
#include "Physika_Surface_Fuild/Surface_Smooth/smooth.h"
#include "Physika_Surface_Fuild/Surface_Utilities/boundrecorder.h"
#include "Physika_Surface_Fuild/Surface_Utilities/windowstimer.h"
#include <vector>
#include <queue>
#include <deque>
#include "Physika_Surface_Fuild/Surface_Model/swe.cuh"
namespace Physika{
class SolverBase{
public:
    Simulator sim;
    SolverBase();
    ~SolverBase();
    virtual void init(int argc, char** argv);
    virtual void run();
    virtual void run_cuda (int frame);
	  virtual void output_obj_cuda(int frame);
    virtual void set_initial_constants();
	  virtual void generate_origin();
	  virtual void generate_mesh();
    virtual void calculate_tensor();
    virtual void set_initial_conditions();
    virtual void output_obj();
}
