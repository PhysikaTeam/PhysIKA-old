/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: parameters setting for mass spring method.
 * @version    : 1.0
 */
#include "para.h"

using namespace std;

// hardcoded parameter for object_creator
string para::out_dir_object_creator = "./input_data/cloth";
string para::object_name            = "cloth";
double para::dmetric                = 0.01;
int    para::lc = 60, para::wc = 60, para::hc = 60, para::dim_simplex = 2;

/*
// hardcoded parameter for simulator 
const string para::out_dir_simulator="./example/cloth/newton";
const string para::simulation_type="dynamic",para::newton_fastMS="newton";
const double para::dt=0.01;
const double para::gravity=9.8,para::density=10,para::stiffness=6000;
const int para::frame=500;
const int para::line_search=1;
const double para::weight_line_search=1e-5;
const string para::input_object="./input_data/cloth/cloth.vtk",para::input_constraint="./input_data/cloth/cloth.csv";
const string para::force_function="gravity";
*/

// hardcoded parameter for simulator
string para::out_dir_simulator = "./example/cloth/newton";
string para::simulation_type = "dynamic", para::newton_fastMS = "newton";
double para::dt      = 0.01;
double para::gravity = 9.8, para::density = 10, para::stiffness = 8000;
int    para::frame              = 500;
bool   para::line_search        = 1;
double para::weight_line_search = 1e-5;
string para::input_object       = "./input_data/cloth/cloth_fine.vtk";
string para::input_constraint   = "";
string para::force_function     = "gravity";
double para::intensity          = 10000;

// hardcoded parameter for openmp num of threads
int  para::num_threads = 6;
bool para::coll_z      = false;
