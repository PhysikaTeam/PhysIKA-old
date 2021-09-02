/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: parameters setting for mass spring method.
 * @version    : 1.0
 */
#ifndef _PARA_
#define _PARA_

// #include "head.h"
// hardcode the test parameter for mass spring testing demo

#include <string>
/**
 * parameters for mass spring class
 *
 */
class para
{
public:
    // hardcoded parameter for object_creator
    static std::string out_dir_object_creator;
    static std::string object_name;
    static double      dmetric;
    static int         lc, wc, hc, dim_simplex;

    // hardcoded parameter for simulator
    static std::string out_dir_simulator, simulation_type, newton_fastMS;
    static double      dt;
    static double      gravity, density, stiffness;
    static int         frame;
    static bool        line_search;
    static double      weight_line_search;
    static std::string input_object, input_constraint;
    static std::string force_function;
    static double      intensity;
    // hardcoded parameter for openmp num of threads
    static int  num_threads;
    static bool coll_z;
};

#endif
