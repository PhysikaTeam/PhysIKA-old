/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: timer config define
 * @version    : 1.0
 */
#include "FEMCommonConfig.h"
#include <chrono>

using namespace std;
using namespace chrono;

/**
 * @brief define TIMING starts.
 * 
 */
namespace PhysIKA {
list<system_clock::time_point> TIMING::starts_;
}
