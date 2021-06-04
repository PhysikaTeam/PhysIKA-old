/**
 * @author     : ZHAO CHONGYAO (cyzhao@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: timing util source for physika library
 * @version    : 2.0.1
 */
#include "config.h"
#include <chrono>

using namespace std;
using namespace chrono;

namespace PhysIKA{
list<system_clock::time_point> TIMING::starts_;
}
