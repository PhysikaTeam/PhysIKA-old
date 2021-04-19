#ifndef MATH_H
#define MATH_H


#include <string>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

namespace TCDSM{namespace Util{

float enforceRange(const float& x,const int& MaxValue)
{

    return std::min(std::max(x,0.0f),(float)(MaxValue-1));
}

}}
#endif // MATH_H
