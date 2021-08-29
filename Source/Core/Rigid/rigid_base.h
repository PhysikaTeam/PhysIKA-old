#pragma once
#include <iostream>
#include "Core/Platform.h"

namespace PhysIKA {

template <typename Scalar, int Dim>
class Rigid
{
public:
    COMM_FUNC Rigid(){};
    COMM_FUNC ~Rigid(){};
};

}  //end of namespace PhysIKA
