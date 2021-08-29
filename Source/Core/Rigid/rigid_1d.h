#pragma once
#include <iostream>
#include "rigid_base.h"

namespace PhysIKA {
template <typename Scalar>
class Rigid<Scalar, 1>
{
public:
    COMM_FUNC Rigid(){};
    COMM_FUNC ~Rigid(){};

private:
};

}  //end of namespace PhysIKA
