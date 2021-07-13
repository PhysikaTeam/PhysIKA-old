#pragma once
#include "Core/Platform.h"

namespace PhysIKA {
template <typename TDataType>
class TPair
{
public:
    typedef typename TDataType::Coord Coord;

    COMM_FUNC TPair(){};
    COMM_FUNC TPair(int id, Coord p)
    {
        index = id;
        pos   = p;
    }

    int   index;
    Coord pos;
};

}  // namespace PhysIKA