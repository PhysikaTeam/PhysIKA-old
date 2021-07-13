#pragma once
#include "PointSet.h"

namespace PhysIKA {
template <typename Coord>
class UnstructuredPointSet : public PointSet<Coord>
{
public:
    UnstructuredPointSet();
    ~UnstructuredPointSet();

private:
};

}  // namespace PhysIKA
