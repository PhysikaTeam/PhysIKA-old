#pragma once
#include <string>
#include <vector>
#include "Core/Vector.h"

namespace PhysIKA {

typedef FixedVector<int, 3> Face;

class ObjFileLoader
{
public:
    ObjFileLoader(std::string filename);
    ~ObjFileLoader() {}

    bool load(const std::string& filename);

    bool save(const std::string& filename);

    std::vector<Vector3f>& getVertexList();
    std::vector<Face>&     getFaceList();

private:
    std::vector<Vector3f> vertList;
    std::vector<Face>     faceList;
};

}  //end of namespace PhysIKA
