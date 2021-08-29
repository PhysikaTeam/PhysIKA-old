#pragma once
#include <vector>
#include <Core/Vector/vector_3d.h>
#include <Framework/Framework/ModuleTopology.h>

namespace PhysIKA {

class Smesh
{
public:
    void loadFile(std::string filename);

    std::vector<Vector3f>                    m_points;
    std::vector<TopologyModule::Edge>        m_edges;
    std::vector<TopologyModule::Triangle>    m_triangles;
    std::vector<TopologyModule::Quad>        m_quads;
    std::vector<TopologyModule::Tetrahedron> m_tets;
    std::vector<TopologyModule::Hexahedron>  m_hexs;
};

}  // namespace PhysIKA