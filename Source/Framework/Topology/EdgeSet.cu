#include "EdgeSet.h"
#include "IO/Smesh_IO/smesh.h"
#include <vector>
#include <Core/Utility.h>

namespace PhysIKA {
template <typename TDataType>
EdgeSet<TDataType>::EdgeSet()
{
}

template <typename TDataType>
EdgeSet<TDataType>::~EdgeSet()
{
}

template <typename TDataType>
EdgeSet<TDataType>::EdgeSet(EdgeSet<TDataType>& edgeset)
{
    m_edgeNeighbors = edgeset.m_edgeNeighbors;
    m_edges         = edgeset.m_edges;
}
template <typename TDataType>
EdgeSet<TDataType>& EdgeSet<TDataType>::operator=(EdgeSet<TDataType>& edgeset)
{
    m_edgeNeighbors = edgeset.m_edgeNeighbors;
    m_edges         = edgeset.m_edges;
    return *this;
}

__global__ void K_updatePointNeighborsInEdges(NeighborList<int> nbl, DeviceArray<TopologyModule::Edge> edges)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= nbl.size())
        return;

    int neibor_num = 0;

    for (int i = 0; i < edges.size(); ++i)
    {
        if (edges[i][0] == pId)
        {
            nbl.setElement(pId, neibor_num++, edges[i][1]);
        }
        if (edges[i][1] == pId)
        {
            nbl.setElement(pId, neibor_num++, edges[i][0]);
        }
        nbl.setNeighborSize(pId, neibor_num);
    }
}

template <typename TDataType>
void EdgeSet<TDataType>::updatePointNeighbors()
{
    if (this->m_coords.isEmpty())
        return;

    auto nbr   = m_edgeNeighbors.getValue();
    uint pDims = cudaGridSize(nbr.size(), BLOCK_SIZE);
    K_updatePointNeighborsInEdges<<<pDims, BLOCK_SIZE>>>(nbr, m_edges);
    cuSynchronize();
}

template <typename TDataType>
void EdgeSet<TDataType>::loadSmeshFile(std::string filename)
{
    Smesh mesh;
    mesh.loadFile(filename);
    std::vector<Coord> vert_list;
    vert_list.resize(mesh.m_points.size());
    for (int i = 0; i < vert_list.size(); ++i)
    {
        vert_list[i][0] = mesh.m_points[i][0];
        vert_list[i][1] = mesh.m_points[i][1];
        vert_list[i][2] = mesh.m_points[i][2];
    }
    this->setPoints(vert_list);

    m_edges.resize(mesh.m_edges.size());
    Function1Pt::copy(m_edges, mesh.m_edges);

    m_edgeNeighbors.setElementCount(vert_list.size(), 4);
    this->updatePointNeighbors();
}

#ifdef PRECISION_FLOAT
template class EdgeSet<DataType3f>;
#else
template class EdgeSet<DataType3d>;
#endif
}  // namespace PhysIKA