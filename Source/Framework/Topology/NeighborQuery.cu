#include <cuda_runtime.h>
#include "NeighborQuery.h"
#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "Framework/Topology/NeighborList.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Framework/SceneGraph.h"
#include "Core/Utility/Scan.h"

namespace PhysIKA {
__constant__ int offset1[27][3] = { 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, 1, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1, 1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0, -1, 1, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1, 0, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1 };

IMPLEMENT_CLASS_1(NeighborQuery, TDataType)

template <typename TDataType>
NeighborQuery<TDataType>::NeighborQuery()
    : ComputeModule()
    , m_maxNum(0)
{

    Vector3f sceneLow = SceneGraph::getInstance().getLowerBound();
    Vector3f sceneUp  = SceneGraph::getInstance().getUpperBound();

    m_lowBound  = Coord(sceneLow[0], sceneLow[1], sceneLow[2]);
    m_highBound = Coord(sceneUp[0], sceneUp[1], sceneUp[2]);
    this->inRadius()->setValue(Real(0.011));

    m_hash.setSpace(this->inRadius()->getValue(), m_lowBound, m_highBound);

    //		attachField(&m_radius, "Radius", "Radius of the searching area", false);
    //		attachField(&m_position, "position", "Storing the particle positions!", false);
    //		attachField(&m_neighborhood, "ParticleNeighbor", "Storing particle neighbors!", false);
}

template <typename TDataType>
NeighborQuery<TDataType>::NeighborQuery(DeviceArray<Coord>& position)
    : ComputeModule()
{
    Vector3f sceneLow = SceneGraph::getInstance().getLowerBound();
    Vector3f sceneUp  = SceneGraph::getInstance().getUpperBound();

    m_lowBound  = Coord(sceneLow[0], sceneLow[1], sceneLow[2]);
    m_highBound = Coord(sceneUp[0], sceneUp[1], sceneUp[2]);
    this->inRadius()->setValue(Real(0.011));

    this->inPosition()->setElementCount(position.size());
    Function1Pt::copy(this->inPosition()->getValue(), position);

    //		attachField(&m_radius, "Radius", "Radius of the searching area", false);
    //		attachField(&m_position, "position", "Storing the particle positions!", false);
    //		attachField(&m_neighborhood, "ParticleNeighbor", "Storing particle neighbors!", false);
}

template <typename TDataType>
NeighborQuery<TDataType>::~NeighborQuery()
{
    m_hash.release();
}

template <typename TDataType>
NeighborQuery<TDataType>::NeighborQuery(Real s, Coord lo, Coord hi)
    : ComputeModule()
    , m_maxNum(0)
{
    this->inRadius()->setValue(Real(s));

    m_lowBound  = lo;
    m_highBound = hi;

    m_hash.setSpace(this->inRadius()->getValue(), m_lowBound, m_highBound);

    //		attachField(&m_radius, "Radius", "Radius of the searching area", false);
    //		attachField(&m_position, "position", "Storing the particle positions!", false);
    //		attachField(&m_neighborhood, "ParticleNeighbor", "Storing particle neighbors!", false);
}

template <typename TDataType>
bool NeighborQuery<TDataType>::initializeImpl()
{
    if (!this->inPosition()->isEmpty() && this->outNeighborhood()->isEmpty())
    {
        this->outNeighborhood()->setElementCount(this->inPosition()->getElementCount(), m_maxNum);
    }

    if (this->inPosition()->isEmpty() || this->inRadius()->isEmpty())
    {
        std::cout << "Exception: " << std::string("NeighborQuery's fields are not fully initialized!") << "\n";
        return false;
    }

    int pNum = this->inPosition()->getElementCount();

    HostArray<Coord> hostPos;
    hostPos.resize(pNum);

    // 		Function1Pt::copy(hostPos, m_position.getValue());
    //
    // 		m_lowBound = Vector3f(10000000, 10000000, 10000000);
    // 		m_highBound = Vector3f(-10000000, -10000000, -10000000);
    //
    // 		for (int i = 0; i < pNum; i++)
    // 		{
    // 			m_lowBound[0] = min(hostPos[i][0], m_lowBound[0]);
    // 			m_lowBound[1] = min(hostPos[i][1], m_lowBound[1]);
    // 			m_lowBound[2] = min(hostPos[i][2], m_lowBound[2]);
    //
    // 			m_highBound[0] = max(hostPos[i][0], m_highBound[0]);
    // 			m_highBound[1] = max(hostPos[i][1], m_highBound[1]);
    // 			m_highBound[2] = max(hostPos[i][2], m_highBound[2]);
    // 		}

    m_hash.setSpace(this->inRadius()->getValue(), m_lowBound, m_highBound);

    //		m_reduce = Reduction<int>::Create(m_position.getElementCount());
    triangle_first = true;
    compute();

    return true;
}

template <typename TDataType>
void NeighborQuery<TDataType>::compute()
{
    if (this->inTriangleIndex()->isEmpty())
    {
        if (!this->inPosition()->isEmpty())
        {
            int p_num = this->inPosition()->getElementCount();
            if (this->outNeighborhood()->getElementCount() != p_num)
            {
                this->outNeighborhood()->setElementCount(p_num);
            }

            m_hash.clear();
            m_hash.construct(this->inPosition()->getValue());

            if (!this->outNeighborhood()->getValue().isLimited())
            {
                queryNeighborDynamic(this->outNeighborhood()->getValue(), this->inPosition()->getValue(), this->inRadius()->getValue());
            }
            else
            {
                queryNeighborFixed(this->outNeighborhood()->getValue(), this->inPosition()->getValue(), this->inRadius()->getValue());
            }
        }
    }
    else
    {
        if (!this->inPosition()->isEmpty())
        {
            int p_num = this->inPosition()->getElementCount();
            if (this->outNeighborhood()->getElementCount() != p_num)
            {
                this->outNeighborhood()->setElementCount(p_num);
            }

            if (triangle_first)
                m_hash.clear();
            //printf("hash clear\n");

            if (triangle_first)
                m_hash.construct(this->inPosition()->getValue(), this->inTriangleIndex()->getValue(), this->inTrianglePosition()->getValue());

            //printf("hash constract\n");
            triangle_first = false;
            queryNeighborTriDynamic(this->outNeighborhood()->getValue(), this->inPosition()->getValue(), this->inTrianglePosition()->getValue(), this->inTriangleIndex()->getValue(), this->inRadius()->getValue());
        }
    }
}

template <typename TDataType>
void NeighborQuery<TDataType>::setBoundingBox(Coord lowerBound, Coord upperBound)
{
    m_lowBound  = lowerBound;
    m_highBound = upperBound;
}

template <typename TDataType>
void NeighborQuery<TDataType>::queryParticleNeighbors(NeighborList<int>& nbr, DeviceArray<Coord>& pos, Real radius)
{
    HostArray<Coord> hostPos;
    hostPos.resize(pos.size());

    Function1Pt::copy(hostPos, pos);

    // 		m_lowBound = Vector3f(10000000, 10000000, 10000000);
    // 		m_highBound = Vector3f(-10000000, -10000000, -10000000);
    //
    // 		for (int i = 0; i < pos.size(); i++)
    // 		{
    // 			m_lowBound[0] = min(hostPos[i][0], m_lowBound[0]);
    // 			m_lowBound[1] = min(hostPos[i][1], m_lowBound[1]);
    // 			m_lowBound[2] = min(hostPos[i][2], m_lowBound[2]);
    //
    // 			m_highBound[0] = max(hostPos[i][0], m_highBound[0]);
    // 			m_highBound[1] = max(hostPos[i][1], m_highBound[1]);
    // 			m_highBound[2] = max(hostPos[i][2], m_highBound[2]);
    // 		}

    m_hash.setSpace(radius, m_lowBound, m_highBound);
    m_hash.construct(this->inPosition()->getValue());

    if (!nbr.isLimited())
    {
        queryNeighborDynamic(nbr, pos, radius);
    }
    else
    {
        queryNeighborFixed(nbr, pos, radius);
    }
}

template <typename Real, typename Coord, typename TDataType>
__global__ void K_CalNeighborSize(
    DeviceArray<int>    count,
    DeviceArray<Coord>  position_new,
    DeviceArray<Coord>  position,
    GridHash<TDataType> hash,
    Real                h)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId > position_new.size())
        return;

    Coord pos_ijk = position_new[pId];
    int3  gId3    = hash.getIndex3(pos_ijk);

    int counter = 0;
    for (int c = 0; c < 27; c++)
    {
        int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
        if (cId >= 0)
        {
            int totalNum = hash.getCounter(cId);
            for (int i = 0; i < totalNum; i++)
            {
                int  nbId = hash.getParticleId(cId, i);
                Real d_ij = (pos_ijk - position[nbId]).norm();
                if (d_ij < h)
                {
                    counter++;
                }
            }
        }
    }

    count[pId] = counter;
}
template <typename Real, typename Coord, typename TDataType>
__global__ void K_CalNeighborSizeTri(
    DeviceArray<int>                      count,
    DeviceArray<Coord>                    position_new,
    DeviceArray<Coord>                    position,
    DeviceArray<TopologyModule::Triangle> m_triangle_index,
    DeviceArray<Coord>                    positionTri,
    GridHash<TDataType>                   hash,
    Real                                  h)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= position_new.size())
        return;

    Coord pos_ijk = position_new[pId];
    int3  gId3    = hash.getIndex3(pos_ijk);

    int  nbr_tri[300];
    bool flag[100];
    //for (int i = 0; i < 100; i++)flag[i] = false;

    int counter = 0;
    for (int c = 0; c < 1; c++)
    {
        int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
        if (cId >= 0)
        {
            int totalNum = hash.getCounter(cId);  // min(hash.getCounter(cId), hash.npMax);
                                                  //	if(counter > 0)
            //printf("%d\n", counter);
            for (int i = 0; i < totalNum; i++)
            {
                int  nbId = hash.getParticleId(cId, i);
                Real d_ij;
                if (nbId >= 0)
                    d_ij = h + EPSILON;  //(pos_ijk - position[nbId]).norm();
                else
                {
                    nbId *= -1;
                    nbId -= 1;
                    Point3D    p3d = Point3D(pos_ijk);
                    Triangle3D t3d = Triangle3D(positionTri[m_triangle_index[nbId][0]], positionTri[m_triangle_index[nbId][1]], positionTri[m_triangle_index[nbId][2]]);
                    d_ij           = p3d.distance(t3d);
                }
                if (d_ij < h)
                {
                    bool legal = true;
                    if (legal)
                    {
                        flag[nbId % 100] = 1;
                        nbr_tri[counter] = nbId;
                        counter++;
                    }
                }
            }
        }
    }

    count[pId] = counter;
}

template <typename Real, typename Coord, typename TDataType>
__global__ void K_GetNeighborElements(
    NeighborList<int>   nbr,
    DeviceArray<Coord>  position_new,
    DeviceArray<Coord>  position,
    GridHash<TDataType> hash,
    Real                h)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId > position_new.size())
        return;

    Coord pos_ijk = position_new[pId];
    int3  gId3    = hash.getIndex3(pos_ijk);

    int j = 0;
    for (int c = 0; c < 27; c++)
    {
        int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
        if (cId >= 0)
        {
            int totalNum = hash.getCounter(cId);  // min(hash.getCounter(cId), hash.npMax);
            for (int i = 0; i < totalNum; i++)
            {
                int  nbId = hash.getParticleId(cId, i);
                Real d_ij = (pos_ijk - position[nbId]).norm();
                if (d_ij < h)
                {
                    nbr.setElement(pId, j, nbId);
                    j++;
                }
            }
        }
    }
}

template <typename Real, typename Coord, typename TDataType>
__global__ void K_GetNeighborElementsTri(
    NeighborList<int>                     nbr,
    DeviceArray<Coord>                    position_new,
    DeviceArray<Coord>                    position,
    DeviceArray<TopologyModule::Triangle> m_triangle_index,
    DeviceArray<Coord>                    positionTri,
    GridHash<TDataType>                   hash,
    Real                                  h)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= position_new.size())
        return;

    Coord pos_ijk = position_new[pId];
    int3  gId3    = hash.getIndex3(pos_ijk);

    int  j   = 0;
    bool tmp = false;
    bool flag[100];
    //for (int i = 0; i < 100; i++)flag[i] = false;

    for (int c = 0; c < 1; c++)
    {
        int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
        if (cId >= 0)
        {
            int totalNum = hash.getCounter(cId);
            for (int i = 0; i < totalNum; i++)
            {
                int nbId = hash.getParticleId(cId, i);
                if (nbId >= 0)
                {
                }
                else
                {
                    int  nb_id = nbId;
                    Real d_ij;
                    nb_id *= -1;
                    nbId *= -1;
                    nbId -= 1;
                    nb_id -= 1;
                    Point3D    p3d = Point3D(pos_ijk);
                    Triangle3D t3d = Triangle3D(positionTri[m_triangle_index[nbId][0]], positionTri[m_triangle_index[nbId][1]], positionTri[m_triangle_index[nbId][2]]);
                    d_ij           = p3d.distance(t3d);
                    if ((d_ij) < h)
                    {
                        if (abs(d_ij) < h)
                        {
                            bool legal = true;
                            if (legal)
                            {
                                nbr.setElement(pId, j, (-nbId - 1));
                                j++;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename TDataType>
void NeighborQuery<TDataType>::queryNeighborSize(DeviceArray<int>& num, DeviceArray<Coord>& pos, Real h)
{
    uint pDims = cudaGridSize(num.size(), BLOCK_SIZE);
    K_CalNeighborSize<<<pDims, BLOCK_SIZE>>>(num, pos, this->inPosition()->getValue(), m_hash, h);
    cuSynchronize();
}

template <typename TDataType>
void NeighborQuery<TDataType>::queryNeighborSizeTri(DeviceArray<int>& num, DeviceArray<Coord>& pos, DeviceArray<Triangle>& Tris, DeviceArray<Coord>& posT, Real h)
{
    uint pDims = cudaGridSize(num.size(), BLOCK_SIZE);
    K_CalNeighborSizeTri<<<pDims, BLOCK_SIZE>>>(num, pos, this->inPosition()->getValue(), Tris, posT, m_hash, h);
    cuSynchronize();
}

template <typename TDataType>
void NeighborQuery<TDataType>::queryNeighborDynamic(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h)
{
    if (pos.size() <= 0)
    {
        return;
    }

    DeviceArray<int>& nbrNum = nbrList.getIndex();
    if (nbrNum.size() != pos.size())
        nbrList.resize(pos.size());

    queryNeighborSize(nbrNum, pos, h);

    int sum = m_reduce.accumulate(nbrNum.getDataPtr(), nbrNum.size());

    m_scan.exclusive(nbrNum, true);
    cuSynchronize();

    if (sum > 0)
    {
        DeviceArray<int>& elements = nbrList.getElements();
        elements.resize(sum);

        uint pDims = cudaGridSize(pos.size(), BLOCK_SIZE);
        K_GetNeighborElements<<<pDims, BLOCK_SIZE>>>(nbrList, pos, this->inPosition()->getValue(), m_hash, h);
        cuSynchronize();
    }
}
template <typename TDataType>
void NeighborQuery<TDataType>::queryNeighborTriDynamic(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, DeviceArray<Coord>& posT, DeviceArray<Triangle>& Tris, Real h)
{
    if (pos.size() <= 0)
    {
        return;
    }

    DeviceArray<int>& nbrNum = nbrList.getIndex();
    if (nbrNum.size() != pos.size())
        nbrList.resize(pos.size());

    queryNeighborSizeTri(nbrNum, pos, Tris, posT, h);

    int sum = m_reduce.accumulate(nbrNum.getDataPtr(), nbrNum.size());

    m_scan.exclusive(nbrNum, true);
    cuSynchronize();

    if (sum > 0)
    {
        DeviceArray<int>& elements = nbrList.getElements();
        elements.resize(sum);
        elements.reset();

        uint pDims = cudaGridSize(pos.size(), BLOCK_SIZE);
        K_GetNeighborElementsTri<<<pDims, BLOCK_SIZE>>>(nbrList, pos, this->inPosition()->getValue(), Tris, posT, m_hash, h);
        cuSynchronize();
    }
}

template <typename Real, typename Coord, typename TDataType>
__global__ void K_ComputeNeighborFixed(
    NeighborList<int>   neighbors,
    DeviceArray<Coord>  position_new,
    DeviceArray<Coord>  position,
    GridHash<TDataType> hash,
    Real                h,
    int*                heapIDs,
    Real*               heapDistance)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId > position_new.size())
        return;

    int nbrLimit = neighbors.getNeighborLimit();

    int*  ids(heapIDs + pId * nbrLimit);            // = new int[nbrLimit];
    Real* distance(heapDistance + pId * nbrLimit);  // = new Real[nbrLimit];

    Coord pos_ijk = position_new[pId];
    int3  gId3    = hash.getIndex3(pos_ijk);

    int counter = 0;
    for (int c = 0; c < 27; c++)
    {
        int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
        if (cId >= 0)
        {
            int totalNum = hash.getCounter(cId);  // min(hash.getCounter(cId), hash.npMax);
            for (int i = 0; i < totalNum; i++)
            {
                int   nbId = hash.getParticleId(cId, i);
                float d_ij = (pos_ijk - position[nbId]).norm();
                if (d_ij < h)
                {
                    if (counter < nbrLimit)
                    {
                        ids[counter]      = nbId;
                        distance[counter] = d_ij;
                        counter++;
                    }
                    else
                    {
                        int   maxId   = 0;
                        float maxDist = distance[0];
                        for (int ne = 1; ne < nbrLimit; ne++)
                        {
                            if (maxDist < distance[ne])
                            {
                                maxDist = distance[ne];
                                maxId   = ne;
                            }
                        }
                        if (d_ij < distance[maxId])
                        {
                            distance[maxId] = d_ij;
                            ids[maxId]      = nbId;
                        }
                    }
                }
            }
        }
    }

    neighbors.setNeighborSize(pId, counter);

    int bId;
    for (bId = 0; bId < counter; bId++)
    {
        neighbors.setElement(pId, bId, ids[bId]);
    }
}

template <typename TDataType>
void NeighborQuery<TDataType>::queryNeighborFixed(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h)
{
    int   num = pos.size();
    int*  ids;
    Real* distance;
    cuSafeCall(cudaMalloc(( void** )&ids, num * sizeof(int) * nbrList.getNeighborLimit()));
    cuSafeCall(cudaMalloc(( void** )&distance, num * sizeof(int) * nbrList.getNeighborLimit()));

    uint pDims = cudaGridSize(num, BLOCK_SIZE);
    K_ComputeNeighborFixed<<<pDims, BLOCK_SIZE>>>(
        nbrList,
        pos,
        this->inPosition()->getValue(),
        m_hash,
        h,
        ids,
        distance);
    cuSynchronize();

    cuSafeCall(cudaFree(ids));
    cuSafeCall(cudaFree(distance));
}
}  // namespace PhysIKA