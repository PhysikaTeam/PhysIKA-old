#include "Dynamics/HeightField/HeightFieldMesh.h"

namespace PhysIKA {
__global__ void HFMesh_generateVertices(DeviceArray<Vector3f> vertices, /*DeviceArray<TopologyModule::Triangle> triangles,*/
                                        DeviceHeightField1d   hfield)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= (hfield.Nx()) || tidy >= (hfield.Ny()))
        return;

    int      id  = tidx + tidy * hfield.Nx();
    Vector3d pos = hfield.gridCenterPosition(tidx, tidy);
    pos[1]       = hfield.get(pos[0], pos[2]);

    Vector3f ver(pos[0], pos[1], pos[2]);
    vertices[id] = ver;

    //if (tidx == tidy)
    //{
    //	printf("%d %d, Pos:  %lf %lf %lf\n", tidx, tidy, pos[0], pos[1], pos[2]);
    //}
}

__global__ void HFMesh_generateTriangle(DeviceArray<TopologyModule::Triangle> triangles,
                                        DeviceHeightField1d                   hfield)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= (hfield.Nx() - 1) || tidy >= (hfield.Ny() - 1))
        return;

    int id = tidx + tidy * (hfield.Nx() - 1);

    TopologyModule::Triangle curtri;
    curtri[0]         = tidx + tidy * hfield.Nx();
    curtri[1]         = (tidx + 1) + (tidy + 1) * hfield.Nx();
    curtri[2]         = (tidx + 1) + ( tidy )*hfield.Nx();
    triangles[2 * id] = curtri;

    curtri[0]             = tidx + tidy * hfield.Nx();
    curtri[1]             = (tidx) + (tidy + 1) * hfield.Nx();
    curtri[2]             = (tidx + 1) + (tidy + 1) * hfield.Nx();
    triangles[2 * id + 1] = curtri;
}

void HeightFieldMesh::generate(std::shared_ptr<TriangleSet<DataType3f>> triset, DeviceHeightField1d& hfield)
{
    auto ptriangles = triset->getTriangles();
    auto pvertices  = &(triset->getPoints());

    uint3 gsizeVer = { hfield.Nx(), hfield.Ny(), 1 };
    uint3 gsizeTri = { hfield.Nx() - 1, hfield.Ny() - 1, 1 };

    pvertices->resize(gsizeVer.x * gsizeVer.y);
    ptriangles->resize(gsizeTri.x * gsizeTri.y * 2);

    cuExecute2D(gsizeVer, HFMesh_generateVertices, *pvertices, hfield);

    cuExecute2D(gsizeTri, HFMesh_generateTriangle, *ptriangles, hfield);
}
}  // namespace PhysIKA