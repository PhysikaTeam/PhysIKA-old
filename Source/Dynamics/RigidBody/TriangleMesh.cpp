#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Framework/Framework/Node.h"
#include "Framework/Collision/CollisionBVH.h"
#include <vector>

#define _CRT_SECURE_NO_WARNINGS

using namespace std;
namespace PhysIKA {
template <typename TDataType>
TriangleMesh<TDataType>::TriangleMesh()
{
}
template <typename TDataType>
void TriangleMesh<TDataType>::buildBVH(std::vector<std::shared_ptr<TriangleMesh<TDataType>>>& meshes)
{
    _bvh = new bvh(meshes);
}
template <typename TDataType>
TriangleMesh<TDataType>::TriangleMesh(const char* path, Vector3f offset, Vector3f axis, Real theta)
{
    bool noerr = readobjfile(path, offset, axis, theta);
    if (!noerr)
    {
        printf("triangle mesh initilize failed");
        return;
    }
}

template <typename TDataType>
void TriangleMesh<TDataType>::translate(Coord t)
{
    triangleSet->translate(t);
}

template <typename TDataType>
TriangleMesh<TDataType>::~TriangleMesh()
{
}
template <typename TDataType>
void TriangleMesh<TDataType>::updataBxs()
{
    for (int i = 0; i < _num_tri; i++)
    {
        Triangle&      a  = triangleSet->h_triangles[i];
        const Coord3D& p0 = triangleSet->h_coords[a[0]];
        const Coord3D& p1 = triangleSet->h_coords[a[1]];
        const Coord3D& p2 = triangleSet->h_coords[a[2]];

        TAlignedBox3D<Real> bx(p0, p1);
        bx += p2;
        _bxs[i] = bx;

        _bx += bx;
    }
}
template <typename TDataType>
TAlignedBox3D<typename TDataType::Real> TriangleMesh<TDataType>::bound()
{
    return _bx;
}

template <typename TDataType>
bvh* TriangleMesh<TDataType>::getBVH()
{
    return _bvh;
}

template <typename TDataType>
void TriangleMesh<TDataType>::loadFromSet(std::shared_ptr<TriangleSet<DataType3f>> set)
{
    triangleSet = TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(m_surfaceNode->getTopologyModule());
    if (set.get() != nullptr)
    {
        *triangleSet.get() = *set.get();
    }
    _num_vtx = triangleSet->h_coords.size();
    _num_tri = triangleSet->h_triangles.size();
    _bxs     = new TAlignedBox3D<Real>[_num_tri];
    _areas   = new Real[_num_tri];
    //calcAreas(texs, ttris);
    updataBxs();
}

template <typename TDataType>
bool TriangleMesh<TDataType>::readobjfile(const char* path, Vector3f offset, Vector3f axis, Real theta)
{
    {
        _offset = offset;
        _axis   = axis;
        _theta  = theta;
    }
    vector<Triangle> triset;
    vector<Vector3f> vtxset;
    vector<Vector2f> texset;
    vector<Triangle> ttriset;

    FILE* fp = fopen(path, "rt");
    if (fp == NULL)
        return false;

    _trf = rotation(axis, theta / 180.0 * M_PI);

    char buf[1024];
    while (fgets(buf, 1024, fp))
    {
        if (buf[0] == 'v' && buf[1] == ' ')
        {
            double x, y, z;
            sscanf(buf + 2, "%lf%lf%lf", &x, &y, &z);
            vtxset.push_back(Vector3f(x, y, z));
        }
        else

            if (buf[0] == 'v' && buf[1] == 't')
        {
            double x, y;
            sscanf(buf + 3, "%lf%lf", &x, &y);

            texset.push_back(Vector2f(x, y));
        }
        else if (buf[0] == 'f' && buf[1] == ' ')
        {
            int  id0, id1, id2, id3;
            int  tid0, tid1, tid2, tid3;
            bool quad = false;

            int   count = sscanf(buf + 2, "%d/%d", &id0, &tid0);
            char* nxt   = strchr(buf + 2, ' ');
            sscanf(nxt + 1, "%d/%d", &id1, &tid1);
            nxt = strchr(nxt + 1, ' ');
            sscanf(nxt + 1, "%d/%d", &id2, &tid2);

            nxt = strchr(nxt + 1, ' ');
            if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9')
            {  // quad
                if (sscanf(nxt + 1, "%d/%d", &id3, &tid3))
                    quad = true;
            }

            if (quad)
            {
                id3--;
            }
            if (quad)
            {
                tid3--;
            }
            id0--, id1--, id2--;
            tid0--, tid1--, tid2--;

            triset.push_back(Triangle(id0, id1, id2));
            if (count == 2)
            {
                ttriset.push_back(Triangle(tid0, tid1, tid2));
            }

            if (quad)
            {
                triset.push_back(Triangle(id0, id2, id3));
                if (count == 2)
                    ttriset.push_back(Triangle(tid0, tid2, tid3));
            }
        }
    }
    fclose(fp);

    if (triset.size() == 0 || vtxset.size() == 0)
        return false;

    _num_vtx = vtxset.size();
    //_vtxs = new Vector3f[_num_vtx];
    //_ivtxs = new Vector3f[_num_vtx];
    for (unsigned int i = 0; i < _num_vtx; i++)
    {
        //_ivtxs[i] = vtxset[i];
        //_vtxs[i] = _trf * vtxset[i] + offset;
    }

    /*
        int numTex = texset.size();
        if (numTex == 0)
            texs = NULL;
        else {
            texs = new Vector2f[numTex];
            for (unsigned int i = 0; i < numTex; i++)
                texs[i] = texset[i];
        }
        */

    _num_tri = triset.size();
    //_tris = new tri3f[_num_tri];
    for (unsigned int i = 0; i < _num_tri; i++)
        ;
    //_tris[i] = triset[i];

    /*
        int numTTri = ttriset.size();
        if (numTTri == 0)
            _ttris = NULL;
        else {
            ttris = new tri3f[numTTri];
            for (unsigned int i = 0; i < numTTri; i++)
                ttris[i] = ttriset[i];
        }
        */
    _ovtxs = new Vector3f[_num_vtx];
    //_nrms = new Vector3f[_num_vtx];
    _bxs   = new TAlignedBox3D<Real>[_num_tri];
    _areas = new Real[_num_tri];
    //calcAreas(texs, ttris);
    updataBxs();
    return true;
}

}  // namespace PhysIKA