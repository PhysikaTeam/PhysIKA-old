#include "Collision.h"
#include "Collid.h"

namespace PhysIKA {
Collision* Collision::instance = NULL;

void Collision::transformPair(unsigned int a, unsigned int b)
{
    mesh_pairs.push_back(MeshPair(a, b));
}

void Collision::transformMesh(unsigned int numVtx, unsigned int numTri, std::vector<unsigned int> tris, std::vector<float> vtxs, std::vector<float> p_vtxs, int m_id, bool able_selfcollision)
{
    CollisionMesh::tri3f* _tris    = new CollisionMesh::tri3f[numTri];
    vec3f*                _vtxs    = new vec3f[numVtx];
    vec3f*                pre_vtxs = new vec3f[numVtx];

    for (int i = 0; i < numVtx; i++)
    {
        _vtxs[i]    = vec3f(vtxs[i * 3], vtxs[i * 3 + 1], vtxs[i * 3 + 2]);
        pre_vtxs[i] = vec3f(p_vtxs[i * 3], p_vtxs[i * 3 + 1], p_vtxs[i * 3 + 2]);
    }

    for (int i = 0; i < numTri; i++)
    {
        _tris[i] = CollisionMesh::tri3f(tris[i * 3], tris[i * 3 + 1], tris[i * 3 + 2]);
    }

    CollisionMesh* m = new CollisionMesh(numVtx, numTri, _tris, _vtxs);

    for (int i = 0; i < numVtx; i++)
    {
        m->_ovtxs[i] = pre_vtxs[i];
    }

    dl_mesh.push_back(m);
    bodys.push_back(CollisionDate(m, able_selfcollision));

    delete[] pre_vtxs;
}

void Collision::transformMesh(unsigned int numVtx, unsigned int numTri, std::vector<unsigned int> tris, std::vector<vec3f> vtxs, std::vector<vec3f> p_vtxs, int m_id, bool able_selfcollision)
{
    CollisionMesh::tri3f* _tris    = new CollisionMesh::tri3f[numTri];
    vec3f*                _vtxs    = new vec3f[numVtx];
    vec3f*                pre_vtxs = new vec3f[numVtx];

    for (int i = 0; i < numVtx; i++)
    {
        _vtxs[i]    = vtxs[i];
        pre_vtxs[i] = p_vtxs[i];
    }

    for (int i = 0; i < numTri; i++)
    {
        _tris[i] = CollisionMesh::tri3f(tris[i * 3], tris[i * 3 + 1], tris[i * 3 + 2]);
    }

    CollisionMesh* m = new CollisionMesh(numVtx, numTri, _tris, _vtxs);

    for (int i = 0; i < numVtx; i++)
    {
        m->_ovtxs[i] = pre_vtxs[i];
    }

    dl_mesh.push_back(m);
    bodys.push_back(CollisionDate(m, able_selfcollision));

    delete[] pre_vtxs;
}

void Collision::transformMesh(
    TriangleMesh<DataType3f> mesh,
    int                      m_id,
    bool                     able_selfcollision)
{
    const auto& vertices  = mesh.getTriangleSet()->gethPoints();
    const auto& triangles = mesh.getTriangleSet()->getHTriangles();

    const unsigned int numOfVertices  = vertices.size();
    const unsigned int numOfTriangles = triangles.size();

    std::vector<vec3f> vtxs;
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        vtxs.push_back(vec3f(vertices[i][0], vertices[i][1], vertices[i][2]));
    }

    std::vector<unsigned int> tris;
    for (std::size_t i = 0; i < triangles.size(); ++i)
    {
        tris.push_back(triangles[i][0]);
        tris.push_back(triangles[i][1]);
        tris.push_back(triangles[i][2]);
    }

    transformMesh(numOfVertices, numOfTriangles, tris, vtxs, vtxs, m_id);
}

void Collision::collid()
{
    contact_pairs.clear();
    contact_info.clear();
    CCDtime = 0;

    body_collide_gpu(bodys, contact_pairs, CCDtime, contact_info, thickness);
}
}  // namespace PhysIKA