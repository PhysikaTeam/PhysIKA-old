#include "Collision.h"
#include "Collid.h"

namespace PhysIKA {
Collision* Collision::m_instance = nullptr;

Collision* Collision::getInstance()
{
    if (m_instance == NULL)
    {
        m_instance = new Collision();
        return m_instance;
    }
    else
        return m_instance;
}

Collision::~Collision()
{
    for (int i = 0; i < m_dl_mesh.size(); i++)
    {
        delete m_dl_mesh[i];
    }
}

void Collision::transformPair(unsigned int a, unsigned int b)
{
    if (m_is_first == true)
        m_mesh_pairs.push_back(MeshPair(a, b));
}

void Collision::transformMesh(
    unsigned int              num_vertices,
    unsigned int              num_triangles,
    std::vector<unsigned int> tris,
    std::vector<float>        vtxs,
    std::vector<float>        p_vtxs,
    int                       m_id,
    bool                      able_selfcollision)
{
    CollisionMesh::tri3f* _tris    = new CollisionMesh::tri3f[num_triangles];
    vec3f*                _vtxs    = new vec3f[num_vertices];
    vec3f*                pre_vtxs = new vec3f[num_vertices];

    for (int i = 0; i < num_vertices; i++)
    {
        _vtxs[i]    = vec3f(vtxs[i * 3], vtxs[i * 3 + 1], vtxs[i * 3 + 2]);
        pre_vtxs[i] = vec3f(p_vtxs[i * 3], p_vtxs[i * 3 + 1], p_vtxs[i * 3 + 2]);
    }

    for (int i = 0; i < num_triangles; i++)
    {
        _tris[i] = CollisionMesh::tri3f(tris[i * 3], tris[i * 3 + 1], tris[i * 3 + 2]);
    }

    CollisionMesh* m = new CollisionMesh(num_vertices, num_triangles, _tris, _vtxs);

    for (int i = 0; i < num_vertices; i++)
    {
        m->_ovtxs[i] = pre_vtxs[i];
    }

    m_dl_mesh.push_back(m);
    if (m_is_first)
    {
        m_bodys.push_back(CollisionDate(m, able_selfcollision));
    }
    else
    {
        memcpy(m_bodys[m_id].ms->_ovtxs, pre_vtxs, num_vertices * sizeof(vec3f));
        memcpy(m_bodys[m_id].ms->_vtxs, m->_vtxs, num_vertices * sizeof(vec3f));
    }

    delete[] pre_vtxs;
}

void Collision::transformMesh(
    unsigned int              num_vertices,
    unsigned int              num_triangles,
    std::vector<unsigned int> tris,
    std::vector<vec3f>        vtxs,
    std::vector<vec3f>        p_vtxs,
    int                       m_id,
    bool                      able_selfcollision)
{
    CollisionMesh::tri3f* _tris    = new CollisionMesh::tri3f[num_triangles];
    vec3f*                _vtxs    = new vec3f[num_vertices];
    vec3f*                pre_vtxs = new vec3f[num_vertices];

    for (int i = 0; i < num_vertices; i++)
    {
        _vtxs[i]    = vtxs[i];
        pre_vtxs[i] = p_vtxs[i];
    }

    for (int i = 0; i < num_triangles; i++)
    {
        _tris[i] = CollisionMesh::tri3f(tris[i * 3], tris[i * 3 + 1], tris[i * 3 + 2]);
    }

    CollisionMesh* m = new CollisionMesh(num_vertices, num_triangles, _tris, _vtxs);

    for (int i = 0; i < num_vertices; i++)
    {
        m->_ovtxs[i] = pre_vtxs[i];
    }

    m_dl_mesh.push_back(m);
    if (m_is_first)
    {
        m_bodys.push_back(CollisionDate(m, able_selfcollision));
    }
    else
    {
        memcpy(m_bodys[m_id].ms->_ovtxs, pre_vtxs, num_vertices * sizeof(vec3f));
        memcpy(m_bodys[m_id].ms->_vtxs, m->_vtxs, num_vertices * sizeof(vec3f));
    }

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

std::vector<std::vector<TrianglePair>> Collision::getContactPairs()
{
    return m_contact_pairs;
}

int Collision::getNumContacts()
{
    return m_contact_pairs.size();
}

int Collision::getCCDTime()
{
    return m_ccd_time;
}

void Collision::setThickness(float thickness)
{
    m_thickness = thickness;
}

std::vector<ImpactInfo> Collision::getImpactInfo()
{
    return m_contact_info;
}

void Collision::collid()
{
    m_contact_pairs.clear();
    m_contact_info.clear();
    m_ccd_time = 0;

    body_collide_gpu(m_bodys, m_contact_pairs, m_ccd_time, m_contact_info, m_thickness);

    m_is_first = false;
}
}  // namespace PhysIKA