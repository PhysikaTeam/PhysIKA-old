#include <set>
#include <iostream>
#include <stdio.h>

#include "CollisionMesh.h"
#include "CollisionBVH.h"
#include "CollisionDate.h"

extern int  getCollisionsGPU(int*, int*, int*, float*, int*, int*, float*);
extern int  getSelfCollisionsSH(int*);
extern void pushMesh2GPU(int numFace, int numVert, void* faces, void* nodes);
extern void updateMesh2GPU(void* nodes, void* prenodes, float thickness);

extern void initGPU();

namespace PhysIKA {
static CollisionMesh::tri3f* s_faces;
static vec3f*                s_nodes;
static int                   s_numFace = 0, s_numVert = 0;

void updateMesh2GPU(std::vector<CollisionMesh*>& ms, float thickness)
{
    vec3f* curVert = s_nodes;

    //rky
    vec3f*             preVert = new vec3f[s_numVert];
    std::vector<vec3f> tem;
    vec3f*             oldcurVert = preVert;
    for (int i = 0; i < ms.size(); i++)
    {
        CollisionMesh* m = ms[i];
        memcpy(oldcurVert, m->_ovtxs, sizeof(vec3f) * m->_num_vtx);
        oldcurVert += m->_num_vtx;
    }

    for (int i = 0; i < ms.size(); i++)
    {
        CollisionMesh* m = ms[i];
        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }

    for (int i = 0; i < ms.size(); i++)
    {
        for (int j = 0; j < ms[i]->_num_vtx; j++)
        {
            tem.push_back(ms[i]->_vtxs[j]);
            tem.push_back(ms[i]->_ovtxs[j]);
        }
    }

    ::updateMesh2GPU(s_nodes, preVert, thickness);
}

void pushMesh2GPU(std::vector<CollisionMesh*>& ms)
{
    for (int i = 0; i < ms.size(); i++)
    {
        s_numFace += ms[i]->_num_tri;
        s_numVert += ms[i]->_num_vtx;
    }

    s_faces = new CollisionMesh::tri3f[s_numFace];
    s_nodes = new vec3f[s_numVert];

    int    curFace   = 0;
    int    vertCount = 0;
    vec3f* curVert   = s_nodes;
    for (int i = 0; i < ms.size(); i++)
    {
        CollisionMesh* m = ms[i];
        for (int j = 0; j < m->_num_tri; j++)
        {
            CollisionMesh::tri3f& t = m->_tris[j];
            s_faces[curFace++]      = CollisionMesh::tri3f(t.id0() + vertCount, t.id1() + vertCount, t.id2() + vertCount);
        }
        vertCount += m->_num_vtx;

        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }

    ::pushMesh2GPU(s_numFace, s_numVert, s_faces, s_nodes);
}

void body_collide_gpu(
    std::vector<CollisionDate>              bodys,
    std::vector<std::vector<TrianglePair>>& contacts,
    int&                                    CCDtime,
    std::vector<ImpactInfo>&                contact_info,
    float                                   thickness)
{
    static bvh*                        bvhC = NULL;
    static front_list                  fIntra;
    static std::vector<CollisionMesh*> meshes;

    static std::vector<int> _tri_offset;

#define MAX_CD_PAIRS 14096

    int* buffer      = new int[MAX_CD_PAIRS * 2];
    int* time_buffer = new int[1];

    int*   buffer_vf_ee     = new int[MAX_CD_PAIRS];
    int*   buffer_vertex_id = new int[MAX_CD_PAIRS * 4];
    float* buffer_dist      = new float[MAX_CD_PAIRS];

    int* buffer_CCD = new int[MAX_CD_PAIRS];

    int count = 0;

    if (bvhC == NULL)
    {
        for (int i = 0; i < bodys.size(); i++)
        {
            meshes.push_back(bodys[i].ms);
            _tri_offset.push_back(i == 0 ? bodys[0].ms->_num_tri : (_tri_offset[i - 1] + bodys[i].ms->_num_tri));
        }
        bvhC = new bvh(meshes);

        bvhC->self_collide(fIntra, meshes);

        ::initGPU();
        pushMesh2GPU(meshes);
        bvhC->push2GPU(true);

        fIntra.push2GPU(bvhC->root());
    }

    updateMesh2GPU(meshes, thickness);
    printf("thickness is %f\n", thickness);

    count = ::getCollisionsGPU(buffer, buffer_vf_ee, buffer_vertex_id, buffer_dist, time_buffer, buffer_CCD, &thickness);

    TrianglePair*             pairs = ( TrianglePair* )buffer;
    std::vector<TrianglePair> ret(pairs, pairs + count);

    for (int i = 0; i < count; i++)
    {
        ImpactInfo tem = ImpactInfo(buffer[i * 2], buffer[i * 2 + 1], buffer_vf_ee[i], buffer_vertex_id[i * 4], buffer_vertex_id[i * 4 + 1], buffer_vertex_id[i * 4 + 2], buffer_vertex_id[i * 4 + 3], buffer_dist[i], time_buffer[0], buffer_CCD[i]);

        contact_info.push_back(tem);
    }

    CCDtime = time_buffer[0];

    //Find mesh id and face id
    for (int i = 0; i < count; i++)
    {
        std::vector<TrianglePair> tem;
        int                       mid1, mid2;
        unsigned int              fid1, fid2;
        ret[i].get(fid1, fid2);

        for (int j = 0; j < _tri_offset.size(); j++)
        {
            if (fid1 <= _tri_offset[j])
            {
                mid1 = j == 0 ? 0 : j;
                break;
            }
        }

        tem.push_back(TrianglePair(mid1, fid1 == 0 ? 0 : fid1 - (mid1 == 0 ? 0 : _tri_offset[mid1 - 1])));

        int temtt = fid1 - 1 - (mid1 == 0 ? 0 : _tri_offset[mid1 - 1]);

        for (int j = 0; j < _tri_offset.size(); j++)
        {
            if (fid2 <= _tri_offset[j])
            {
                mid2 = j == 0 ? 0 : j;
                break;
            }
        }

        tem.push_back(TrianglePair(mid2, fid2 == 0 ? 0 : fid2 - (mid2 == 0 ? 0 : _tri_offset[mid2 - 1])));

        contacts.push_back(tem);
    }
    delete[] buffer;
    delete[] time_buffer;
    delete[] buffer_vf_ee;
    delete[] buffer_vertex_id;
    delete[] buffer_dist;
    delete[] buffer_CCD;
}
}  // namespace PhysIKA