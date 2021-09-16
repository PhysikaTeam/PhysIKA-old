#pragma once

/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: collision detection api entry point
 * @version    : 1.0
 */

#include "Dynamics/RigidBody/TriangleMesh.h"
#include "CollidableTriangle.h"
#include "CollisionMesh.h"
#include "CollisionBVH.h"
#include "CollisionDate.h"

#include <iostream>
#include <vector>

namespace PhysIKA {
/**
 * Singleton class for collision detection
 *
 * Sample usage:
 * std::unique_ptr<Collision> collision = Collision::getInstance();
 * TriangleMesh triangleMeshes[2];
 * // init two triangleMeshes
 * // ...
 * // transform data
 * collsion->transformMesh(triangleMeshes[0], 0);
 * collsion->transformMesh(triangleMeshes[1], 1);
 * // collision detection
 * collision->collid();
 * // get results
 * auto results = collision->getContactPairs();
 */
class Collision
{
public:
    using MeshPair = std::pair<int, int>;

    /**
     * get collision instance
     *
     * @return collision instance
     */
    static Collision* getInstance();

    /**
      * destructor
      */
    ~Collision();

    /**
     * check collision for input meshes, and the result will be
     * store in the class member variables.
     */
    void collid();

    /**
     * specify the mesh pairs for c
     *
     * @param[in]     in     parameter to read only
     * @param[in,out] in_out parameter to read and write
     * @param[out]    out    parameter to write only
     */
    void transformPair(unsigned int a, unsigned int b);

    /**
     * input the mesh in data array format to the collision instance
     *
     * @param[in]     num_vertices        num of vertices
     * @param[in]     num_triangles       num of triangles
     * @param[in]     tris                indices
     * @param[in]     vtxs                vertices
     * @param[in]     pre_vtxs            previous vertices
     * @param[in]     m_id                mesh id
     * @param[in]     able_selfcollision  check self collison or not
     */
    void transformMesh(unsigned int              num_vertices,
                       unsigned int              num_triangles,
                       std::vector<unsigned int> tris,
                       std::vector<float>        vtxs,
                       std::vector<float>        pre_vtxs,
                       int                       m_id,
                       bool                      able_selfcollision = false);

    /**
     * input the mesh in vertices and indices format to the collision instance
     *
     * @param[in]     num_vertices              num of vertices
     * @param[in]     num_triangles              num of triangles
     * @param[in]     tris                indices
     * @param[in]     vtxs                vertices
     * @param[in]     pre_vtxs            previous vertices
     * @param[in]     m_id                mesh id
     * @param[in]     able_selfcollision  check self collison or not
     */
    void transformMesh(unsigned int              num_vertices,
                       unsigned int              num_triangles,
                       std::vector<unsigned int> tris,
                       std::vector<vec3f>        vtxs,
                       std::vector<vec3f>        pre_vtxs,
                       int                       m_id,
                       bool                      able_selfcollision = false);

    /**
      * input the triangle mesh to the collision instance
      *
      * @param[in]     mesh                triangle mesh to for collision detection
      * @param[in]     m_id                mesh id
      * @param[in]     able_selfcollision  check self collison or not
      */
    void transformMesh(TriangleMesh<DataType3f> mesh,
                       int                      m_id,
                       bool                     able_selfcollision = false);

    /**
     * get the collided meshes indices and corresponding triangles
     *
     * @return the collided meshes indices and corresponding triangles
     */
    std::vector<std::vector<TrianglePair>> getContactPairs();

    /**
     * get number of collision pairs
     *
     * @return number of collision pairs
     */
    int getNumContacts();

    /**
     * return ccd results
     *
     * @return ccd time
     * @retval 1 collision appears
     * @retval 0 no collision
     */
    int getCCDTime();

    /**
      * set thickness
      *
      * @param[in]     thickness     thickness of the face
      */
    void setThickness(float thickness);

    /**
     * get collision info
     *
     * @return array of impact info
     */
    std::vector<ImpactInfo> getImpactInfo();

private:
    /**
     * the constructor is private since this is a singleton class
     */
    Collision() = default;

private:
    static Collision*                      m_instance;          //!< singleton instance
    std::vector<CollisionDate>             m_bodys;             //!< collision meshes
    std::vector<MeshPair>                  m_mesh_pairs;        //!< collision mesh pairs
    std::vector<std::vector<TrianglePair>> m_contact_pairs;     //!< collision results
    std::vector<CollisionMesh*>            m_dl_mesh;           //!< delete mesh points
    std::vector<ImpactInfo>                m_contact_info;      //!< collision impact info
    int                                    m_ccd_time  = 0;     //!< collision time
    float                                  m_thickness = 0.0f;  //!< face thickness
    bool                                   m_is_first  = true;  //!< record the first time to transform a mesh into api
};
}  // namespace PhysIKA