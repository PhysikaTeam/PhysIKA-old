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
         * destructor
         */
    ~Collision()
    {
        for (int i = 0; i < dl_mesh.size(); i++)
        {
            delete dl_mesh[i];
        }
    }

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
         * @param[in]     numVtx              num of vertices
         * @param[in]     numTri              num of triangles
         * @param[in]     tris                indices
         * @param[in]     vtxs                vertices
         * @param[in]     pre_vtxs            previous vertices
         * @param[in]     m_id                  mesh id
         * @param[in]     able_selfcollision  check self collison or not
         */
    void transformMesh(unsigned int numVtx, unsigned int numTri, std::vector<unsigned int> tris, std::vector<float> vtxs, std::vector<float> pre_vtxs, int m_id, bool able_selfcollision = false);

    /**
         * input the mesh in vertices and indices format to the collision instance
         *
         * @param[in]     numVtx              num of vertices
         * @param[in]     numTri              num of triangles
         * @param[in]     tris                indices
         * @param[in]     vtxs                vertices
         * @param[in]     pre_vtxs            previous vertices
         * @param[in]     m_id                  mesh id
         * @param[in]     able_selfcollision  check self collison or not
         */
    void transformMesh(unsigned int numVtx, unsigned int numTri, std::vector<unsigned int> tris, std::vector<vec3f> vtxs, std::vector<vec3f> pre_vtxs, int m_id, bool able_selfcollision = false);

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
    std::vector<std::vector<TrianglePair>> getContactPairs()
    {
        return contact_pairs;
    }

    /**
         * get number of collision pairs
         *
         * @return number of collision pairs
         */
    int getNumContacts()
    {
        return contact_pairs.size();
    }

    /**
         * return ccd results
         *
         * @return ccd time
         * @retval 1 collision appears
         * @retval 0 no collision
         */
    int getCCD_res()
    {
        return CCDtime;
    }

    /**
         * set thickness
         *
         * @param[in]     thickness     thickness of the face
         */
    void setThickness(float tt)
    {
        thickness = tt;
    }

    /**
         * get collision info
         *
         * @return array of impact info
         */
    std::vector<ImpactInfo> getImpactInfo()
    {
        return contact_info;
    }

    /**
         * get collision instance
         *
         * @return collision instance
         */
    static Collision* getInstance()
    {
        if (instance == NULL)
        {
            instance = new Collision();
            return instance;
        }
        else
            return instance;
    }

    static Collision* instance;  //!< singleton instance

private:
    /**
         * the constructor is private since this is a singleton class
         */
    Collision() = default;

private:
    std::vector<CollisionDate>             bodys;             //!< collision meshes
    std::vector<MeshPair>                  mesh_pairs;        //!< collision mesh pairs
    std::vector<std::vector<TrianglePair>> contact_pairs;     //!< collision results
    std::vector<CollisionMesh*>            dl_mesh;           //!< delete mesh points
    std::vector<ImpactInfo>                contact_info;      //!< collision impact info
    int                                    CCDtime   = 0;     //!< collision time
    float                                  thickness = 0.0f;  //!< face thickness
};
}  // namespace PhysIKA