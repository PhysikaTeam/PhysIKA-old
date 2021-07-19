/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: collision detection api entry point
 * @version    : 1.0
 */

#pragma once

#include "Core/Array/Array.h"
#include "Framework/Framework/CollidableObject.h"
#include "Framework/Collision/CollisionBVH.h"
#include <vector>
#include <memory>
#include "Core/DataTypes.h"
#include "Dynamics/RigidBody/TriangleMesh.h"
#include <time.h>

namespace PhysIKA {
class TrianglePair;
class bvh;
/**
     * Simple collision detection class
     *
     * Sample usage:
     * bool collid = DCD->checkCollision(mesh1, mesh2);
     */
template <typename TDataType>
class CollidatableTriangleMesh : public CollidableObject
{
    DECLARE_CLASS_1(CollidatableTriangleMesh, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    /**
         * constructor
         */
    CollidatableTriangleMesh();

    /**
         * destructor
         */
    virtual ~CollidatableTriangleMesh();

    static std::shared_ptr<bvh> bvh1;  // static bvh *bvh1;
    static std::shared_ptr<bvh> bvh2;  // static bvh *bvh2;

    /**
         * check whether two mesh is collided
         * 
         * @param[in] b1 triangle mesh 1 to be tested
         * @param[in] b2 triangle mesh 2 to be tested
         * @return whether two mesh is collided
         */
    static bool checkCollision(std::shared_ptr<TriangleMesh<TDataType>> b1, std::shared_ptr<TriangleMesh<TDataType>> b2)
    {

        if (bvh1.get() == nullptr)
        {
            std::vector<std::shared_ptr<TriangleMesh<TDataType>>> meshes;
            meshes.push_back(b1);

            bvh1 = std::make_shared<bvh>(meshes);
            bvh2 = std::make_shared<bvh>(meshes);
        }

        std::vector<std::shared_ptr<TriangleMesh<TDataType>>> meshes1;
        meshes1.push_back(b1);
        std::vector<std::shared_ptr<TriangleMesh<TDataType>>> meshes2;
        meshes2.push_back(b2);
        clock_t start, end;
        start = clock();
        //rebuild bvh
        bvh1->refit(meshes1);
        bvh2->refit(meshes2);
        end = clock();
        std::vector<TrianglePair> ret;

        bvh1.get()->collide(bvh2.get(), ret);
        if (ret.size())
        {
            printf("check collision elapsed=%f ms.\n", ( float )(end - start) * 1000 / CLOCKS_PER_SEC);
            printf("to checked ret size%d\n", ret.size());
        }
        for (size_t i = 0; i < ret.size(); i++)
        {
            TrianglePair& t = ret[i];
            unsigned int  id0, id1;
            t.get(id0, id1);
            if (CollidableTriangle<DataType3f>::checkSelfIJ(b1.get(), id0, b2.get(), id1))
                return true;
        }

        //return (ret.size() > 0);
        return false;
    }

    /**
         * override function for initialization
         */
    bool initializeImpl() override;

    /**
         * update colliable objects
         */
    void updateCollidableObject() override;

    /**
         * update mechanical state
         */
    void updateMechanicalState() override;

private:
};

/**
     * manager of the meshes in the scene, used to save and load data 
     *
       * Sample usage:
     * // storage data
     * auto obj = std::make_shared<RigidCollisionBody<DataType3f>>();
     * // after some initialiation of obj
     * // ...
     * CollisionManager::Meshes.push_back(obj->getmeshPtr());
     * 
     * // load data
     * CollisionManager::Meshes[i]
     */
class CollisionManager
{
public:
    CollisionManager(){};
    static std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> Meshes;  //!< storage for input mesh

    /**
         * get the id of the mesh
         * 
         * @param[in]  id  id 
         * @param[in]  m   mesh
         * @param[out] mid mesh id
         * @param[out] fid face id
         */
    static void mesh_id(int id, std::vector<std::shared_ptr<TriangleMesh<DataType3f>>>& m, int& mid, int& fid)
    {
        fid = id;
        for (mid = 0; mid < m.size(); mid++)
            if (fid < m[mid]->_num_tri)
            {
                return;
            }
            else
            {
                fid -= m[mid]->_num_tri;
            }

        assert(false);
        fid = -1;
        mid = -1;
        printf("mesh_id error!!!!\n");
        abort();
    }

    /**
         * check whether two mesh specified by ids is covertex
         *
         * @param[in] id1 mesh id1
         * @param[in] id2 mesh id2
         * @return whether two mesh specified by ids is covertex
         */
    static bool covertex(int id1, int id2)
    {
        if (Meshes.empty())
            return false;

        int mid1, fid1, mid2, fid2;

        mesh_id(id1, Meshes, mid1, fid1);
        mesh_id(id2, Meshes, mid2, fid2);

        if (mid1 != mid2)
            return false;

        TopologyModule::Triangle& f1 = Meshes[mid1]->triangleSet->getHTriangles()[fid1];
        TopologyModule::Triangle& f2 = Meshes[mid2]->triangleSet->getHTriangles()[fid2];

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (f1[i] == f2[2])
                    return true;

        return false;
    }

    /**
         * storage mesh
         *
         * @param[in] meshes meshes to be storaged
         */
    static void self_mesh(std::vector<std::shared_ptr<TriangleMesh<DataType3f>>>& meshes)
    {
        Meshes = meshes;
    }
};
}  // namespace PhysIKA
