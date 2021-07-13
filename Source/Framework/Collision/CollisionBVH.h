/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: bvh data structure
 * @version    : 1.0
 */

#pragma once

#include <vector>
#include <memory>
#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Framework/Collision/CollidableTriangle.h"
#include "Framework/Collision/CollidableTriangleMesh.h"
#include "Framework/Framework/ModuleTopology.h"
#include "CollisionMesh.h"

namespace PhysIKA {
class front_node;
class bvh_node;

static vec3f* s_fcenters;  //!< used to store temporary data

/**
     * front list of the front node
     */
class front_list : public std::vector<front_node>
{
public:
    /**
         * push the data structure to gpu
         * 
         * @param[in] r1 bvh node 1
         * @param[in] r2 bvh node 2
         */
    void push2GPU(bvh_node* r1, bvh_node* r2 = NULL);
};

/**
     * bvh node
     */
class bvh_node
{
    TAlignedBox3D<float> _box;       //!< aabb box
    static bvh_node*     s_current;  //!< store temporary data
    int                  _child;     //!< child >=0 leaf with tri_id, <0 left & right
    int                  _parent;    //!< parent

    void setParent(int p);

public:
    /**
         * constructor
         */
    bvh_node();

    /**
         * destructor
         */
    ~bvh_node();

    /**
         * check collision with the other bvh node
         * 
         * @param[in]  other another bvh node
         * @param[out] ret   triangle pair to store the result
         */
    void collide(bvh_node* other, std::vector<TrianglePair>& ret);

    /**
         * check collision with the bvtt front
         *
         * @param[in]  other another bvh node
         * @param[out] f     bvtt front that stores the collision
         * @param[in]  level level
         * @param[in]  ptr   parent
         */
    void collide(bvh_node* other, front_list& f, int level, int ptr);

    /**
         * check self collision within a bvtt front and a node 
         *
         * @param[out] lst front list
         * @param[in]  r   node
         */
    void self_collide(front_list& lst, bvh_node* r);

    /**
         * construct the bvh node
         *
         * @param[in] id        index
         * @param[in] s_fboxes  boxes
         */
    void construct(unsigned int id, TAlignedBox3D<float>* s_fboxes);

    /**
         * construct the bvh node
         *
         * @param[in] lst        list
         * @param[in] num        list item number
         * @param[in] s_fcenters centers
         * @param[in] s_fboxes   boxes
         * @param[in] s_current  current node
         */
    void construct(
        unsigned int*         lst,
        unsigned int          num,
        vec3f*                s_fcenters,
        TAlignedBox3D<float>* s_fboxes,
        bvh_node*&            s_current);

    /**
         * refit algorithm, internal
         *
         * @param[in] s_fboxes boxes
         */
    void refit(TAlignedBox3D<float>* s_fboxes);

    /**
         * reset parents of the tree, internal
         *
         * @param[in] root       tree node
         */
    void resetParents(bvh_node* root);

    /**
         * get boxes
         *
         * @return boxes
         */
    FORCEINLINE TAlignedBox3D<float>& box()
    {
        return _box;
    }

    /**
         * get left child
         *
         * @return left child
         */
    FORCEINLINE bvh_node* left()
    {
        return this - _child;
    }

    /**
         * get right child
         *
         * @return right child
         */
    FORCEINLINE bvh_node* right()
    {
        return this - _child + 1;
    }

    /**
         * get triangle id
         *
         * @return triangle id
         */
    FORCEINLINE int triID()
    {
        return _child;
    }

    /**
         * check leaf node
         *
         * @return whether current node is a leaf node
         */
    FORCEINLINE int isLeaf()
    {
        return _child >= 0;
    }

    /**
         * get parent
         *
         * @return parent id
         */
    FORCEINLINE int parentID()
    {
        return _parent;
    }

    /**
         * get current level
         * 
         * @param[in]  current   current index
         * @param[out] max_level max level
         */
    FORCEINLINE void getLevel(int current, int& max_level);

    /**
         * get level index
         *
         * @param[in]  current current index
         * @param[in] idx      index
         */
    FORCEINLINE void getLevelIdx(int current, unsigned int* idx);

    /**
         * sprout algorithm
         *
         * @param[in]  other   another bvh node
         * @param[out] append  front list
         * @param[out] ret     result in triangle pairs
         */
    void sprouting(bvh_node* other, front_list& append, std::vector<TrianglePair>& ret);

    /**
         * sprout algorithm second version
         *
         * @param[in]  other   another bvh node
         * @param[out] append  front list
         * @param[out] ret     result in triangle pairs
         */
    void sprouting2(bvh_node* other, front_list& append, std::vector<TrianglePair>& ret);

    friend class bvh;
};

/**
     * front node in the bvtt front list
     */
class front_node
{
public:
    bvh_node*    _left;   //!< left child
    bvh_node*    _right;  //!< right child
    unsigned int _flag;   //!< vailid or not
    unsigned int _ptr;    //!< self-coliding parent;

    /**
         * constructor
         * 
         * @param[in] l   left child
         * @param[in] r   right child
         * @param[in] ptr parant
         */
    front_node(bvh_node* l, bvh_node* r, unsigned int ptr);
};

/**
     * internal class for bvh construction
     */
class aap
{
public:
    int   _xyz;  //!< type
    float _p;    //!< center

    /**
         * constructor
         * 
         * @param[in] total box in whole
         */
    FORCEINLINE aap(const TAlignedBox3D<float>& total)
    {
        vec3f center = vec3f(total.center().getDataPtr());
        int   xyz    = 2;

        if (total.width() >= total.height() && total.width() >= total.depth())
        {
            xyz = 0;
        }
        else if (total.height() >= total.width() && total.height() >= total.depth())
        {
            xyz = 1;
        }

        _xyz = xyz;
        _p   = center[xyz];
    }

    /**
         * check if center is inside
         * 
         * @param[in] mid mid point
         * @return whether inside
         */
    inline bool inside(const vec3f& mid) const
    {
        return mid[_xyz] > _p;
    }
};

/**
     * bvh class for acceleration
     */
class bvh
{
    int                   _num         = 0;        //!< node number
    bvh_node*             _nodes       = nullptr;  //!< node array pointer
    TAlignedBox3D<float>* s_fboxes     = nullptr;  //!< boxes
    unsigned int*         s_idx_buffer = nullptr;  //!< index buffeer
public:
    /**
         * default constuctor
         */
    bvh(){};

    /**
         * constuctor
         * 
         * @param ms triangle meshes for bvh construction
         */
    template <typename T>
    bvh(std::vector<std::shared_ptr<TriangleMesh<T>>>& ms)
    {
        _num   = 0;
        _nodes = NULL;

        construct<T>(ms);
        reorder();
        resetParents();  //update the parents after reorder ...
    }

    /**
         * constuctor
         *
         * @param[in] ms collision mesh for construction
         */
    bvh(const std::vector<CollisionMesh*>& ms);

    /**
         * refit algorithm, used for bvh construction
         *
         * @param s_fboxes aabb boxes
         */
    void refit(TAlignedBox3D<float>* s_fboxes);

    /**
         * construct algorithm
         *
         * @param[in] ms triangle meshes
         */
    template <typename T>
    void construct(std::vector<std::shared_ptr<TriangleMesh<T>>>& ms)
    {
        TAlignedBox3D<float> total;

        for (int i = 0; i < ms.size(); i++)
            for (int j = 0; j < ms[i]->_num_vtx; j++)
            {
                total += ms[i]->triangleSet->gethPoints()[j];
            }

        _num = 0;
        for (int i = 0; i < ms.size(); i++)
            _num += ms[i]->_num_tri;

        s_fcenters = new vec3f[_num];
        s_fboxes   = new TAlignedBox3D<float>[_num];

        int tri_idx    = 0;
        int vtx_offset = 0;

        for (int i = 0; i < ms.size(); i++)
        {
            for (int j = 0; j < ms[i]->_num_tri; j++)
            {
                TopologyModule::Triangle& f  = ms[i]->triangleSet->getHTriangles()[j];
                Vector3f&                 p1 = ms[i]->triangleSet->gethPoints()[f[0]];
                Vector3f&                 p2 = ms[i]->triangleSet->gethPoints()[f[1]];
                Vector3f&                 p3 = ms[i]->triangleSet->gethPoints()[f[2]];

                s_fboxes[tri_idx] += p1;
                s_fboxes[tri_idx] += p2;
                s_fboxes[tri_idx] += p3;

                auto _s             = p1 + p2 + p3;
                auto sum            = _s.getDataPtr();
                s_fcenters[tri_idx] = vec3f(sum);
                s_fcenters[tri_idx] /= 3.0;
                //s_fcenters[tri_idx] = (p1 + p2 + p3) / double(3.0);
                tri_idx++;
            }
            vtx_offset += ms[i]->_num_vtx;
        }

        aap pln(total);
        s_idx_buffer          = new unsigned int[_num];
        unsigned int left_idx = 0, right_idx = _num;

        tri_idx = 0;
        for (int i = 0; i < ms.size(); i++)
            for (int j = 0; j < ms[i]->_num_tri; j++)
            {
                if (pln.inside(s_fcenters[tri_idx]))
                    s_idx_buffer[left_idx++] = tri_idx;
                else
                    s_idx_buffer[--right_idx] = tri_idx;

                tri_idx++;
            }

        _nodes = new bvh_node[_num * 2 - 1];

        _nodes[0]._box      = total;
        bvh_node* s_current = _nodes + 3;
        if (_num == 1)
            _nodes[0]._child = 0;
        else
        {
            _nodes[0]._child = -1;

            if (left_idx == 0 || left_idx == _num)
                left_idx = _num / 2;
            _nodes[0].left()->construct(s_idx_buffer, left_idx, s_fcenters, s_fboxes, s_current);
            _nodes[0].right()->construct(s_idx_buffer + left_idx, _num - left_idx, s_fcenters, s_fboxes, s_current);
        }

        delete[] s_idx_buffer;
        s_idx_buffer = nullptr;
        delete[] s_fcenters;
        s_fcenters = nullptr;

        refit(s_fboxes);
        //delete[] s_fboxes;
    }

    /**
         *reorder algorithm, used for construction
         */
    void reorder();  // for breath-first refit

    /**
         *reset parent algorithm
         */
    void resetParents();

    /**
         *destructor
         */
    ~bvh()
    {
        if (_nodes)
            delete[] _nodes;
    }

    /**
         * get root node
         * 
         * @return root node
         */
    bvh_node* root()
    {
        return _nodes;
    }

    /**
         * refit algorithm
         *
         * @param[in] ms triangle mesh
         */
    void refit(std::vector<std::shared_ptr<TriangleMesh<DataType3f>>>& ms);

    /**
         * push the host bvh structure to device bvh structure
         *
         * @param[in] isCloth is cloth
         */
    void push2GPU(bool);

    /**
         * collide with other bvh structure
         *
         * @param[in]  other another bvh structure
         * @param[out] f     bvtt front
         * 
         */
    void collide(bvh* other, front_list& f);

    /**
         * collide with other bvh structure
         *
         * @param[in]  other another bvh structure
         * @param[out] ret   triangle pair as result
         */
    void collide(bvh* other, std::vector<TrianglePair>& ret);

    /**
         * self collision detection
         *
         * @param[out] f bvtt front
         * @param[in]  c triangle mesh
         */
    void self_collide(front_list& f, std::vector<std::shared_ptr<TriangleMesh<DataType3f>>>& c);  // hxl

    /**
         * self collision detection
         *
         * @param[out] f bvtt front
         * @param[in]  c collision mesh
         */
    void self_collide(front_list& f, std::vector<CollisionMesh*>& c);
};
}  // namespace PhysIKA