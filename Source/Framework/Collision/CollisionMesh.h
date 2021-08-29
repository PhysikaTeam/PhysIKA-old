/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: collision mesh class to interpret input data into Collision handlable data structure,
 *               should not be used directly
 * 
 * @version    : 1.0
 */

#pragma once

#include "CollisionVec3.h"

namespace PhysIKA {
/**
     * collision mesh class to interpret input data into Collision handlable data structure
     */
class CollisionMesh
{
public:
    struct tri3f
    {
        unsigned int ids[3];

        tri3f()
        {
            ids[0] = ids[1] = ids[2] = -1;
        }

        tri3f(unsigned int id0, unsigned int id1, unsigned int id2)
        {
            set(id0, id1, id2);
        }

        void set(unsigned int id0, unsigned int id1, unsigned int id2)
        {
            ids[0] = id0;
            ids[1] = id1;
            ids[2] = id2;
        }

        unsigned int id(int i) const
        {
            return ids[i];
        }
        unsigned int id0() const
        {
            return ids[0];
        }
        unsigned int id1() const
        {
            return ids[1];
        }
        unsigned int id2() const
        {
            return ids[2];
        }
    };

    tri3f* _tris = nullptr;

    vec3f* _vtxs = nullptr;  //!< used by time integration
    //vec3f *_ivtxs = nullptr; //!< initial positions
    vec3f* _ovtxs = nullptr;  //!< previous positions

    unsigned int _num_vtx;  //!< number of vertices
    unsigned int _num_tri;  //!< number of triangles

    /**
         * constructor
         * 
         * @param[in] numVtx number of vertices
         * @param[in] numTri number of triangles
         * @param[in] tris   triangle array pointer
         * @param[in] vtxs   vertex array pointer
         */
    CollisionMesh(unsigned int numVtx, unsigned int numTri, tri3f* tris, vec3f* vtxs)
    {
        _num_vtx = numVtx;
        _num_tri = numTri;

        _tris = tris;
        _vtxs = vtxs;
        //_ivtxs = new vec3f[numVtx];
        _ovtxs = new vec3f[numVtx];
    }

    /**
         * destructor
         */
    ~CollisionMesh()
    {
        delete[] _tris;
        delete[] _vtxs;
        //delete [] _ivtxs;
        delete[] _ovtxs;
    }

    /**
         * get the number of vertices
         * 
         * @return the number of vertices
         */
    unsigned int getNbVertices() const
    {
        return _num_vtx;
    }

    /**
         * get the number of triangle faces
         *
         * @return the number of triangle faces
         */
    unsigned int getNbFaces() const
    {
        return _num_tri;
    }

    /**
         * get the vertex array pointer
         *
         * @return the vertex array pointer
         */
    vec3f* getVtxs() const
    {
        return _vtxs;
    }

    /**
         * get the previous vertex array pointer
         *
         * @return the previous vertex array pointer
         */
    vec3f* getOVtxs() const
    {
        return _ovtxs;
    }
};
}  // namespace PhysIKA