/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: data structure to store collision info
 *               should not be used directly
 *
 * @version    : 1.0
 */

#pragma once

#include "CollisionMesh.h"

namespace PhysIKA {
/**
     * Store collision mesh and if self collision check is needed, internal data structure
     */
struct CollisionDate
{
    CollisionDate(CollisionMesh* m, bool flag)
        : ms(m), enable_selfcollision(flag) {}
    CollisionMesh* ms;                    //!< pointer to the mesh
    bool           enable_selfcollision;  //!< enable self collision
};

/**
     * Store the collision impact information
     */
struct ImpactInfo
{
    /*
         * constructor, used internally
         */
    ImpactInfo(int fid1, int fid2, int vf_ee, int v, int v2, int v3, int v4, float d, float t, int CCD)
    {
        f_id[0] = fid1;
        f_id[1] = fid2;

        IsVF_OR_EE = vf_ee;

        vertex_id[0] = v;
        vertex_id[1] = v2;
        vertex_id[2] = v3;
        vertex_id[3] = v4;

        dist = d;
        time = t;

        CCDres = CCD;
    }

    int f_id[2];  //<! face id

    int IsVF_OR_EE;  //<! 0:vf 1:ee

    int vertex_id[4];  //<! vertices ids

    float dist;  //<! distance
    float time;  //<! time

    int CCDres;  //<! ccd results
};

}  // namespace PhysIKA