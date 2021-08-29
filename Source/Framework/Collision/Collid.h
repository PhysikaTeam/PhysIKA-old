/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: implementation for collision detection cuda functions
 * @version    : 1.0
 */

#pragma once

#include "CollisionDate.h"
#include "Collision.h"

namespace PhysIKA {
/**
 * implementation of mesh collision detection on gpu
 *
 * @param[in]     body          meshes for collision detection
 * @param[out]    contacts      collision triangle pairs
 * @param[out]    CCDtime       process time
 * @param[out]    impact        process time
 * @param[out]    thickness     thickness of face
 */
void body_collide_gpu(
    std::vector<CollisionDate>              bodys,
    std::vector<std::vector<TrianglePair>>& contacts,
    int&                                    CCDtime,
    std::vector<ImpactInfo>&                contact_info,
    float                                   thickness);
}  // namespace PhysIKA