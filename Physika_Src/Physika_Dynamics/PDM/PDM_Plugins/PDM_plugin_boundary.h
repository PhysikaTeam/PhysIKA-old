/*
 * @file PDM_plugin_boundary.h 
 * @brief boundary plugins class for PDM drivers. It will exert boundary conditions to PDM systems
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

// Note: the boundary plugin are still preliminary, 
// which means this class are not complete, and still needs further consideration.
// the user should use it with greate caution.

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_BOUNDARY_H
#define PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_BOUNDARY_H

#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_base.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar, int Dim>
class PDMPluginBoundary: public PDMPluginBase<Scalar, Dim>
{
public:
    PDMPluginBoundary();
    ~PDMPluginBoundary();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame);
    virtual void onEndFrame(unsigned int frame);
    virtual void onBeginTimeStep(Scalar time, Scalar dt);
    virtual void onEndTimeStep(Scalar time, Scalar dt);

    void addFixedParticle(unsigned int par_idx);
    void addEliminateOffsetParticle(unsigned int par_idx);
    void addParticleAdditionalForce(unsigned int par_idx, const Vector<Scalar, Dim> & f);
    void setFloorPos(Scalar pos);
    void setRecoveryCoefficient(Scalar recovery_coefficient);
    void setEliminateAxis(bool x_axis = true, bool y_axis = false, bool z_axis = false);

    void setBulletPos(const Vector<Scalar,Dim> & bullet_pos);
    void setBulletVelocity(const Vector<Scalar, Dim> & bullet_velocity);
    void setBulletRadius(Scalar bullet_radius);
    void setBulletKs(Scalar Ks);

    void setSpherePos(const Vector<Scalar,Dim> & sphere_pos);
    void setSphereVelocity(const Vector<Scalar, Dim> & sphere_velocity);
    void setSphereRadius(Scalar sphere_radius);

    void enableFloor();
    void disableFloor();
    void enableFix();
    void disableFix();
    void enableAdditionalForce();
    void disableAdditionalForce();
    void enableEliminateOffset();
    void disableEliminateOffset();
    void enableBullet();
    void disableBullet();
    void enableSphere();
    void disableSphere();

protected:
    void applyBullet();
    void applySphere();

protected:
    Scalar floor_pos_; // floor y position

    // the idx vec to specify the particle to be fixed
    std::vector<unsigned int> fixed_idx_vec_;

    // the idx vec to specify the particle with additional force
    std::vector<unsigned int> force_idx_vec_;
    std::vector<Vector<Scalar,Dim> > force_vec_;
    
    std::vector<unsigned int> eliminate_offset_vec_;

    Scalar recovery_coefficient_;

    bool enable_eliminate_offset_;
    bool eliminate_x_offset_;
    bool eliminate_y_offset_;
    bool eliminate_z_offset_;

    bool enable_floor_;
    bool enable_fix_;
    bool enable_force_;

    bool enable_bullet_;
    Vector<Scalar,Dim> bullet_velocity_;
    Scalar bullet_radius_; //projectile radius
    Vector<Scalar,Dim> bullet_pos_;
    Scalar Ks_;

    bool enable_sphere_;
    Vector<Scalar,Dim> sphere_velocity_;
    Scalar sphere_radius_;
    Vector<Scalar,Dim> sphere_pos_;
};

}// end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_BOUNDARY_H
