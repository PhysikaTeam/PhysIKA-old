/*
 * @file rigid_driver_plugin_motion.h 
 * @Customize the motion of objects
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_MOTION_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_MOTION_H_

#include <map>

namespace Physika{

template <typename Scalar>
class RigidPluginMotionCustomization
{
public:
    RigidPluginMotionCustomization();
    ~RigidPluginMotionCustomization();

    void setRigidBody(RigidBody<Scalar, 3>* rigid_body);
    void setConstantTranslation(const Vector<Scalar, 3>& velocity);
    void setConstantRotation(const Vector<Scalar, 3>& velocity);
    void setPeriodTranslation(const Vector<Scalar, 3>& velocity, Scalar period);
    void setPeriodRotation(const Vector<Scalar, 3>& velocity, Scalar period);

    void update(Scalar dt, Scalar current_time);

protected:
    RigidBody<Scalar, 3>* rigid_body_;
    Vector<Scalar, 3> constant_translation_velocity_;
    Vector<Scalar, 3> constant_rotation_velocity_;
    Vector<Scalar, 3> period_translation_velocity_;
    Vector<Scalar, 3> period_rotation_velocity_;
    Scalar translation_period_;
    Scalar rotation_period_;
};

template <typename Scalar>
class RigidDriverPluginMotion: public RigidDriverPlugin<Scalar, 3>
{
public:
    RigidDriverPluginMotion();
    ~RigidDriverPluginMotion();

    //functions called in driver
    void onBeginFrame(unsigned int frame);
    void onEndFrame(unsigned int frame);
    void onBeginTimeStep(Scalar time, Scalar dt);
    void onEndTimeStep(Scalar time, Scalar dt);

    void onBeginRigidStep(unsigned int step, Scalar dt);//replace the original onBeginTimeStep in rigid body simulation
    void onEndRigidStep(unsigned int step, Scalar dt);//replace the original onEndTimeStep in rigid body simulation

    void onAddRigidBody(RigidBody<Scalar, 3>* rigid_body);
    void onBeginCollisionDetection();
    void onEndCollisionDetection();

    void setConstantTranslation(RigidBody<Scalar, 3>* rigid_body, const Vector<Scalar, 3>& velocity);//WARNING! rigid_body will be set fixed after calling this function
    void setConstantRotation(RigidBody<Scalar, 3>* rigid_body, const Vector<Scalar, 3>& velocity);//WARNING! rigid_body will be set fixed after calling this function
    void setPeriodTranslation(RigidBody<Scalar, 3>* rigid_body, const Vector<Scalar, 3>& velocity, Scalar period);//WARNING! rigid_body will be set fixed after calling this function
    void setPeriodRotation(RigidBody<Scalar, 3>* rigid_body, const Vector<Scalar, 3>& velocity, Scalar period);//WARNING! rigid_body will be set fixed after calling this function

protected:
    Scalar time_;
    std::map<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> > customized_motions_;

};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_MOTION_H_
