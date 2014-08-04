/*
 * @file rigid_driver_plugin_print.h 
 * @Print plugin of rigid body driver. Print information during simulation
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_PRINT_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_PRINT_H_

namespace Physika{

template <typename Scalar,int Dim>
class RigidDriverPluginPrint: public RigidDriverPlugin<Scalar, Dim>
{
public:
    RigidDriverPluginPrint();
    ~RigidDriverPluginPrint();

    //functions called in driver
    void onBeginFrame(unsigned int frame);
    void onEndFrame(unsigned int frame);
    void onBeginTimeStep(Scalar time, Scalar dt);
    void onEndTimeStep(Scalar time, Scalar dt);

    void onBeginRigidStep(unsigned int step, Scalar dt);//replace the original onBeginTimeStep in rigid body simulation
    void onEndRigidStep(unsigned int step, Scalar dt);//replace the original onEndTimeStep in rigid body simulation

    void onAddRigidBody(RigidBody<Scalar, Dim>* rigid_body);
    void onBeginCollisionDetection();
    void onEndCollisionDetection();
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_PRINT_H_
