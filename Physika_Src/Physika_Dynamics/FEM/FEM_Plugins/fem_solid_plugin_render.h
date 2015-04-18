/*
 * @file fem_solid_plugin_render.h 
 * @brief plugin for real-time render of drivers derived from FEMSolid.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_PLUGINS_FEM_SOLID_PLUGIN_RENDER_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_PLUGINS_FEM_SOLID_PLUGIN_RENDER_H_

#include "Physika_Core/Timer/timer.h"
#include "Physika_Dynamics/FEM/fem_solid.h"
#include "Physika_Dynamics/FEM/FEM_Plugins/fem_solid_plugin_base.h"

namespace Physika{

class GlutWindow;
template <typename Scalar, int Dim> class VolumetricMeshRender;

/*
 * FEMSolidPluginRender: plugin that provides a real-time window
 * for drivers derived from FEMSolid
 * 
 * Keyboard bindings:
 * 1. ESC: quit
 * 2. s: save current screen
 * 3. S: save screen every simulation frame
 * 4. v: display vertex velocity
 * 5. SPACE: pause/continue simulation
 * 6. i: increase velocity display scale
 * 7. d: decrease velocity display scale
 */

template <typename Scalar, int Dim>
class FEMSolidPluginRender: public FEMSolidPluginBase<Scalar,Dim>
{
public:
    //constructors && deconstructors
    FEMSolidPluginRender();
    virtual ~FEMSolidPluginRender();

    //functions called in driver
    virtual void onBeginFrame(unsigned int frame);
    virtual void onEndFrame(unsigned int frame);
    virtual void onBeginTimeStep(Scalar time, Scalar dt);
    virtual void onEndTimeStep(Scalar time, Scalar dt); 

    //setters&&getters
    GlutWindow* window();
    void setWindow(GlutWindow *window);

protected:
    static void idleFunction(void);
    static void displayFunction(void);
    static void keyboardFunction(unsigned char key, int x, int y);
    void activateCurrentInstance(); //activate this instance, called in constructor

    void renderVertexVelocity();
    void renderVolumetricMesh();

protected:
    GlutWindow *window_;
    static FEMSolidPluginRender<Scalar,Dim> *active_instance_; //current active instance
    VolumetricMeshRender<Scalar,Dim> *volumetric_mesh_render_; //for rendering volumetric mesh
    bool pause_simulation_;
    bool simulation_finished_;
    bool render_velocity_;
    Scalar velocity_scale_;
    bool auto_capture_frame_;
    Timer timer_;
    Scalar total_time_; //total time spent on simulation
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_PLUGINS_FEM_SOLID_PLUGIN_RENDER_H_
