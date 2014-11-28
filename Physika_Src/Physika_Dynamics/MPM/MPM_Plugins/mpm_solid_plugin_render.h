/*
 * @file mpm_solid_plugin_render.h 
 * @brief plugin for real-time render of drivers derived from MPMSolid.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_RENDER_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_RENDER_H_

#include "Physika_Core/Timer/timer.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"

namespace Physika{

class GlutWindow;

/*
 * MPMSolidPluginRender: plugin that provides a real-time window
 * for drivers derived from MPMSolid
 * 
 * Keyboard bindings:
 * 1. ESC: quit
 * 2. s: save current screen
 * 3. S: save screen every simulation frame
 * 4. p: display particle
 * 5. P: display particle velocity
 * 6. g: display grid
 * 7. G: display grid velocity
 * 8. m: render particle as point or sphere
 * 9. SPACE: pause/continue simulation
 * 10. c: display particle domain (if it's CPDI MPM)
 * 11. i: increase velocity display scale
 * 12. d: decrease velocity display scale
 */

template <typename Scalar, int Dim>
class MPMSolidPluginRender: public MPMSolidPluginBase<Scalar,Dim>
{
public:
    MPMSolidPluginRender();
    ~MPMSolidPluginRender();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame);
    virtual void onEndFrame(unsigned int frame);
    virtual void onBeginTimeStep(Scalar time, Scalar dt);
    virtual void onEndTimeStep(Scalar time, Scalar dt);
    virtual MPMSolid<Scalar,Dim>* driver();
    virtual void setDriver(DriverBase<Scalar> *driver);

    //MPM Solid driver specific virtual methods
    virtual void onRasterize();
    virtual void onSolveOnGrid(Scalar dt);
    virtual void onResolveContactOnGrid(Scalar dt);
    virtual void onResolveContactOnParticles(Scalar dt);
    virtual void onUpdateParticleInterpolationWeight();
    virtual void onUpdateParticleConstitutiveModelState(Scalar dt);
    virtual void onUpdateParticleVelocity();
    virtual void onApplyExternalForceOnParticles(Scalar dt);
    virtual void onUpdateParticlePosition(Scalar dt);

    //setters&&getters
    GlutWindow* window();
    void setWindow(GlutWindow *window);

protected:
    static void idleFunction(void); 
    static void displayFunction(void);
    static void keyboardFunction(unsigned char key, int x, int y);
    void activateCurrentInstance();  //activate this instance, called in constructor
    
    void renderParticles();
    void renderGrid();
    void renderParticleVelocity();
    void renderGridVelocity();
    void renderParticleDomain();

protected:
    GlutWindow *window_;
    static MPMSolidPluginRender<Scalar,Dim> *active_instance_;  //current active instance
    //pause
    bool pause_simulation_;
    //determine if simulation is finished (e.g., max frame is reached)
    bool simulation_finished_;
    //render switch
    bool render_particle_;
    bool render_grid_;
    bool render_particle_velocity_;
    bool render_grid_velocity_;
    bool render_particle_domain_;
    unsigned int particle_render_mode_;  // 0: particle rendered as point; otherwise: particle rendered as sphere
    //for render velocities
    Scalar velocity_scale_;
    //screen capture switch
    bool auto_capture_frame_; //screen capture each frame's screen
    //timer to compute timing information of simulation
    Timer timer_;  
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_RENDER_H_
