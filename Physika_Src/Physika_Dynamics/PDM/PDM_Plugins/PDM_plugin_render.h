/*
 * @file PDM_plugin_render.h 
 * @brief render plugins class for PDM drivers.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_RENDER_H
#define PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_RENDER_H

#include "Physika_Render/ColorBar/ColorMap/color_map.h"
#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_base.h"
#include "Physika_Dynamics/PDM/PDM_base.h"

namespace Physika{

class GlutWindow;
template <typename Scalar, int Dim> class PDMImpactMethodBase;

template <typename Scalar, int Dim>
class PDMPluginRender: public PDMPluginBase<Scalar, Dim>
{
public:
    PDMPluginRender();
    ~PDMPluginRender();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame);
    virtual void onEndFrame(unsigned int frame);
    virtual void onBeginTimeStep(Scalar time, Scalar dt);
    virtual void onEndTimeStep(Scalar time, Scalar dt);

    void enableCaptureScreen();
    void disableCaptureScreen();

    //setter & getter
    GlutWindow * window();
    void setWindow(GlutWindow * window);

    Scalar pointSize() const;
    void   setPointSize(Scalar point_size);
    Scalar velocityScale() const;
    void   setVelocityScale(Scalar velocity_scale);

    void setFloorPos(Scalar pos);

    void setImpactMethod(PDMImpactMethodBase<Scalar, Dim> * impact_method);
    void setBulletPos(const Vector<Scalar,Dim> & bullet_pos);
    void setBulletRadius(Scalar bullet_radius);
    void setBulletVelocity(const Vector<Scalar,Dim> & bullet_velocity);

    void setCriticalStretchVec(const std::vector<Scalar> & s_vec);

    std::vector<unsigned int > & specifiedIdxVec();
    const std::vector<unsigned int> & specifiedIdxVec() const;

    // note: this function should be called after dirver_base has been set.
    void addIdxToSpecifiedIdxVec(unsigned int par_idx);
    void addIdxToSpecifiedParVec(unsigned int par_idx);

    void enableRenderParticle();
    void enableRenderFamily();
    void enableRenderVelocity();
    void enableRenderFloor();
    void enableRenderSpecifiedParticle();
    void enableRenderBullet();
    void enableRenderCriticalStretch();
    void enableRenderCurSize();
    void enableRenderMesh();

    void disableRenderParticle();
    void disableRenderFamily();
    void disableRenderVelocity();
    void disableRenderFloor();
    void disableRenderSpecifiedParticle();
    void disableRenderBullet();
    void disableRenderCriticalStretch();
    void disableRenderCurSize();
    void disableRenderMesh();

protected:
    static void idleFunction(void);
    static void displayFunction(void);
    static void keyboardFunction(unsigned char key, int x, int y);

    void activateCurrentInstance();

    void renderParticles();
    void renderParticleVelocity();
    void renderParticleFamily(unsigned int par_idx);
    void renderFloor();
    void renderSpeicifiedParticles();
    void renderBullet();
    void renderMesh();
    void renderCompressionPlane();

protected:
    GlutWindow * window_;
    static PDMPluginRender<Scalar, Dim> * active_instance_;

    bool auto_capture_frame_;

    bool render_particle_;
    bool render_specified_family_;
    std::vector<unsigned int> specified_idx_vec_; // the idx specified to render its family

    bool render_specified_particle_;
    std::vector<unsigned int> specified_par_vec_; // the particle specified to render

    bool render_velocity_;
    Scalar velocity_scale_;

    Scalar floor_pos_;
    bool render_floor_;

    PDMImpactMethodBase<Scalar, Dim> * impact_method_; //used to render bullet
    Vector<Scalar,Dim> bullet_velocity_;
    Scalar bullet_radius_;              
    Vector<Scalar,Dim> bullet_pos_;
    bool render_bullet_;

    ColorMap<Scalar> color_map_;

    // need further consideration
    std::vector<Scalar> critical_s_vec_; // critical stretch vec used to render
    bool render_s_;

    // need further consideration
    bool render_cur_size_;

    // need further consideration
    bool render_mesh_;

    //need further consideration: only used for compression demo
    /////////////////////////////////////////////////////////////////////
    bool render_compression_plane_;        //default: false
    Scalar up_plane_pos_;                         //default: 0.0
    Scalar down_plane_pos_;                       //default: 0.0
    Scalar up_plane_vel_;                         //default: 0.0
    Scalar down_plane_vel_;                       //default: 0.0
    /////////////////////////////////////////////////////////////////////

    Scalar point_size_;
};

} // end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_RENDER_H