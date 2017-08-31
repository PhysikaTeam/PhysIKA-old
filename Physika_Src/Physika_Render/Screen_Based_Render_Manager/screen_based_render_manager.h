/*
* @file screen_based_render_manager.h
* @Brief class ScreenBasedRenderManager
* @author Wei Chen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#ifndef PHYSIKA_RENDER_SCREEN_BASED_RENDER_MANAGER_SCREEN_BASED_RENDER_MANAGER_H
#define PHYSIKA_RENDER_SCREEN_BASED_RENDER_MANAGER_SCREEN_BASED_RENDER_MANAGER_H

#include <list>

#include "Physika_Core/Vectors/vector_3d.h"

#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/Color/color.h"

#include "Physika_Render/OpenGL_Shaders/shader_program.h"
#include "Physika_Render/Screen_Based_Render_Manager/Fluid_Render/fluid_render.h"
#include "Physika_Render/Screen_Based_Render_Manager/Shadow_Map/shadow_map.h"

namespace Physika{

class GlutWindow;
class RenderBase;
class ShaderProgram;

class ScreenBasedRenderManager 
{
public:

    //ScreenBasedRenderManager();
    //ScreenBasedRenderManager(unsigned int screen_width, unsigned int screen_height);
    ScreenBasedRenderManager(GlutWindow * glut_window);

    ~ScreenBasedRenderManager();

    ScreenBasedRenderManager(const ScreenBasedRenderManager &) = delete;
    ScreenBasedRenderManager & operator = (const ScreenBasedRenderManager &) = delete;

    void render();
    void drawShapes();

    void applyShadowMap();
    void applyShadowMapWithSpecifiedShaderProgram(ShaderProgram & shader_program);

    //getters
    const GlutWindow * glutWindow() const;

    const Vector<float, 3> & lightPos() const;
    const Vector<float, 3> & lightTarget() const;
    float lightFov() const;
    float lightSpotMin() const;
    float lightSpotMax() const;

    //setters
    void addRender(RenderBase * render);
    void addPlane(const Vector<float, 4> & plane);

    void setGlutWindow(GlutWindow * glut_window);
    void setFluidRender(FluidRender * fluid_render);

    //set light parameters
    void setLightPos(const Vector<float, 3> & light_pos);
    void setLightTarget(const Vector<float, 3> & light_target);
    void setLightFov(float light_fov);
    void setLightSpotMin(float light_spot_min);
    void setLightSpotMax(float light_spot_max);
    void setFogDistance(float fog_distance);
    void setShadowBias(float shadow_bias);

    GLuint msaaFBO() const;
    const ShadowMap & shadowMap() const;

private:
    void initMsaaFrameBuffer();
    void destroyMsaaFrameBuffer();

private:

    void drawPlanes();
    void drawPlane(const Vector<float, 4> & plane);
    void getBasisFromNormalVector(const Vector<float, 3> & w, Vector<float, 3> & u, Vector<float, 3> & v);

    void beginFrame(const Color<float> & clear_color);

    void createShadowMap();
    void createCameraMap();

    void endFrame();

private:
    void beginLighting();
    void endLighting();

private:
    //need further consideration
    void setCameraProjAndModelViewMatrix();
    void setLightProjAndModelViewMatrix();
    void setProjAndModelViewMatrix(GLfloat * proj_mat, GLfloat * model_view_mat);

private:

    GlutWindow * glut_window_ = nullptr;
    unsigned int screen_width_;
    unsigned int screen_height_;

    int msaa_samples_ = 8;

    GLuint msaa_FBO_ = 0;
    GLuint msaa_color_RBO_ = 0;
    GLuint msaa_depth_RBO_ = 0;

    ShaderProgram shape_program_;

    //render
    FluidRender * fluid_render_ = nullptr;
    std::list<RenderBase *> render_list_;

    //planes
    std::vector<Vector<float, 4> > plans_;

    //shader map
    ShadowMap shadow_map_;
    float shadow_bias_ = 0.05f;

    //light parameters
    Vector<float, 3> light_pos_;
    Vector<float, 3> light_target_;
    float            light_fov_ = 45.0f;
    float            light_spot_min_ = 0.5f;
    float            light_spot_max_ = 1.0f;

    float fog_distance_ = 0.0f;
    
};

}//end of namespace Physika


#endif //PHYSIKA_RENDER_SCREEN_BASED_RENDER_MANAGER_SCREEN_BASED_RENDER_MANAGER_H