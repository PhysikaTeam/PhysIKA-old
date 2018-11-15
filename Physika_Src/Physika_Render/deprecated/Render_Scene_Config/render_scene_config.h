/*
 * @file render_scene_config.h 
 * @Basic RenderSceneConfig class
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

#pragma once

#include <memory>

#include <glm/fwd.hpp>
#include "Physika_Render/Camera/camera.h"
#include "Physika_Render/Render_Task_Manager/render_task_manager.h"
#include "Physika_Render/Lights/light_manager.h"
#include "Physika_Render/Screen_Based_Render_Manager/screen_based_render_manager.h"

namespace Physika{

class RenderTaskBase;
class LightBase;

/*
 * Note: We apply "Singleton Pattern & Facade Pattern" for RenderSceneConfig.
 *       Need further consideration. 
 */
    
class RenderSceneConfig
{
public:
    static RenderSceneConfig & getSingleton();

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //screen based render manager

    unsigned int screenWidth() const;
    unsigned int screenHeight() const;

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //camera operations

    double cameraRadius() const;

    const Vector3d & cameraPosition() const;
    void setCameraPosition(const Vector3d &position);
    const Vector3d & cameraUpDirection() const;
    void setCameraUpDirection(const Vector3d &up);
    const Vector3d & cameraFocusPosition() const;
    void setCameraFocusPosition(const Vector3d &focus);

    double cameraFOV() const;
    void setCameraFOV(double fov);
    double cameraAspect() const;
    void setCameraAspect(double aspect);
    double cameraNearClip() const;
    void setCameraNearClip(double near_clip);
    double cameraFarClip() const;
    void setCameraFarClip(double far_clip);

    void orbitCameraUp(double rad);
    void orbitCameraDown(double rad);
    void orbitCameraLeft(double rad);
    void orbitCameraRight(double rad);
    void zoomCameraIn(double dist);
    void zoomCameraOut(double dist);
    void yawCamera(double rad);
    void pitchCamera(double rad);
    void rollCamera(double rad);
    void translateCameraUp(double dist);
    void translateCameraDown(double dist);
    void translateCameraLeft(double dist);
    void translateCameraRight(double dist);

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //manage render tasks

    unsigned int numRenderTasks() const;                                                             //length of the render queue
    void pushBackRenderTask(std::shared_ptr<RenderTaskBase> task);                                   //insert new task at back of render queue
    void pushFrontRenderTask(std::shared_ptr<RenderTaskBase> task);                                  //insert new task at front of render queue
    void insertRenderTaskAtIndex(unsigned int index, std::shared_ptr<RenderTaskBase> task);          //insert new task before the index-th task
    void popBackRenderTask();                                                                        //remove task at back of render queue
    void popFrontRenderTask();                                                                       //remove task at front of render queue
    void removeRenderTaskAtIndex(unsigned int index);                                                //remove the index-th task in queue
    void removeAllRenderTasks();                                                                     //remove all render tasks
    
    std::shared_ptr<const RenderTaskBase> getRenderTaskAtIndex(unsigned int index) const;            //return pointer to the render task at given index
    std::shared_ptr<RenderTaskBase> getRenderTaskAtIndex(unsigned int index);                        //return pointer to the render task at given index
    int getRenderTaskIndex(const std::shared_ptr<const RenderTaskBase> & task) const;                //return index of task in queue, if task not in queue, return -1

    void renderAllTasks();

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //manage lights
    unsigned int numLights() const;
    void pushBackLight(std::shared_ptr<LightBase> light);
    void pushFrontLight(std::shared_ptr<LightBase> light);
    void insertLightAtIndex(unsigned int index, std::shared_ptr<LightBase>light);
    void popBackLight();
    void popFrontLight();
    void removeLightAtIndex(unsigned int index);
    void removeAllLights();

    std::shared_ptr<const LightBase> lightAtIndex(unsigned int index) const;
    std::shared_ptr<LightBase> lightAtIndex(unsigned int index);
    int lightIndex(const std::shared_ptr< const LightBase> &light) const;
    
    void turnAllLightsOn();                                          //turn all lights on
    void turnAllLightsOff();                                         //turn all lights off
    void turnLightOnAtIndex(unsigned int index);                     //turn light at given index in list ON
    void turnLightOffAtIndex(unsigned int index);                    //turn light at given index in list Off

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //direct access to camera, render task manager, and light manager
    const Camera<double> & camera() const;
    Camera<double> & camera();

    const RenderTaskManager & renderTaskManager() const;
    RenderTaskManager & renderTaskManager();

    const LightManager & lightManager() const;
    LightManager & lightManager();

    const ScreenBasedRenderManager & screenBasedRenderManager() const;
    ScreenBasedRenderManager & screenBasedRenderManager();


private:
    RenderSceneConfig() = default;
    RenderSceneConfig(const RenderSceneConfig & rhs) = default;
    RenderSceneConfig & operator = (const RenderSceneConfig & rhs) = default;
    
    ~RenderSceneConfig() = default;

private:
    Camera<double> camera_;
    RenderTaskManager render_task_manager_;
    LightManager light_manager_;
    ScreenBasedRenderManager screen_based_render_manager_;
};

}//end of namespace Physika