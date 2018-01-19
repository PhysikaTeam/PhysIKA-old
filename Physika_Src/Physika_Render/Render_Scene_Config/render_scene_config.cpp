/*
 * @file render_scene_config.cpp 
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

#include "Physika_Render/Lights/light_base.h"
#include "Physika_Render/Render_Task_Base/render_task_base.h"

#include "render_scene_config.h"

namespace Physika{

RenderSceneConfig & RenderSceneConfig::getSingleton()
{
    static RenderSceneConfig singleton;
    return singleton;
}

////////////////////////////////////////////////// screen based render manager ////////////////////////////////////////////////////////////////////////
unsigned int RenderSceneConfig::screenWidth() const
{
    return screen_based_render_manager_.screenWidth();
}

unsigned int RenderSceneConfig::screenHeight() const
{
    return screen_based_render_manager_.screenHeight();
}


////////////////////////////////////////////////// camera operations ////////////////////////////////////////////////////////////////////////

double RenderSceneConfig::cameraRadius() const
{
    return (camera_.cameraFocusPosition() - camera_.cameraPosition()).norm();
}

const Vector3d & RenderSceneConfig::cameraPosition() const
{
    return camera_.cameraPosition();
}

void RenderSceneConfig::setCameraPosition(const Vector3d & position)
{
    camera_.setCameraPosition(position);
}

const Vector3d & RenderSceneConfig::cameraUpDirection() const
{
    return camera_.cameraUpDirection();
}

void RenderSceneConfig::setCameraUpDirection(const Vector3d &up)
{
    camera_.setCameraUpDirection(up);
}

const Vector3d & RenderSceneConfig::cameraFocusPosition() const
{
    return camera_.cameraFocusPosition();
}

void RenderSceneConfig::setCameraFocusPosition(const Vector3d &focus)
{
    camera_.setCameraFocusPosition(focus);
}

double RenderSceneConfig::cameraFOV() const
{
    return camera_.cameraFOV();
}

void RenderSceneConfig::setCameraFOV(double fov)
{
    camera_.setCameraFOV(fov);
}

double RenderSceneConfig::cameraAspect() const
{
    return camera_.cameraAspect();
}

void RenderSceneConfig::setCameraAspect(double aspect)
{
    camera_.setCameraAspect(aspect);
}

double RenderSceneConfig::cameraNearClip() const
{
    return camera_.cameraNearClip();
}

void RenderSceneConfig::setCameraNearClip(double near_clip)
{
    camera_.setCameraNearClip(near_clip);
}

double RenderSceneConfig::cameraFarClip() const
{
    return camera_.cameraFarClip();
}

void RenderSceneConfig::setCameraFarClip(double far_clip)
{
    camera_.setCameraFarClip(far_clip);
}

void RenderSceneConfig::orbitCameraUp(double rad)
{
    camera_.orbitUp(rad);
}

void RenderSceneConfig::orbitCameraDown(double rad)
{
    camera_.orbitDown(rad);
}

void RenderSceneConfig::orbitCameraLeft(double rad)
{
    camera_.orbitLeft(rad);
}

void RenderSceneConfig::orbitCameraRight(double rad)
{
    camera_.orbitRight(rad);
}

void RenderSceneConfig::zoomCameraIn(double dist)
{
    camera_.zoomIn(dist);
}

void RenderSceneConfig::zoomCameraOut(double dist)
{
    camera_.zoomOut(dist);
}

void RenderSceneConfig::yawCamera(double rad)
{
    camera_.yaw(rad);
}

void RenderSceneConfig::pitchCamera(double rad)
{
    camera_.pitch(rad);
}

void RenderSceneConfig::rollCamera(double rad)
{
    camera_.roll(rad);
}

void RenderSceneConfig::translateCameraUp(double dist)
{
    camera_.translateUp(dist);
}

void RenderSceneConfig::translateCameraDown(double dist)
{
    camera_.translateDown(dist);
}

void RenderSceneConfig::translateCameraLeft(double dist)
{
    camera_.translateLeft(dist);
}

void RenderSceneConfig::translateCameraRight(double dist)
{
    camera_.translateRight(dist);
}

////////////////////////////////////////////////// manages lights in scene ///////////////////////////////////////////////////////////////////////////

unsigned int RenderSceneConfig::numLights() const
{
    return light_manager_.numLights();
}

void RenderSceneConfig::pushBackLight(std::shared_ptr<LightBase> light)
{
    light_manager_.insertBack(std::move(light));
}

void RenderSceneConfig::pushFrontLight(std::shared_ptr<LightBase> light)
{
    light_manager_.insertFront(std::move(light));
}

void RenderSceneConfig::insertLightAtIndex(unsigned int index, std::shared_ptr<LightBase> light)
{
    light_manager_.insertAtIndex(index, std::move(light));
}

void RenderSceneConfig::popBackLight()
{
    light_manager_.removeBack();
}

void RenderSceneConfig::popFrontLight()
{
    light_manager_.removeFront();
}

void RenderSceneConfig::removeLightAtIndex(unsigned int index)
{
    light_manager_.removeAtIndex(index);
}

void RenderSceneConfig::removeAllLights()
{
    light_manager_.removeAll();
}

std::shared_ptr<const LightBase> RenderSceneConfig::lightAtIndex(unsigned int index) const
{
    return light_manager_.lightAtIndex(index);
}

std::shared_ptr<LightBase> RenderSceneConfig::lightAtIndex(unsigned int index)
{
    return light_manager_.lightAtIndex(index);
}

int RenderSceneConfig::lightIndex(const std::shared_ptr< const LightBase> &light) const
{
    return light_manager_.lightIndex(light);
}

void RenderSceneConfig::turnAllLightsOn()
{
    light_manager_.turnAllOn();
}

void RenderSceneConfig::turnAllLightsOff()
{
    light_manager_.turnAllOff();
}

void RenderSceneConfig::turnLightOnAtIndex(unsigned int index)
{
    light_manager_.turnLightOnAtIndex(index);
}

void RenderSceneConfig::turnLightOffAtIndex(unsigned int index)
{
    light_manager_.turnLightOffAtIndex(index);
}


////////////////////////////////////////////////// manages render tasks ////////////////////////////////////////////////////////////////////

unsigned int RenderSceneConfig::numRenderTasks() const
{
    return render_task_manager_.numRenderTasks();
}

void RenderSceneConfig::pushBackRenderTask(std::shared_ptr<RenderTaskBase> task)
{
    render_task_manager_.insertBack(task);
}

void RenderSceneConfig::pushFrontRenderTask(std::shared_ptr<RenderTaskBase> task)
{
    render_task_manager_.insertFront(task);
}

void RenderSceneConfig::insertRenderTaskAtIndex(unsigned int index, std::shared_ptr<RenderTaskBase> task)
{
    render_task_manager_.insertAtIndex(index, task);
}

void RenderSceneConfig::popBackRenderTask()
{
    render_task_manager_.removeBack();
}

void RenderSceneConfig::popFrontRenderTask()
{
    render_task_manager_.removeFront();
}

void RenderSceneConfig::removeRenderTaskAtIndex(unsigned int index)
{
    render_task_manager_.removeAtIndex(index);
}

void RenderSceneConfig::removeAllRenderTasks()
{
    render_task_manager_.removeAll();
}

std::shared_ptr<const RenderTaskBase> RenderSceneConfig::getRenderTaskAtIndex(unsigned int index) const
{
    return render_task_manager_.taskAtIndex(index);
}

std::shared_ptr<RenderTaskBase> RenderSceneConfig::getRenderTaskAtIndex(unsigned int index)
{
    return render_task_manager_.taskAtIndex(index);
}

int RenderSceneConfig::getRenderTaskIndex(const std::shared_ptr<const RenderTaskBase> & task) const
{
    return render_task_manager_.taskIndex(task);
}

void RenderSceneConfig::renderAllTasks()
{
    render_task_manager_.renderAllTasks();
}

const Camera<double> & RenderSceneConfig::camera() const
{
    return camera_;
}

Camera<double> & RenderSceneConfig::camera()
{
    return camera_;
}

const RenderTaskManager & RenderSceneConfig::renderTaskManager() const
{
    return render_task_manager_;
}

RenderTaskManager & RenderSceneConfig::renderTaskManager()
{
    return render_task_manager_;
}

const LightManager & RenderSceneConfig::lightManager() const
{
    return light_manager_;
}

LightManager & RenderSceneConfig::lightManager()
{
    return light_manager_;
}

const ScreenBasedRenderManager & RenderSceneConfig::screenBasedRenderManager() const
{
    return screen_based_render_manager_;
}

ScreenBasedRenderManager & RenderSceneConfig::screenBasedRenderManager()
{
    return screen_based_render_manager_;
}

}//end of namespace Physika