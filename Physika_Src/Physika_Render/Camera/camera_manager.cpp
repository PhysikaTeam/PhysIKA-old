/*
 * @file camera_manager.cpp 
 * @Brief camera manager class, manages a couple of cameras, only one can be active at the same time.
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

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/Camera/camera.h"
#include "Physika_Render/Camera/camera_manager.h"

namespace Physika{

CameraManager::CameraManager()
    :active_camera_index_(0)
{
}

CameraManager::~CameraManager()
{
}

unsigned int CameraManager::numCameras() const
{
    return camera_list_.size();
}

void CameraManager::addCamera(CameraBase *camera)
{
    if(camera==NULL)
        std::cerr<<"Specified camera is NULL, operation ignored.\n";
    else
        camera_list_.push_back(camera);
}

void CameraManager::removeCamera(unsigned int index)
{
    if(index<0||index>=camera_list_.size())
    {
        std::cerr<<"Camera index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    if(active_camera_index_ == index)
    {
        std::cerr<<"The camera to delete is the active camera, new active camera is the first camera in the list.\n";
        active_camera_index_ = 0;
    }
    std::list<CameraBase*>::iterator iter = camera_list_.begin();
    for(unsigned int i = 0; i < index; ++i)
        ++iter;
    camera_list_.erase(iter);
}

void CameraManager::removeCamera(CameraBase *camera)
{
    if(camera==NULL)
        std::cerr<<"Specified camera is NULL, operation ignored.\n";
    else
    {
        std::list<CameraBase*>::iterator iter = camera_list_.begin();
        for(unsigned int i = 0; i < camera_list_.size(); ++i)
        {
            if(*iter == camera)
            {
                if(active_camera_index_ == i)
                {
                    std::cerr<<"The camera to delete is the active camera, new active camera is the first camera in the list.\n";
                    active_camera_index_ = 0;
                }
                camera_list_.erase(iter);
                return;
            }
            ++iter;
        }
        //camera not in the list, print warning
        std::cerr<<"Specified camera not in the list, operation ignored.\n";
    }
}

void CameraManager::setActiveCamera(unsigned int index)
{
    if(index<0||index>=camera_list_.size())
    {
        std::cerr<<"Camera index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    active_camera_index_ = index;
}

void CameraManager::setActiveCamera(CameraBase *camera)
{
    if(camera==NULL)
        std::cerr<<"Cannot set NULL camera as the active camera, operation ignored.\n";
    else
    {
        std::list<CameraBase*>::iterator iter = camera_list_.begin();
        for(unsigned int i = 0; i < camera_list_.size(); ++i)
        {
            if(*iter == camera)
            {
                active_camera_index_ = i;
                return;
            }
            ++iter;
        }
        //camera not in the list, insert it into the list and make it active
        std::cerr<<"Specified camera is not in the camera list, it's now inserted into the list and set active.\n";
        camera_list_.push_back(camera);
        active_camera_index_ = camera_list_.size() - 1;
    }
}

CameraBase* CameraManager::activeCamera() const
{
    if(camera_list_.empty())
        return NULL;
    PHYSIKA_ASSERT(active_camera_index_>=0);
    PHYSIKA_ASSERT(active_camera_index_<camera_list_.size());
    std::list<CameraBase*>::const_iterator iter = camera_list_.begin();
    for(unsigned int i = 0; i < active_camera_index_; ++i)
        ++iter;
    return *iter;
}

void CameraManager::applyCamera() const
{
    CameraBase *active_camera = this->activeCamera();
    PHYSIKA_ASSERT(active_camera);
    active_camera->look();
}

} //end of namespace Physika
