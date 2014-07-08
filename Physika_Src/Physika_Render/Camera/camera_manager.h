/*
 * @file camera_manager.h 
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

#ifndef PHYSIKA_RENDER_CAMERA_CAMERA_MANAGER_H_
#define PHYSIKA_RENDER_CAMERA_CAMERA_MANAGER_H_

#include <list>

namespace Physika{

class CameraBase;

/*
 * CameraManager maintains a list of cameras, but only one of them is active at the same time
 * If not specified, the first camera will be the active camera
 * Only after calling applyCamera() of CameraManager, will the active camera be put into use
 *
 */

class CameraManager
{
public:
    CameraManager();
    ~CameraManager();
    unsigned int numCameras() const;
    void addCamera(CameraBase *camera);

    /*
     * remove camera from the list
     * if the camera removed is the active camera, the first camera will be the new active camera
     */
    void removeCamera(unsigned int index);  //remove camera at given index
    void removeCamera(CameraBase *camera);  //remove the specific camera, ignore if it's not in the list

    /* set the given camera as the active camera
     * if the camera is not in the list, insert into the list and make it the active camera
     */
    void setActiveCamera(CameraBase *camera);
    void setActiveCamera(unsigned int index);  //set the camera at given index as the active camera

    //return pointer to the active camera, if the camera list is empty, return NULL
    CameraBase* activeCamera() const;

    void applyCamera() const;  //apply the active camera
protected:
    unsigned int active_camera_index_;
    std::list<CameraBase*> camera_list_;
};

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_CAMERA_CAMERA_MANAGER_H_
