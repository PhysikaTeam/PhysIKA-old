/*
 * @file camera.h 
 * @Brief a camera class for OpenGL.
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

#ifndef PHYSIKA_GUI_CAMERA_CAMERA_H_
#define PHYSIKA_GUI_CAMERA_CAMERA_H_

#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

/*
 * Camera is defined for float and double type. Generally float is sufficient.
 * Camera properties:
 * 1. camera_position_: where the camera is located
 * 2. focus_position_: where the camera is looking at
 * 3. camera_up_: up direction of the camera
 * 4. fov_: field of view of the camera within [0,180]
 * 5. near_clip_: distance from the camera position to the near clip plane
 * 6. far_clip_: distance from the camera position to the far clip plane
 * 7. view_aspect_: aspect of the camera's view (width/height)
 * Usage in OpenGL envrionment:
 * 1. Set view port via glVieport()
 * 2. Call look() method of camera to set the projection and model view transformation
 * Example code:
 *    glMatrixMode(GL_PROJECTION);
 *    glViewport(0,0,width,height);
 *    camera.look();
 */

template <typename Scalar>
class Camera
{
public:
    Camera();   //default camera: positioned at origin, up direction y axis, look into negative z axis (0,0,-1)
                //field of view 45 degrees, view aspect 640/480, near clip 1.0, far clip 100.0
    Camera(const Vector<Scalar,3> &camera_position, const Vector<Scalar,3> &camera_up, const Vector<Scalar,3> &focus_position,
           Scalar field_of_view, Scalar view_aspect, Scalar near_clip, Scalar far_clip);
    Camera(const Camera<Scalar> &camera);
    Camera<Scalar>& operator= (const Camera<Scalar> &camera);
    ~Camera();

    //apply the camera
    void look();

    //rotate the camera around the focus (spherical camera)
    void orbitUp(Scalar rad);
    void orbitDown(Scalar rad);
    void orbitRight(Scalar rad);
    void orbitLeft(Scalar rad);

    //zoom in and zoom out
    void zoomIn(Scalar dist);
    void zoomOut(Scalar dist);

    //change the up direction or look direction (focus position) via yaw, pitch, and roll
    //http://sidvind.com/wiki/Yaw,_pitch,_roll_camera
    void yaw(Scalar rad);
    void pitch(Scalar rad);
    void roll(Scalar rad);

    //translate the camera along with the focus
    void translate(const Vector<Scalar,3> &vec);

    //getters && setters
    const Vector<Scalar,3>& cameraPosition() const;
    void setCameraPosition(const Vector<Scalar,3> &position);
    const Vector<Scalar,3>& cameraUpDirection() const;
    void setCameraUpDirection(const Vector<Scalar,3> &up);
    const Vector<Scalar,3>& cameraFocusPosition() const;
    void setCameraFocusPosition(const Vector<Scalar,3> &focus);
    Scalar cameraFOV() const;
    void setCameraFOV(Scalar fov);
    Scalar cameraAspect() const;
    void setCameraAspect(Scalar aspect);
    Scalar cameraNearClip() const;
    void setCameraNearClip(Scalar near_clip);
    Scalar cameraFarClip() const;
    void setCameraFarClip(Scalar far_clip);
protected:
    Vector<Scalar,3> camera_position_;
    Vector<Scalar,3> camera_up_;
    Vector<Scalar,3> focus_position_;
    Scalar fov_;
    Scalar view_aspect_;
    Scalar near_clip_;
    Scalar far_clip_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GUI_CAMERA_CAMERA_H_
