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
 */

template <typename Scalar>
class Camera
{
public:
    Camera();   //default camera: positioned at origin, up direction y axis, look into negative z axis (0,0,-1)
    Camera(const Vector<Scalar,3> &camera_position, const Vector<Scalar,3> &camera_up, const Vector<Scalar,3> &focus_position);
    ~Camera();

    //update the camera with new position, up direction, and focus position
    void update();

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
protected:
    Vector<Scalar,3> camera_position_;
    Vector<Scalar,3> camera_up_;
    Vector<Scalar,3> focus_position_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GUI_CAMERA_CAMERA_H_
