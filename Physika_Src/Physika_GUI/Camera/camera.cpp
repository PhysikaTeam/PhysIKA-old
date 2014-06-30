/*
 * @file camera.cpp
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

#include <GL/gl.h>
#include <GL/glu.h>
#include "Physika_Core/Quaternion/quaternion.h"
#include "Physika_GUI/Camera/camera.h"

namespace Physika{

template <typename Scalar>
Camera<Scalar>::Camera()
    :camera_position_(0),camera_up_(0,1,0),focus_position_(0,0,-1)
{
}

template <typename Scalar>
Camera<Scalar>::Camera(const Vector<Scalar,3> &camera_position, const Vector<Scalar,3> &camera_up, const Vector<Scalar,3> &focus_position)
    :camera_position_(camera_position),camera_up_(camera_up),focus_position_(focus_position)
{
}

template <typename Scalar>
Camera<Scalar>::~Camera()
{
}

template <typename Scalar>
void Camera<Scalar>::update()
{
    gluLookAt(camera_position_[0],camera_position_[1],camera_position_[2],
               focus_position_[0],focus_position_[1],focus_position_[2],
               camera_up_[0],camera_up_[1],camera_up_[2]);
}

template <typename Scalar>
void Camera<Scalar>::orbitUp(Scalar rad)
{
    orbitDown(-rad);
}

template <typename Scalar>
void Camera<Scalar>::orbitDown(Scalar rad)
{
    //first update camera position
    Vector<Scalar,3> camera_direction = focus_position_ - camera_position_;
    Vector<Scalar,3> axis = camera_direction.cross(camera_up_);
    axis.normalize();
    Quaternion<Scalar> quat(axis,rad);
    camera_position_ = quat.rotate(camera_position_);
    //then update up direction
    camera_direction = focus_position_ - camera_position_;
    camera_up_ = axis.cross(camera_direction);
    camera_up_.normalize();
}

template <typename Scalar>
void Camera<Scalar>::orbitLeft(Scalar rad)
{
    orbitRight(-rad);
}

template <typename Scalar>
void Camera<Scalar>::orbitRight(Scalar rad)
{
    Quaternion<Scalar> quat(camera_up_,rad);
    camera_position_ = quat.rotate(camera_position_);
}

template <typename Scalar>
void Camera<Scalar>::zoomIn(Scalar dist)
{
    Vector<Scalar,3> camera_direction = focus_position_ - camera_position_;
    camera_direction.normalize();
    camera_position_ += dist*camera_direction;
}

template <typename Scalar>
void Camera<Scalar>::zoomOut(Scalar dist)
{
    zoomIn(-dist);
}

template <typename Scalar>
void Camera<Scalar>::yaw(Scalar rad)
{
    Quaternion<Scalar> quat(camera_up_,rad);
    Vector<Scalar,3> camera_direction = focus_position_ - camera_position_;
    camera_direction = quat.rotate(camera_direction);
    focus_position_ = camera_position_ + camera_direction;
}

template <typename Scalar>
void Camera<Scalar>::pitch(Scalar rad)
{
    Vector<Scalar,3> camera_direction = focus_position_ - camera_position_;
    Vector<Scalar,3> axis = camera_direction.cross(camera_up_);
    axis.normalize();
    Quaternion<Scalar> quat(axis,rad);
    camera_direction = quat.rotate(camera_direction);
    focus_position_ = camera_position_ + camera_direction;
}

template <typename Scalar>
void Camera<Scalar>::roll(Scalar rad)
{
    Vector<Scalar,3> camera_direction = focus_position_ - camera_position_;
    camera_direction.normalize();
    Quaternion<Scalar> quat(camera_direction,rad);
    camera_up_ = quat.rotate(camera_up_);
}

template <typename Scalar>
void Camera<Scalar>::translate(const Vector<Scalar,3> &vec)
{
    camera_position_ += vec;
    focus_position_ += vec;
}

template <typename Scalar>
const Vector<Scalar,3>& Camera<Scalar>::cameraPosition() const
{
    return camera_position_;
}

template <typename Scalar>
void Camera<Scalar>::setCameraPosition(const Vector<Scalar,3> &position)
{
    camera_position_ = position;
}

template <typename Scalar>
const Vector<Scalar,3>& Camera<Scalar>::cameraUpDirection() const
{
    return camera_up_;
}

template <typename Scalar>
void Camera<Scalar>::setCameraUpDirection(const Vector<Scalar,3> &up)
{
    camera_up_ = up;
}

template <typename Scalar>
const Vector<Scalar,3>& Camera<Scalar>::cameraFocusPosition() const
{
    return focus_position_;
}

template <typename Scalar>
void Camera<Scalar>::setCameraFocusPosition(const Vector<Scalar,3> &focus)
{
    focus_position_ = focus;
}
//explicit instantiation
template class Camera<float>;
template class Camera<double>;

} //end of namespace Physika
