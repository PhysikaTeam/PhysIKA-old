/*
 * @file camera.cpp
 * @Brief a camera class for OpenGL.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include "Physika_Core/Quaternion/quaternion.h"
#include "Physika_Render/Camera/camera.h"

namespace Physika{

template <typename Scalar>
Camera<Scalar>::Camera()
    :camera_position_(0),camera_up_(0,1,0),focus_position_(0,0,-1),fov_(45),view_aspect_(640.0/480.0),near_clip_(1.0),far_clip_(100.0)
{
}

template <typename Scalar>
Camera<Scalar>::Camera(const Vector<Scalar,3> &camera_position, const Vector<Scalar,3> &camera_up, const Vector<Scalar,3> &focus_position,
                       Scalar field_of_view, Scalar view_aspect, Scalar near_clip, Scalar far_clip)
    :camera_position_(camera_position),camera_up_(camera_up),focus_position_(focus_position),
     fov_(field_of_view),view_aspect_(view_aspect),near_clip_(near_clip),far_clip_(far_clip)
{
}

template <typename Scalar>
Camera<Scalar>::Camera(const Camera<Scalar> &camera)
{
    camera_position_ = camera.camera_position_;
    camera_up_ = camera.camera_up_;
    focus_position_ = camera.focus_position_;
    fov_ = camera.fov_;
    view_aspect_ = camera.view_aspect_;
    near_clip_ = camera.near_clip_;
    far_clip_ = camera.far_clip_;
}

template <typename Scalar>
Camera<Scalar>& Camera<Scalar>::operator= (const Camera<Scalar> &camera)
{
    camera_position_ = camera.camera_position_;
    camera_up_ = camera.camera_up_;
    focus_position_ = camera.focus_position_;
    fov_ = camera.fov_;
    view_aspect_ = camera.view_aspect_;
    near_clip_ = camera.near_clip_;
    far_clip_ = camera.far_clip_;
    return *this;
}

template <typename Scalar>
Camera<Scalar>::~Camera()
{
}

template <typename Scalar>
void Camera<Scalar>::look()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov_,view_aspect_,near_clip_,far_clip_);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
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
void Camera<Scalar>::translateUp(Scalar dist)
{
    camera_position_ += dist*camera_up_;
    focus_position_ += dist*camera_up_;
}

template <typename Scalar>
void Camera<Scalar>::translateDown(Scalar dist)
{
    translateUp(-dist);
}

template <typename Scalar>
void Camera<Scalar>::translateRight(Scalar dist)
{
    Vector<Scalar,3> camera_direction = focus_position_ - camera_position_;
    Vector<Scalar,3> axis = camera_direction.cross(camera_up_);
    axis.normalize();
    camera_position_ += dist*axis;
    focus_position_ += dist*axis;
}

template <typename Scalar>
void Camera<Scalar>::translateLeft(Scalar dist)
{
    translateRight(-dist);
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
    if(up==Vector<Scalar,3>(0))
    {
        std::cerr<<"Cannot use zero vector as camera up direction, (0,1,0) is used instead.\n";
        camera_up_ = Vector<Scalar,3>(0,1,0);
    }
    else
    {
        camera_up_ = up;
        camera_up_.normalize();
    }
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

template <typename Scalar>
Scalar Camera<Scalar>::cameraFOV() const
{
    return fov_;
}

template <typename Scalar>
void Camera<Scalar>::setCameraFOV(Scalar fov)
{
    fov_ = fov;
}

template <typename Scalar>
Scalar Camera<Scalar>::cameraAspect() const
{
    return view_aspect_;
}

template <typename Scalar>
void Camera<Scalar>::setCameraAspect(Scalar aspect)
{
    view_aspect_ = aspect;
}

template <typename Scalar>
Scalar Camera<Scalar>::cameraNearClip() const
{
    return near_clip_;
}

template <typename Scalar>
void Camera<Scalar>::setCameraNearClip(Scalar near_clip)
{
    near_clip_ = near_clip;
}

template <typename Scalar>
Scalar Camera<Scalar>::cameraFarClip() const
{
    return far_clip_;
}

template <typename Scalar>
void Camera<Scalar>::setCameraFarClip(Scalar far_clip)
{
    far_clip_ = far_clip;
}

//explicit instantiation
template class Camera<float>;
template class Camera<double>;

} //end of namespace Physika
