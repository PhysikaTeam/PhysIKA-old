/******************************************************************************
Copyright (c) 2007 Bart Adams (bart.adams@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. The authors shall be
acknowledged in scientific publications resulting from using the Software
by referencing the ACM SIGGRAPH 2007 paper "Adaptively Sampled Particle
Fluids".

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
******************************************************************************/

#include <Windows.h>
#include <GL/gl.h>
#include "Camera.h"
#include <iostream>
#include <math.h>
using namespace std;


Camera::Camera() {
    _eye = Vectorold3f(0,0,3);
    _light = Vectorold3f(0,0,3);
    _rotation = 0;
    _rotation_axis = Vectorold3f(0,1,0);
    _fov = 0.90f;
}


void Camera::SetGL(float neardist, float fardist, float width, float height) {
    float diag = sqrt(width*width+height*height);
    float top = height/diag * 0.5f*_fov*neardist;
    float bottom = -top;
    float right = width/diag* 0.5f*_fov*neardist;
    float left = -right;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(left, right, bottom, top, neardist, fardist);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(180.0f/3.1415926*_rotation, _rotation_axis.x, _rotation_axis.y, _rotation_axis.z);
    glTranslatef(-_eye.x, -_eye.y, -_eye.z);

    GLfloat pos[] = {_light.x, _light.y, _light.z,1};
    glLightfv(GL_LIGHT0, GL_POSITION, pos);

    _width = (int)width;
    _height = (int)height;
    _pixelarea = 4*right*top/(width*height);
    _neardist = neardist;
    _fardist = fardist;
    _right = right;
}

int Camera::GetWidth() const {
    return _width;
}

int Camera::GetHeight() const {
    return _height;
}

float Camera::GetPixelArea() const {
    return _pixelarea;
}

Vectorold3f Camera::GetEye() const {
    return _eye;
}

Transform3D<float> Camera::GetCombinedMatrix2() const {
    float neardist = _neardist;
    float fardist = _fardist;
    float width = _width;
    float height = _height;
    // set up modelview
    Vectorold3f n; Vectorold3f u; Vectorold3f v;
    GetCoordSystem(n,v,u);
    Transform3D<float> modelview(u.x, u.y, u.z, -_eye.Dot(u),
                         v.x, v.y, v.z, -_eye.Dot(v),
                         n.x, n.y, n.z, -_eye.Dot(n),
                         0.0f, 0.0f, 0.0f, 1.0f);
    // set up projection
    float diag = sqrt(width*width+height*height);
    float top = height/diag * 0.5f*_fov*neardist;
    float bottom = -top;
    float right = width/diag* 0.5f*_fov*neardist;
    float left = -right;
    Transform3D<float> projection(
        -2*neardist/(left-right), 0.0f, (right+left)/(right-left), 0.0f,
        0.0f, -2*neardist/(bottom-top), (top+bottom)/(top-bottom), 0.0f,
        0.0f, 0.0f, (neardist+fardist)/(neardist-fardist), (-2*neardist*fardist)/(fardist-neardist),
        0.0f, 0.0f, -1.0f, 0.0f);


    Transform3D<float> combined = projection*modelview;
    return combined;
}

Transform3D<float> Camera::GetCombinedMatrix() const {
    float neardist = _neardist;
    float fardist = _fardist;
    float width = _width;
    float height = _height;
    // set up modelview
    Vectorold3f n; Vectorold3f u; Vectorold3f v;
    GetCoordSystem(n,v,u);
    Transform3D<float> modelview(u.x, u.y, u.z, -_eye.Dot(u),
                         v.x, v.y, v.z, -_eye.Dot(v),
                         n.x, n.y, n.z, -_eye.Dot(n),
                         0.0f, 0.0f, 0.0f, 1.0f);
    // set up projection
    float diag = sqrt(width*width+height*height);
    float top = height/diag * 0.5f*_fov*neardist;
    float bottom = -top;
    float right = width/diag* 0.5f*_fov*neardist;
    float left = -right;


    Transform3D<float> projection(
        2*neardist/(left-right), 0.0f, (right+left)/(right-left), 0.0f,
        0.0f, 2*neardist/(bottom-top), (top+bottom)/(top-bottom), 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);

    Transform3D<float> combined = projection*modelview;
    int hw = (int)width/2; int hh = (int)height/2;
    combined.x[0] *= hw; combined.x[4] *= hw; combined.x[8] *= hw; combined.x[12] *= hw;
    combined.x[1] *= hh; combined.x[5] *= hh; combined.x[9] *= hh; combined.x[13] *= hh;
    return combined;
}

void Camera::Rotate(Quat1f &rotquat) {
    // set up orthogonal camera system
    Quat1f q(_rotation, _rotation_axis);
    q.x = -q.x;
    Vectorold3f viewdir(0,0,-1);
    q.Rotate(viewdir);
    // end set up orth system
    //   q = Quat1f(angle, axis);
    q = rotquat;
    Quat1f currq(_rotation, _rotation_axis);
    Vectorold3f rotcenter = _eye + 3.0f*viewdir;
    Vectorold3f rotcenter2 = _light + 3.0f*viewdir;
    currq = q.ComposeWith(currq);
    currq.ToRotAxis(_rotation, _rotation_axis);
    // set up orthogonal camera system
    Quat1f q2(_rotation, _rotation_axis);
    q2.x = -q2.x;
    Vectorold3f viewdir2(0,0,-1);
    q2.Rotate(viewdir2);

    _eye = rotcenter - 3.0f*viewdir2;
    _light = rotcenter2 - 3.0f*viewdir2;
}

Vectorold3f Camera::GetViewDir() const {
    Quat1f q(_rotation, _rotation_axis);
    q.x = -q.x;
    Vectorold3f viewdir(0,0,1);
    q.Rotate(viewdir);
    return viewdir;
}

void Camera::GetCoordSystem(Vectorold3f &view, Vectorold3f &up, Vectorold3f &right) const {
    Quat1f q(_rotation, _rotation_axis);
    q.x = -q.x;
    view = Vectorold3f(0,0,1);
    q.Rotate(view);
    up = Vectorold3f(0,1,0);
    q.Rotate(up);
    right = -view.Cross(up);
}

void Camera::Translate(const Vectorold3f translation) {
    Quat1f q(_rotation, _rotation_axis);
    q.x = -q.x;
    Vectorold3f xax(1,0,0);
    Vectorold3f yax(0,1,0);
    Vectorold3f zax(0,0,1);

    q.Rotate(xax);
    q.Rotate(yax);
    q.Rotate(zax);

    _eye+=translation.x * xax +
          translation.y * yax +
          translation.z * zax;
}

void Camera::TranslateLight(const Vectorold3f translation) {
    Quat1f q(_rotation, _rotation_axis);
    q.x = -q.x;
    Vectorold3f xax(1,0,0);
    Vectorold3f yax(0,1,0);
    Vectorold3f zax(0,0,1);

    q.Rotate(xax);
    q.Rotate(yax);
    q.Rotate(zax);

    _light+=translation.x * xax +
            translation.y * yax +
            translation.z * zax;
}

void Camera::Zoom(float amount) {
    _fov+=amount/10;
    _fov = max(_fov, 0.01f);
}

Vectorold3f Camera::GetPosition(float x, float y) {
    float r = x*x+y*y;
    float t = 0.5f * 1*1;
    if (r<t) {
        Vectorold3f result(x,y,sqrt(2.0f*t-r));
        result.Normalize();
        return result;
    }
    else {
        Vectorold3f result(x,y, t/sqrt(r));
        result.Normalize();
        return result;
    }
}

Quat1f Camera::GetQuaternion(float x1, float y1, float x2, float y2) {
    if ((x1==x2)&&(y1==y2)) {
        return Quat1f(1,0,0,0);
    }
    Vectorold3f pos1 = GetPosition(x1,y1);
    Vectorold3f pos2 = GetPosition(x2,y2);
    Vectorold3f rotaxis = pos1.Cross(pos2);
    rotaxis.Normalize();
    float rotangle = 2*sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    return Quat1f(rotangle, rotaxis);
}

void Camera::RegisterPoint(float x, float y) {
    _x = x;
    _y = y;
}
void Camera::RotateToPoint(float x, float y) {
    Quat1f q = GetQuaternion(_x,_y,x,y);
    RegisterPoint(x,y);
    Rotate(q);
}
void Camera::TranslateToPoint(float x, float y) {
    float dx = x - _x;
    float dy = y - _y;
    float dz = 0;
    RegisterPoint(x,y);
    Translate(Vectorold3f(-dx,-dy,-dz));
}

void Camera::TranslateLightToPoint(float x, float y) {
    float dx = x - _x;
    float dy = y - _y;
    float dz = 0;
    RegisterPoint(x,y);
    TranslateLight(Vectorold3f(3*dx,3*dy,3*dz));
}


