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

#include <GL/gl.h>
#include "Camera.h"
#include <iostream>
#include <math.h>
using namespace std;

namespace PhysIKA
{
	Camera::Camera() {
		m_eye = Vector3f(0, 0, 3);
		m_light = Vector3f(0, 0, 3);
		m_rotation = -0;
		m_rotation_axis = Vector3f(0, 1, 0);
		m_fov = 0.90f;

		Quat1f curq(m_rotation, m_rotation_axis);
		m_updir = curq.rotate(Vector3f(0, 1, 0));
		m_rightdir = curq.rotate(Vector3f(1, 0, 0));
	}


	void Camera::setGL(float neardist, float fardist, float width, float height) {
		float diag = sqrt(width*width + height*height);
		float top = height / diag * 0.5f*m_fov*neardist;
		float bottom = -top;
		float right = width / diag* 0.5f*m_fov*neardist;
		float left = -right;

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(left, right, bottom, top, neardist, fardist);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glRotatef(-180.0f / M_PI * m_rotation, m_rotation_axis[0], m_rotation_axis[1], m_rotation_axis[2]);
		glTranslatef(-m_eye[0], -m_eye[1], -m_eye[2]);
		

		GLfloat pos[] = { m_light[0], m_light[1], m_light[2],1 };
		glLightfv(GL_LIGHT0, GL_POSITION, pos);

		m_width = (int)width;
		m_height = (int)height;
		m_pixelarea = 4 * right*top / (width*height);
		m_near = neardist;
		m_far = fardist;
		m_right = right;
	}

	int Camera::width() const {
		return m_width;
	}

	int Camera::height() const {
		return m_height;
	}

	float Camera::getPixelArea() const {
		return m_pixelarea;
	}

	Vector3f Camera::getEye() const {
		return m_eye;
	}

	void Camera::rotate(Quat1f &rotquat) {
		// set up orthogonal camera system

		Quat1f currq (m_rotation, m_rotation_axis);

		Vector3f tmpv = currq.rotate(Vector3f(1.0, 0, 0));

		currq = rotquat * currq;
		currq.toRotationAxis(m_rotation, m_rotation_axis);

		return;
	}

	Vector3f Camera::getViewDir() const {
		Quat1f q(m_rotation, m_rotation_axis);
		//q.x = -q.x;
		//q.setW(-q.w());
		Vector3f viewdir(0, 0, -1);
		viewdir = q.rotate(viewdir);
		return viewdir;
	}

	void Camera::getCoordSystem(Vector3f &view, Vector3f &up, Vector3f &right) const {
		Quat1f q(m_rotation, m_rotation_axis);
		//q.x = -q.x;
		//q.setW(-q.w());
		view = Vector3f(0, 0, -1);
		view = q.rotate(view);
		up = Vector3f(0, 1, 0);
		up = q.rotate(up);
		right = view.cross(up);
	}

	void Camera::translate(const Vector3f translation) {
		Quat1f q(m_rotation, m_rotation_axis);
		m_eye += q.rotate(translation);
	}

	void Camera::translateLight(const Vector3f translation) {
		Quat1f q(m_rotation, m_rotation_axis);
		//q.x = -q.x;
		//q.setW(-q.w());
		Vector3f xax(1, 0, 0);
		Vector3f yax(0, 1, 0);
		Vector3f zax(0, 0, 1);

		xax = q.rotate(xax);
		yax = q.rotate(yax);
		zax = q.rotate(zax);

		m_light += translation[0] * xax +
			translation[1] * yax +
			translation[2] * zax;
	}

	void Camera::zoom(float amount) {
		m_fov += amount / 10;
		m_fov = max(m_fov, 0.01f);
	}

	Vector3f Camera::getPosition(float x, float y) {
		float r = x*x + y*y;
		float t = 0.5f * 1 * 1;
		if (r < t) {
			Vector3f result(x, y, sqrt(2.0f*t - r));
			result.normalize();
			return result;
		}
		else {
			Vector3f result(x, y, t / sqrt(r));
			result.normalize();
			return result;
		}
	}

	Quat1f Camera::getQuaternion(float x1, float y1, float x2, float y2) {
		
		Quat1f currq(m_rotation, m_rotation_axis);

		Vector3f viewdir(0, 0, -1);
		viewdir = currq.rotate(viewdir);
		//viewdir = currq.rotate(viewdir);

		Vector3f cur_axis(y1 - y2, x2 - x1, 0);
		if ((cur_axis[0] <1e-6 && cur_axis[0] > -1e-6) && ((cur_axis[1] <1e-6 && cur_axis[1] > -1e-6)))
		{
			return Quat1f();
		}
		cur_axis = currq.rotate(cur_axis);

		float range_ = 1.0;
		float ran1 = range_ * sqrt((y2 - y1)*(y2 - y1) + (x2 - x1)*(x2 - x1));
		Quat1f q1(cur_axis, ran1);

		viewdir = q1.rotate(viewdir);
		

		Vector3f rightdir1(1.0, 0, 0);
		rightdir1 = q1.rotate(currq.rotate(rightdir1)); 
		rightdir1.normalize();

		Vector3f tmpv1(1.0, 0, 0);
		Vector3f tmpv2 = q1.rotate(tmpv1);
		Vector3f tmpv3 = currq.rotate(tmpv2);
		Vector3f tmpv4 = (currq*q1).rotate(tmpv1);
		Vector3f tmpv5 = (q1*currq).rotate(tmpv1);

		Vector3f tmpv6 = currq.rotate(tmpv1);
		Vector3f tmpv7 = q1.rotate(tmpv6);

		
		Vector3f rightdir2 = rightdir1;

		Vector3f updir(0, 1, 0);
		//if ((rightdir2[1] * rightdir2[1]) > 1e-8)
		{
			rightdir2 = viewdir.cross(updir);
			rightdir2.normalize();
		}

		//Vector3f rightdir2(rightdir1[0], 0.0, rightdir1[2]);
		//rightdir2.normalize();
		float norm = (rightdir1.norm() * rightdir2.norm());
		float rightsin = rightdir1.cross(rightdir2).norm() / norm;
		rightsin = (rightsin > 1.0) ? 1.0 : rightsin;
		rightsin = (rightsin < -1.0) ? -1.0 : rightsin;

		float ran2 = asin(rightsin);
		float tmpSin = sin(ran2);

		Quat1f q2(viewdir, ran2);
		q2.normalize();
		
		//if (q2.rotate(rightdir1).dot(rightdir2) < (1.0 - 1e-5))
		if(rightdir1.cross(rightdir2).dot(viewdir) < 0)
		{
			q2 = q2.getConjugate();
		}

		Quat1f rotquat = q2 * q1;
		rotquat = rotquat * currq;// currq * rotquat;//rotquat * currq;
		rotquat.normalize();

		Vector3f newaxis;
		float newran;
		rotquat.toRotationAxis(newran, newaxis);
		if (!((newran < 30) && (newran > -30)))
		{
			return Quat1f();
		}

		Vector3f tmpAxisX(1.0, 0, 0);
		Vector3f tmpX1 = currq.rotate(tmpAxisX);
		Vector3f tmpX2 = q1.rotate(tmpX1);
		Vector3f tmpX3 = q2.rotate(tmpX2);
		Vector3f tmpX4 = q2.getConjugate().rotate(tmpX2);


		tmpAxisX = rotquat.rotate(tmpAxisX);
		
		if ((tmpAxisX[1] > 1e-6) || (tmpAxisX[1] < -1e-6))
		{
			return q2 * q1;
		}

		return q2 * q1;

		//cur_axis = currq.rotate(cur_axis);


		
	}

	void Camera::registerPoint(float x, float y) {
		m_x = x;
		m_y = y;
	}
	void Camera::rotateToPoint(float x, float y) {
		/// The rotation will be devided into 2 steps
		/// 1. Rotate the camera to correct the new view direction
		/// 2. Rotate the camera along view direction to constrain camera's X axis in world XZ plane.

		//Quat1f q = getQuaternion(m_x, m_y, x, y);

		float eps = 1e-5;
		float eps2 = 1e-10;

		float y1 = m_y, y2 = y;
		float x1 = m_x, x2 = x;

		Quat1f currq(m_rotation, m_rotation_axis);

		Vector3f viewdir(0, 0, -1);
		viewdir = currq.rotate(viewdir);		///< view direction in world frame
		
		Vector3f cur_axis(y1 - y2, x2 - x1, 0);
		if (cur_axis.normSquared()>eps2)
		{
			cur_axis = currq.rotate(cur_axis);		///< axis of q1 in world frame

			float range_ = 1.0;
			float ran1 = range_ * sqrt((y2 - y1)*(y2 - y1) + (x2 - x1)*(x2 - x1));
			Quat1f q1(cur_axis, ran1);

			viewdir = q1.rotate(viewdir);			///< new view direction (after q1 rotation)

			Vector3f rightdir1(1.0, 0, 0);
			rightdir1 = q1.rotate(currq.rotate(rightdir1));	///< camera's X axis in world frame ( after q1 rotation)
			rightdir1.normalize();

			///< camera's X axis in world frame (after q2 rotation)
			///< the initialization maybe in negative direction of actual right direction
			Vector3f rightdir2(viewdir[2], 0, -viewdir[0]);		
			if (rightdir2.normSquared() < eps2)					///< in this case, view direction is along Y axis, rightdir1 is in XZ plane
			{
				rightdir2 = rightdir1;
			}
			else if (rightdir2.dot(rightdir1) < 0)				///< in this case, rightdir2 is in negative direction
			{
				rightdir2 = -rightdir2;
			}
			rightdir2.normalize();


			float norm = (rightdir1.norm() * rightdir2.norm());
			float rightsin = rightdir1.cross(rightdir2).norm() / norm;		///< sin value of the angle between rightdir1 and rightdir2, don't use cos!
			rightsin = (rightsin > 1.0) ? 1.0 : rightsin;
			rightsin = (rightsin < -1.0) ? -1.0 : rightsin;
			float ran2 = asin(rightsin);					///< rotation radian of q2

			Quat1f q2(viewdir, ran2);
			q2.normalize();

			if (rightdir1.cross(rightdir2).dot(viewdir) < 0)
			{
				q2 = q2.getConjugate();
			}

			(q2 * q1 * currq).toRotationAxis(m_rotation, m_rotation_axis);
		}
		
		registerPoint(x, y);
		//rotate(q);
	}
	void Camera::translateToPoint(float x, float y) {
		float dx = x - m_x;
		float dy = y - m_y;
		float dz = 0;
		registerPoint(x, y);
		translate(Vector3f(-dx, -dy, -dz));

		
	}

	void Camera::translateLightToPoint(float x, float y) {
		float dx = x - m_x;
		float dy = y - m_y;
		float dz = 0;
		registerPoint(x, y);
		translateLight(Vector3f(3 * dx, 3 * dy, 3 * dz));
	}

}
