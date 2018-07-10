#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <vector>
#include "Quaternion.h"
#include "Vec.h"
#include "Matrix.h"

using namespace Physika;

class Camera {
public:
	Vector3f _eye;
	Vector3f _light;

	Camera();

	void RegisterPoint(float x, float y);
	void RotateToPoint(float x, float y);
	void TranslateToPoint(float x, float y);
	void TranslateLightToPoint(float x, float y);
	void Zoom(float amount);
	void SetGL(float neardist, float fardist, float width, float height);
	Transform3D<float> GetCombinedMatrix() const;
	Transform3D<float> GetCombinedMatrix2() const;

	Vector3f GetViewDir() const;

	float GetPixelArea() const;
	int GetWidth() const;
	int GetHeight() const;
	Vector3f GetEye() const;


	float _neardist;
	float _fardist;
	float _right;

	void GetCoordSystem(Vector3f &view, Vector3f &up, Vector3f &right) const;
	void Rotate(Quat1f &rotquat);

private:

	void Translate(const Vector3f translation);
	void TranslateLight(const Vector3f translation);
	Vector3f GetPosition(float x, float y);
	Quat1f GetQuaternion(float x1, float y1, float x2, float y2);

	float _rotation;
	Vector3f _rotation_axis;
	float _fov;

	float _x;
	float _y;

	float _pixelarea;
	int _width;
	int _height;

};

#endif
