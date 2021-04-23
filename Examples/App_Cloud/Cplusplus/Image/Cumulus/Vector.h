#ifndef _VVECTOR_H
#define _VVECTOR_H
#include <math.h>
//为什么不是class
//默认的访问权限希望都为public，所以使用struct定义类
struct Vector3 {
public:

	Vector3()
	{
		x = y = z = 0.0f;
	}

	Vector3(float px, float py, float pz)
	{
		x = px; y = py; z = pz;
	}

	Vector3 operator+(Vector3 vVector3)
	{
		return Vector3(vVector3.x + x, vVector3.y + y, vVector3.z + z);
	}

	Vector3 operator-(Vector3 vVector3)
	{
		return Vector3(x - vVector3.x, y - vVector3.y, z - vVector3.z);
	}

	Vector3 operator*(float num)
	{
		return Vector3(x * num, y * num, z * num);
	}

	Vector3 operator/(float num)
	{
		return Vector3(x / num, y / num, z / num);
	}

	Vector3 operator-()
	{
		return Vector3(-x, -y, -z);
	}

	float* operator!()
	{
		return (float*)this;
	}

	Vector3& operator *=(const float Scalar)
	{
		x *= Scalar; y *= Scalar; z *= Scalar;
		return *this;
	}

	Vector3& operator +=(const Vector3 Other)
	{
		x += Other.x;	y += Other.y;	z += Other.z;
		return *this;
	}

	Vector3& operator -=(const Vector3 Other)
	{
		x -= Other.x;	y -= Other.y;	z -= Other.z;
		return *this;
	}

	float x, y, z;
};

struct Vector2
{
public:
	Vector2()
	{
		x = y = 0.0f;
	}

	Vector2(float px, float py)
	{
		x = px; y = py;
	}

	Vector2 operator+(Vector2 vVector2)
	{
		return Vector2(vVector2.x + x, vVector2.y + y);
	}

	Vector2 operator-(Vector2 vVector2)
	{
		return Vector2(x - vVector2.x, y - vVector2.y);
	}

	Vector2 operator*(float num)
	{
		return Vector2(x * num, y * num);
	}

	Vector2 operator/(float num)
	{
		return Vector2(x / num, y / num);
	}

	Vector2 operator-()
	{
		return Vector2(-x, -y);
	}

	float* operator!()
	{
		return (float*)this;
	}

	Vector2& operator *=(const float Scalar)
	{
		x *= Scalar; y *= Scalar;
		return *this;
	}

	Vector2& operator +=(const Vector2 Other)
	{
		x += Other.x;	y += Other.y;
		return *this;
	}

	Vector2& operator -=(const Vector2 Other)
	{
		x -= Other.x;	y -= Other.y;
		return *this;
	}

	float x, y;
};

inline float Magnitude(Vector3 v)
{
	return (float)sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline float Magnitude(Vector2 v)
{
	return (float)sqrtf(v.x * v.x + v.y * v.y);
}

inline float Dot(Vector3 v1, Vector3 v2)
{
	return (float)(v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);//点积
}

inline float Dot(Vector2 v1, Vector2 v2)
{
	return (float)(v1.x * v2.x + v1.y * v2.y);
}

inline float Angle(Vector3 v1, Vector3 v2)
{
	return Dot(v1, v2) / (Magnitude(v1) * Magnitude(v2));//计算夹角
}

inline Vector3 Normalize(Vector3 vVector3)
{
	vVector3 = vVector3 / Magnitude(vVector3);
	return vVector3;
}

inline Vector2 Normalize(Vector2 vVector2)
{
	vVector2 = vVector2 / Magnitude(vVector2);
	return vVector2;
}

inline float Dist(Vector3 pn1, Vector3 pn2)
{
	return (float)sqrtf((pn1.x - pn2.x) * (pn1.x - pn2.x) + (pn1.y - pn2.y) * (pn1.y - pn2.y) + (pn1.z - pn2.z) * (pn1.z - pn2.z));
}

inline float SqDist(Vector3 pn1, Vector3 pn2)
{
	return (float)((pn1.x - pn2.x) * (pn1.x - pn2.x) + (pn1.y - pn2.y) * (pn1.y - pn2.y) + (pn1.z - pn2.z) * (pn1.z - pn2.z));
}

inline Vector3 Cross(Vector3 vVector31, Vector3 vVector32)
{
	Vector3 vNormal;

	vNormal.x = ((vVector31.y * vVector32.z) - (vVector31.z * vVector32.y));
	vNormal.y = ((vVector31.z * vVector32.x) - (vVector31.x * vVector32.z));
	vNormal.z = ((vVector31.x * vVector32.y) - (vVector31.y * vVector32.x));

	return vNormal;
}

#endif
