#pragma once 

#include <iostream>

namespace PhysIKA {
	class vec3f {
	public:
		union {
			struct {
				float x, y, z;
			};
			struct {
				float v[3];
			};
		};

		vec3f()
		{
			x = 0; y = 0; z = 0;
		}

		vec3f(const vec3f& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
		}

		vec3f(const float* v)
		{
			x = v[0];
			y = v[1];
			z = v[2];
		}
		vec3f(float* v)
		{
			x = v[0];
			y = v[1];
			z = v[2];
		}

		vec3f(float x, float y, float z)
		{
			this->x = x;
			this->y = y;
			this->z = z;
		}

		float operator [] (int i) const { return v[i]; }
		float& operator [] (int i) { return v[i]; }

		vec3f& operator += (const vec3f& v) {
			x += v.x;
			y += v.y;
			z += v.z;
			return *this;
		}

		vec3f& operator -= (const vec3f& v) {
			x -= v.x;
			y -= v.y;
			z -= v.z;
			return *this;
		}

		vec3f& operator *= (float t) {
			x *= t;
			y *= t;
			z *= t;
			return *this;
		}

		vec3f& operator /= (float t) {
			x /= t;
			y /= t;
			z /= t;
			return *this;
		}

		void negate() {
			x = -x;
			y = -y;
			z = -z;
		}

		vec3f operator - () const {
			return vec3f(-x, -y, -z);
		}

		vec3f operator+ (const vec3f& v) const
		{
			return vec3f(x + v.x, y + v.y, z + v.z);
		}

		vec3f operator- (const vec3f& v) const
		{
			return vec3f(x - v.x, y - v.y, z - v.z);
		}

		vec3f operator *(float t) const
		{
			return vec3f(x * t, y * t, z * t);
		}

		vec3f operator /(float t) const
		{
			return vec3f(x / t, y / t, z / t);
		}

		// cross product
		const vec3f cross(const vec3f& vec) const
		{
			return vec3f(y * vec.z - z * vec.y, z * vec.x - x * vec.z, x * vec.y - y * vec.x);
		}

		float dot(const vec3f& vec) const {
			return x * vec.x + y * vec.y + z * vec.z;
		}

		void normalize()
		{
			float sum = x * x + y * y + z * z;
			if (sum > float(10e-12)) {
				float base = float(1.0 / sqrt(sum));
				x *= base;
				y *= base;
				z *= base;
			}
		}

		float length() const {
			return float(sqrt(x * x + y * y + z * z));
		}

		vec3f getUnit() const {
			return (*this) / length();
		}
		inline bool isEqual(float a, float b, float tol = float(10e-6)) const
		{
			return fabs(a - b) < tol;
		}
		bool isUnit() const {
			return isEqual(squareLength(), 1.f);
		}

		//! max(|x|,|y|,|z|)
		float infinityNorm() const
		{
			return fmax(fmax(fabs(x), fabs(y)), fabs(z));
		}

		vec3f& set_value(const float& vx, const float& vy, const float& vz)
		{
			x = vx; y = vy; z = vz; return *this;
		}

		bool equal_abs(const vec3f& other) {
			return x == other.x && y == other.y && z == other.z;
		}

		float squareLength() const {
			return x * x + y * y + z * z;
		}

		static vec3f zero() {
			return vec3f(0.f, 0.f, 0.f);
		}

		//! Named constructor: retrieve vector for nth axis
		static vec3f axis(int n) {
			switch (n) {
			case 0: {
				return xAxis();
			}
			case 1: {
				return yAxis();
			}
			case 2: {
				return zAxis();
			}
			}
			return vec3f();
		}

		//! Named constructor: retrieve vector for x axis
		static vec3f xAxis() { return vec3f(1.f, 0.f, 0.f); }
		//! Named constructor: retrieve vector for y axis
		static vec3f yAxis() { return vec3f(0.f, 1.f, 0.f); }
		//! Named constructor: retrieve vector for z axis
		static vec3f zAxis() { return vec3f(0.f, 0.f, 1.f); }

	};

	inline vec3f operator * (float t, const vec3f& v) {
		return vec3f(v.x * t, v.y * t, v.z * t);
	}

	inline vec3f interp(const vec3f& a, const vec3f& b, float t)
	{
		return a * (1 - t) + b * t;
	}

	inline vec3f vinterp(const vec3f& a, const vec3f& b, float t)
	{
		return a * t + b * (1 - t);
	}

	inline vec3f interp(const vec3f& a, const vec3f& b, const vec3f& c, float u, float v, float w)
	{
		return a * u + b * v + c * w;
	}

	inline float vdistance(const vec3f& a, const vec3f& b)
	{
		return (a - b).length();
	}


	inline std::ostream& operator<<(std::ostream& os, const vec3f& v) {
		os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
		return os;
	}

	inline void vmin(vec3f& a, const vec3f& b)
	{
		a.set_value(
			fmin(a[0], b[0]),
			fmin(a[1], b[1]),
			fmin(a[2], b[2]));
	}

	inline void vmax(vec3f& a, const vec3f& b)
	{
		a.set_value(
			fmax(a[0], b[0]),
			fmax(a[1], b[1]),
			fmax(a[2], b[2]));
	}

	inline vec3f lerp(const vec3f& a, const vec3f& b, float t)
	{
		return a + t * (b - a);
	}
}