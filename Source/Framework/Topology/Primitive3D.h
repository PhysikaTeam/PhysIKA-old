#include "Core/Vector.h"

namespace PhysIKA
{
#ifdef PRECISION_FLOAT
	#define REAL_EPSILON 1e-6
#else
	#define REAL_EPSILON 1e-10
#endif

#ifdef PRECISION_FLOAT
	typedef Vector3f Coord3D;
#else
	typedef Vector3d Coord3D;
#endif

	class Line3D;
	class Ray3D;
	class Segment3D;
	class Triangle3D;

	class Point3D
	{
	public:
		COMM_FUNC Point3D();
		COMM_FUNC Point3D(const Real& val);
		COMM_FUNC Point3D(const Real& c0, const Real& c1, const Real& c2);
		COMM_FUNC Point3D(const Coord3D& pos);
		COMM_FUNC Point3D(const Point3D& pt);

		Real project(const Line3D& line, Point3D& q) const;
		Real project(const Ray3D& ray, Point3D& q) const;
		Real project(const Segment3D& segment, Point3D& q) const;
		Real project(const Triangle3D& line, Point3D& q) const;

		Real distance(const Point3D& pt) const;
		Real distance(const Line3D& line) const;
		Real distance(const Ray3D& ray) const;
		Real distance(const Segment3D& segment) const;

		Coord3D origin;
	};

	class Line3D
	{
	public:
		COMM_FUNC Line3D();
		COMM_FUNC Line3D(const Coord3D& pos, const Coord3D& dir);
		COMM_FUNC Line3D(const Line3D& line);

		COMM_FUNC Real distance(const Point3D& pt) const;

		Coord3D origin;

		//guarantee direction is a unit vector
		Coord3D direction;
	};

	class Ray3D
	{
	public:
		COMM_FUNC Ray3D();
		COMM_FUNC Ray3D(const Coord3D& pos, const Coord3D& dir);
		COMM_FUNC Ray3D(const Ray3D& ray);

		COMM_FUNC Real distance(const Point3D& pt) const;


		Coord3D origin;

		//guarantee direction is a unit vector
		Coord3D direction;
	};

	class Segment3D
	{
	public:
		COMM_FUNC Segment3D();
		COMM_FUNC Segment3D(const Coord3D& p0, const Coord3D& p1);
		COMM_FUNC Segment3D(const Segment3D& segment);

		Coord3D v0;
		Coord3D v1;
	};

	class Plane3D
	{
	public:
		COMM_FUNC Plane3D() {};

		Coord3D origin;
		Coord3D direction;
	};

	class Triangle3D
	{
	public:
		COMM_FUNC Triangle3D() {};
	};

}

