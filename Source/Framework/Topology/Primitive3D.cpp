#include "Primitive3D.h"

namespace PhysIKA
{
	COMM_FUNC Point3D::Point3D()
	{
		origin = Coord3D(0);
	}

	COMM_FUNC Point3D::Point3D(const Real& val)
	{
		origin = Coord3D(val);
	}

	COMM_FUNC Point3D::Point3D(const Real& c0, const Real& c1, const Real& c2)
	{
		origin = Coord3D(c0, c1, c2);
	}

	COMM_FUNC Point3D::Point3D(const Coord3D& pos)
	{
		origin = pos;
	}

	COMM_FUNC Point3D::Point3D(const Point3D& pt)
	{
		origin = pt.origin;
	}

	Real Point3D::project(const Line3D& line, Point3D& q) const
	{
		Coord3D l = origin - line.origin;
		Real t = l.dot(line.direction);

		q.origin = line.origin + t * line.direction;
		
		Coord3D vec = origin - q.origin;
		return vec.norm();
	}

	//TODO:
	Real Point3D::project(const Triangle3D& line, Point3D& q) const
	{
		return 0;
	}

	Real Point3D::project(const Ray3D& ray, Point3D& q) const
	{
		Coord3D l = origin - ray.origin;
		Real t = l.dot(ray.direction);

		t = t < 0 ? 0 : t;
		q.origin = ray.origin + t * ray.direction;

		Coord3D vec = origin - q.origin;
		return vec.norm();
	}


	Real Point3D::project(const Segment3D& segment, Point3D& q) const
	{
		Coord3D l = origin - segment.v0;
		Coord3D dir = segment.v1 - segment.v0;
		if (dir.norm() < REAL_EPSILON)
		{
			q.origin = segment.v0;
			return (origin - q.origin).norm();
		}

		Real t = l.dot(dir) / dir.normSquared();

		q.origin = segment.v0 + t * dir;
		q.origin = t < 0 ? segment.v0 : q.origin;
		q.origin = t > 1 ? segment.v1 : q.origin;

		return (origin - q.origin).norm();
	}

	Real Point3D::distance(const Point3D& pt) const
	{
		return (origin - pt.origin).norm();
	}


	Real Point3D::distance(const Line3D& line) const
	{
		Point3D q;
		return project(line, q);
	}

	Real Point3D::distance(const Ray3D& ray) const
	{
		Point3D q;
		return project(ray, q);
	}

	Real Point3D::distance(const Segment3D& segment) const
	{
		Point3D q;
		return project(segment, q);
	}

	COMM_FUNC Line3D::Line3D()
	{
		origin = Coord3D(0);
		direction = Coord3D(1, 0, 0);
	}

	COMM_FUNC Line3D::Line3D(const Coord3D& pos, const Coord3D& dir)
	{
		origin = pos;

		//To avoid zero vector
		direction = dir.norm() < REAL_EPSILON ? Coord3D(1, 0, 0) : dir;
		direction.normalize();
	}

	COMM_FUNC Line3D::Line3D(const Line3D& line)
	{
		origin = line.origin;
		direction = line.direction;
	}

	COMM_FUNC Real Line3D::distance(const Point3D& pt) const
	{
		return pt.distance(*this);
	}

	COMM_FUNC Ray3D::Ray3D()
	{
		origin = Coord3D(0);
		direction = Coord3D(1, 0, 0);
	}

	COMM_FUNC Ray3D::Ray3D(const Coord3D& pos, const Coord3D& dir)
	{
		origin = pos;

		//To avoid zero vector
		direction = dir.norm() < REAL_EPSILON ? Coord3D(1, 0, 0) : dir;
		direction.normalize();
	}

	COMM_FUNC Ray3D::Ray3D(const Ray3D& ray)
	{
		origin = ray.origin;
		direction = ray.direction;
	}

	COMM_FUNC Real Ray3D::distance(const Point3D& pt) const
	{
		return pt.distance(*this);
	}

	COMM_FUNC Segment3D::Segment3D()
	{
		v0 = Coord3D(0, 0, 0);
		v1 = Coord3D(1, 0, 0);
	}

	COMM_FUNC Segment3D::Segment3D(const Coord3D& p0, const Coord3D& p1)
	{
		v0 = p0;
		v1 = p1;
	}

	COMM_FUNC Segment3D::Segment3D(const Segment3D& segment)
	{
		v0 = segment.v0;
		v1 = segment.v1;
	}

}