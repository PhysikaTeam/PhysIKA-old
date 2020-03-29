#include "Primitive3D.h"
#include "Core/Utility/SimpleMath.h"
#include "Core/Interval.h"

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

	COMM_FUNC Point3D Point3D::operator=(const Coord3D& p)
	{
		Point3D pt;
		pt.origin = p;
		return pt;
	}

	COMM_FUNC Point3D Point3D::project(const Line3D& line) const
	{
		Coord3D u = origin - line.origin;
		Real tNum = u.dot(line.direction);
		Real a = line.direction.normSquared();
		Real t = a < REAL_EPSILON_SQUARED ? 0 : tNum / a;

		return Point3D(line.origin + t * line.direction);
	}

	COMM_FUNC Point3D Point3D::project(const Ray3D& ray) const
	{
		Coord3D u = origin - ray.origin;

		Real tNum = u.dot(ray.direction);
		Real a = ray.direction.normSquared();
		Real t = a < REAL_EPSILON_SQUARED ? 0 : tNum / a;
		
		t = t < 0 ? 0 : t;

		return Point3D(ray.origin + t * ray.direction);
	}


	COMM_FUNC Point3D Point3D::project(const Segment3D& segment) const
	{
		Coord3D l = origin - segment.v0;
		Coord3D dir = segment.v1 - segment.v0;
		if (dir.normSquared() < REAL_EPSILON_SQUARED)
		{
			return Point3D(segment.v0);
		}

		Real t = l.dot(dir) / dir.normSquared();

		Coord3D q = segment.v0 + t * dir;
		q = t < 0 ? segment.v0 : q;
		q = t > 1 ? segment.v1 : q;

		return Point3D(q);
	}

	COMM_FUNC Point3D Point3D::project(const Plane3D& plane) const
	{
		Real t = (origin - plane.origin).dot(plane.normal);

		Real n2 = plane.normal.normSquared();

		return n2 < REAL_EPSILON ? Point3D(plane.origin) : Point3D(origin - t / n2 * plane.normal);
	}

	COMM_FUNC Point3D Point3D::project(const Triangle3D& triangle) const
	{
		Coord3D dir = triangle.v[0] - origin;
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Real a = e0.dot(e0);
		Real b = e0.dot(e1);
		Real c = e1.dot(e1);
		Real d = e0.dot(dir);
		Real e = e1.dot(dir);
		Real f = dir.dot(dir);

		Real det = a * c - b * b;
		Real s = b * e - c * d;
		Real t = b * d - a * e;

		//handle degenerate triangles
		if (det < REAL_EPSILON)
		{
			Real g = (triangle.v[2] - triangle.v[1]).normSquared();

			Real l_max = a;
			Coord3D p0 = triangle.v[0];
			Coord3D p1 = triangle.v[1];

			if (c > l_max)
			{
				p0 = triangle.v[0];
				p1 = triangle.v[2];

				l_max = c;
			}

			if (g > l_max)
			{
				p0 = triangle.v[1];
				p1 = triangle.v[2];
			}

			return project(Segment3D(p0, p1));
		}

		if (s + t <= det)
		{
			if (s < 0)
			{
				if (t < 0)
				{
					//region 4
					s = 0;
					t = 0;
				}
				else
				{
					// region 3
					// F(t) = Q(0, t) = ct^2 + 2et + f
					// F'(t)/2 = ct + e
					// F'(t) = 0 when t = -e/c

					s = 0;
					t = (e >= 0 ? 0 : (-e >= c ? 1 : -e / c));
				}
			}
			else
			{
				if (t < 0)
				{
					//region 5
					// F(s) = Q(s, 0) = as^2 + 2ds + f
					// F'(s)/2 = as + d
					// F'(s) = 0 when t = -d/a

					s = (d >= 0 ? 0 : (-d >= a ? 1 : -d / a));
					t = 0;
				}
				else
				{
					//region 0
					Real invDet = 1 / det;
					s *= invDet;
					t *= invDet;
				}
			}
		}
		else
		{
			if (s < 0)
			{
				//region 2
				s = 0;
				t = 1;
			}
			else if (t < 0)
			{
				//region 6
				s = 1;
				t = 0;
			}
			else
			{
				//region 1
				// F(s) = Q(s, 1 - s) = (a - 2b + c)s^2 + 2(b - c + d - e)s + (c + 2e + f)
				// F'(s)/2 = (a - 2b + c)s + (b - c + d - e)
				// F'(s} = 0 when s = (c + e - b - d)/(a - 2b + c)
				// a - 2b + c = |e0 - e1|^2 > 0,
				// so only the sign of c + e - b - d need be considered

				Real numer = c + e - b - d;
				if (numer <= 0) {
					s = 0;
				}
				else {
					Real denom = a - 2 * b + c; // positive quantity
					s = (numer >= denom ? 1 : numer / denom);
				}
				t = 1 - s;
			}
		}
		return Point3D(triangle.v[0] + s * e0 + t * e1);
	}

	COMM_FUNC Point3D Point3D::project(const Rectangle3D& rectangle) const
	{
		Coord3D diff = origin - rectangle.center;
		Real b0 = diff.dot(rectangle.axis[0]);
		Real b1 = diff.dot(rectangle.axis[1]);
		Coord2D uv(b0, b1);

		uv = clamp(uv, -rectangle.extent, rectangle.extent);

		return Point3D(rectangle.center + uv[0] * rectangle.axis[0] + uv[1] * rectangle.axis[1]);
	}

	COMM_FUNC Point3D Point3D::project(const Disk3D& disk) const
	{
		Coord3D cp = origin - disk.center;
		Coord3D cq = cp - cp.dot(disk.normal)*disk.normal;

		Coord3D q;
		q = disk.center + cq;
		if (cq.normSquared() > disk.radius*disk.radius)
		{
			q = disk.center + disk.radius*cq.normalize();
		}
		return Point3D(q);
	}

	COMM_FUNC Point3D Point3D::project(const Sphere3D& sphere, bool& bInside) const
	{
		Coord3D cp = origin - sphere.center;
		Coord3D q = sphere.center + sphere.radius*cp.normalize();

		bInside = cp.normSquared() >= sphere.radius*sphere.radius ? false : true;

		return Point3D(q);
	}

	COMM_FUNC Point3D Point3D::project(const Capsule3D& capsule, bool& bInside /*= Bool(false)*/) const
	{
		Coord3D coordQ = project(capsule.segment).origin;
		Coord3D dir = origin - coordQ;

		bInside = dir.normSquared() < capsule.radius*capsule.radius ? true : false;
		return Point3D(coordQ + capsule.radius*dir.normalize());
	}

	COMM_FUNC Point3D Point3D::project(const Tet3D& tet, bool& bInside) const
	{
		bInside = true;

		Point3D closestPt;
		Real minDist = REAL_MAX;
		for (int i = 0; i < 4; i++)
		{
			Triangle3D face = tet.face(i);
			Point3D q = project(tet.face(i));
			Real d = (origin - q.origin).normSquared();
			if (d < minDist)
			{
				minDist = d;
				closestPt = q;
			}
			bInside &= (origin - face.v[0]).dot(face.normal()) < Real(0);
		}

		return closestPt;
	}

	COMM_FUNC Point3D Point3D::project(const AlignedBox3D& abox, bool& bInside) const
	{
		bInside = inside(abox);

		if (!bInside)
		{
			return Point3D(clamp(origin, abox.v0, abox.v1));
		}

		//compute the distance to six faces
		Coord3D q;
		Real minDist = REAL_MAX;
		Coord3D offset0 = abs(origin - abox.v0);
		if (offset0[0] < minDist)
		{
			q = Coord3D(abox.v0[0], origin[1], origin[2]);
			minDist = offset0[0];
		}

		if (offset0[1] < minDist)
		{
			q = Coord3D(origin[0], abox.v0[1], origin[2]);
			minDist = offset0[1];
		}

		if (offset0[2] < minDist)
		{
			q = Coord3D(origin[0], origin[1], abox.v0[2]);
			minDist = offset0[2];
		}


		Coord3D offset1 = abs(origin - abox.v1);
		if (offset1[0] < minDist)
		{
			q = Coord3D(abox.v1[0], origin[1], origin[2]);
			minDist = offset1[0];
		}

		if (offset1[1] < minDist)
		{
			q = Coord3D(origin[0], abox.v1[1], origin[2]);
			minDist = offset1[1];
		}

		if (offset1[2] < minDist)
		{
			q = Coord3D(origin[0], origin[1], abox.v1[2]);
			minDist = offset1[2];
		}

		return Point3D(q);
	}

	COMM_FUNC Point3D Point3D::project(const OrientedBox3D& obb, bool& bInside) const
	{
		Coord3D offset = origin - obb.center;
		Coord3D pPrime(offset.dot(obb.u), offset.dot(obb.v), offset.dot(obb.w));
		
		Coord3D qPrime = Point3D(pPrime).project(AlignedBox3D(-obb.extent, obb.extent), bInside).origin;

		Coord3D q = obb.center + qPrime[0] * obb.u + qPrime[1] * obb.v + qPrime[2] * obb.w;

		return Point3D(q);
	}

	COMM_FUNC Real Point3D::distance(const Point3D& pt) const
	{
		return (origin - pt.origin).norm();
	}


	COMM_FUNC Real Point3D::distance(const Line3D& line) const
	{
		return (origin - project(line).origin).norm();
	}

	COMM_FUNC Real Point3D::distance(const Ray3D& ray) const
	{
		return (origin - project(ray).origin).norm();
	}

	COMM_FUNC Real Point3D::distance(const Segment3D& segment) const
	{
		return (origin - project(segment).origin).norm();
	}

	COMM_FUNC Real Point3D::distance(const Plane3D& plane) const
	{
		Coord3D q = project(plane).origin;
		Real sign = (origin - q).dot(plane.normal) < Real(0) ? Real(-1) : Real(1);

		return sign*(origin - q).norm();
	}

	COMM_FUNC Real Point3D::distance(const Triangle3D& triangle) const
	{
		Coord3D q = project(triangle).origin;
		Real sign = (origin - q).dot(triangle.normal()) < Real(0) ? Real(-1) : Real(1);

		return sign*(origin - q).norm();
	}

	COMM_FUNC Real Point3D::distance(const Rectangle3D& rectangle) const
	{
		Coord3D q = project(rectangle).origin;
		Real sign = (origin - q).dot(rectangle.normal()) < Real(0) ? Real(-1) : Real(1);

		return (origin - q).norm();
	}

	COMM_FUNC Real Point3D::distance(const Disk3D& disk) const
	{
		Coord3D q = project(disk).origin;
		Real sign = (origin - q).dot(disk.normal) < Real(0) ? Real(-1) : Real(1);

		return (origin - q).norm();
	}

	COMM_FUNC Real Point3D::distance(const Sphere3D& sphere) const
	{
		return (origin - sphere.center).norm() - sphere.radius;
	}

	COMM_FUNC Real Point3D::distance(const Tet3D& tet) const
	{
		bool bInside;
		Real d = (origin - project(tet, bInside).origin).norm();
		return bInside == true ? -d : d;
	}

	COMM_FUNC Real Point3D::distance(const AlignedBox3D& abox) const
	{
		bool bInside;
		Real d = (origin - project(abox, bInside).origin).norm();
		return bInside == true ? -d : d;
	}

	COMM_FUNC Real Point3D::distance(const OrientedBox3D& obb) const
	{
		bool bInside;
		Real d = (origin - project(obb, bInside).origin).norm();
		return bInside == true ? -d : d;
	}

	COMM_FUNC Real Point3D::distance(const Capsule3D& capsule) const
	{
		bool bInside;
		Real d = (origin - project(capsule, bInside).origin).norm();
		return bInside == true ? -d : d;
	}

	COMM_FUNC Real Point3D::distanceSquared(const Point3D& pt) const
	{
		return (origin - pt.origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Line3D& line) const
	{
		return (origin - project(line).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Ray3D& ray) const
	{
		return (origin - project(ray).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Segment3D& segment) const
	{
		return (origin - project(segment).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Plane3D& plane) const
	{
		return (origin - project(plane).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Triangle3D& triangle) const
	{
		return (origin - project(triangle).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Rectangle3D& rectangle) const
	{
		return (origin - project(rectangle).origin).normSquared();
	}


	COMM_FUNC Real Point3D::distanceSquared(const Disk3D& disk) const
	{
		return (origin - project(disk).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Sphere3D& sphere) const
	{
		return (origin - project(sphere).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const AlignedBox3D& abox) const
	{
		return (origin - project(abox).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const OrientedBox3D& obb) const
	{
		return (origin - project(obb).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Tet3D& tet) const
	{
		return (origin - project(tet).origin).normSquared();
	}

	COMM_FUNC Real Point3D::distanceSquared(const Capsule3D& capsule) const
	{
		return (origin - project(capsule).origin).normSquared();
	}

	COMM_FUNC bool Point3D::inside(const Line3D& line) const
	{
		if (!line.isValid())
		{
			return false;
		}

		return (origin - line.origin).cross(line.direction).normSquared() < REAL_EPSILON_SQUARED;
	}

	COMM_FUNC bool Point3D::inside(const Ray3D& ray) const
	{
		if (!inside(Line3D(ray.origin, ray.direction)))
		{
			return false;
		}

		Coord3D offset = origin - ray.origin;
		Real t = offset.dot(ray.direction);

		return t > Real(0);
	}

	COMM_FUNC bool Point3D::inside(const Segment3D& segment) const
	{
		Coord3D dir = segment.direction();
		if (!inside(Line3D(segment.startPoint(), dir)))
		{
			return false;
		}

		Coord3D offset = origin - segment.startPoint();
		Real t = offset.dot(dir) / dir.normSquared();

		return t > Real(0) && t < Real(1);
	}

	COMM_FUNC bool Point3D::inside(const Plane3D& plane) const
	{
		if (!plane.isValid())
		{
			return false;
		}

		return abs((origin - plane.origin).dot(plane.normal)) < REAL_EPSILON;
	}

	COMM_FUNC bool Point3D::inside(const Triangle3D& triangle) const
	{
		Plane3D plane(triangle.v[0], triangle.normal());
		if (!inside(plane))
		{
			return false;
		}

		Triangle3D::Param tParam;
		bool bValid = triangle.computeBarycentrics(origin, tParam);
		if (bValid)
		{
			return tParam.u > Real(0) && tParam.u < Real(1) && tParam.v > Real(0) && tParam.v < Real(1) && tParam.w > Real(0) && tParam.w < Real(1);
		}
		else
		{
			return false;
		}
	}

	COMM_FUNC bool Point3D::inside(const Rectangle3D& rectangle) const
	{
		Plane3D plane(rectangle.center, rectangle.normal());
		if (!inside(plane))
		{
			return false;
		}

		Rectangle3D::Param recParam;
		bool bValid = rectangle.computeParams(origin, recParam);
		if (bValid)
		{
			return recParam.u < rectangle.extent[0] && recParam.u > -rectangle.extent[0] && recParam.v < rectangle.extent[1] && recParam.v > -rectangle.extent[1];
		}
		else
		{
			return false;
		}
	}

	COMM_FUNC bool Point3D::inside(const Disk3D& disk) const
	{
		Plane3D plane(disk.center, disk.normal);
		if (!inside(plane))
		{
			return false;
		}

		return (origin - disk.center).normSquared() < disk.radius*disk.radius;
	}

	COMM_FUNC bool Point3D::inside(const Sphere3D& sphere) const
	{
		return (origin - sphere.center).normSquared() < sphere.radius*sphere.radius;
	}

	COMM_FUNC bool Point3D::inside(const Capsule3D& capsule) const
	{
		return distanceSquared(capsule.segment) < capsule.radius * capsule.radius;
	}

	COMM_FUNC bool Point3D::inside(const Tet3D& tet) const
	{
		bool bInside = true;

		Triangle3D face;
		face = tet.face(0);
		bInside &= (origin - face.v[0]).dot(face.normal()) < 0;

		face = tet.face(1);
		bInside &= (origin - face.v[0]).dot(face.normal()) < 0;

		face = tet.face(2);
		bInside &= (origin - face.v[0]).dot(face.normal()) < 0;

		face = tet.face(3);
		bInside &= (origin - face.v[0]).dot(face.normal()) < 0;
		
		return bInside;
	}

	COMM_FUNC bool Point3D::inside(const OrientedBox3D& obb) const
	{
		Coord3D offset = origin - obb.center;
		Coord3D pPrime(offset.dot(obb.u), offset.dot(obb.v), offset.dot(obb.w));

		bool bInside = true;
		bInside &= pPrime[0] < obb.extent[0] && pPrime[0] >  -obb.extent[0];
		bInside &= pPrime[1] < obb.extent[1] && pPrime[1] >  -obb.extent[1];
		bInside &= pPrime[2] < obb.extent[2] && pPrime[2] >  -obb.extent[2];
		return bInside;
	}

	COMM_FUNC bool Point3D::inside(const AlignedBox3D& box) const
	{
		Coord3D offset = origin - box.v0;
		Coord3D extent = box.v1 - box.v0;

		bool bInside = true;
		bInside &= offset[0] < extent[0] && offset[0] >  Real(0);
		bInside &= offset[1] < extent[1] && offset[1] >  Real(0);
		bInside &= offset[2] < extent[2] && offset[2] >  Real(0);

		return bInside;
	}

	COMM_FUNC const Segment3D Point3D::operator-(const Point3D& pt) const
	{
		return Segment3D(pt.origin, origin);
	}

	COMM_FUNC Line3D::Line3D()
	{
		origin = Coord3D(0);
		direction = Coord3D(1, 0, 0);
	}

	COMM_FUNC Line3D::Line3D(const Coord3D& pos, const Coord3D& dir)
	{
		origin = pos;
		direction = dir;
	}

	COMM_FUNC Line3D::Line3D(const Line3D& line)
	{
		origin = line.origin;
		direction = line.direction;
	}


	COMM_FUNC Segment3D Line3D::proximity(const Line3D& line) const
	{
		Coord3D u = origin - line.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(line.direction);
		Real c = line.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(line.direction);
		Real f = u.normSquared();
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			Point3D p = Point3D(line.origin);
			return c < REAL_EPSILON ? p - p.project(*this) : Segment3D(origin, line.origin + e / c * line.direction);
		}
		else
		{
			Real invDet = 1 / det;
			Real s = (b * e - c * d) * invDet;
			Real t = (a * e - b * d) * invDet;
			return Segment3D(origin + s * direction, line.origin + t * line.direction);
		}
	}

	COMM_FUNC Segment3D Line3D::proximity(const Ray3D& ray) const
	{
		Coord3D u = origin - ray.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(ray.direction);
		Real c = ray.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(ray.direction);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			Point3D p0(origin);
			Point3D p1(ray.origin);

			return a < REAL_EPSILON ? p0.project(*this) - p0 : p1 - p1.project(*this);
		}

		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		if (tNum < 0) {
			tNum = 0;
			sNum = -d;
			sDenom = a;
		}
		// Parameters of nearest points on restricted domain
		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return Segment3D(origin + (s * direction), ray.origin + (t * ray.direction));
	}

	COMM_FUNC Segment3D Line3D::proximity(const Segment3D& segment) const
	{
		Coord3D u = origin - segment.startPoint();
		Coord3D dir1 = segment.endPoint() - segment.startPoint();
		Real a = direction.normSquared();
		Real b = direction.dot(dir1);
		Real c = dir1.dot(dir1);
		Real d = u.dot(direction);
		Real e = u.dot(dir1);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			Point3D p0(origin);
			Point3D p1(segment.startPoint());

			return a < REAL_EPSILON ? p0.project(*this) - p0 : p1 - p1.project(*this);
		}

		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		// Check t
		if (tNum < 0) {
			tNum = 0;
			sNum = -d;
			sDenom = a;
		}
		else if (tNum > tDenom) {
			tNum = tDenom;
			sNum = -d + b;
			sDenom = a;
		}
		// Parameters of nearest points on restricted domain
		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return Segment3D(origin + (s * direction), segment.startPoint() + (t * dir1));
	}

	COMM_FUNC Segment3D Line3D::proximity(const Triangle3D& triangle) const
	{
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Coord3D normal = e0.cross(e1);
		Real NdD = normal.dot(direction);
		if (abs(NdD) > REAL_EPSILON)
		{
			// The line and triangle are not parallel, so the line
			// intersects/ the plane of the triangle.
			Coord3D diff = origin - triangle.v[0];

			//compute a right - handed orthonormal basis
			Coord3D W = direction;
			W.normalize();
			Coord3D U = W[0] > W[1] ? Coord3D(-W[2], (Real)0, W[0]) : Coord3D((Real)0, W[2], -W[1]);
			Coord3D V = W.cross(U);

			Real UdE0 = U.dot(e0);
			Real UdE1 = U.dot(e1);
			Real UdDiff = U.dot(diff);
			Real VdE0 = V.dot(e0);
			Real VdE1 = V.dot(e1);
			Real VdDiff = V.dot(diff);
			Real invDet = ((Real)1) / (UdE0 * VdE1 - UdE1 * VdE0);

			// Barycentric coordinates for the point of intersection.
			Real u = (VdE1 * UdDiff - UdE1 * VdDiff) * invDet;
			Real v = (UdE0 * VdDiff - VdE0 * UdDiff) * invDet;
			Real b0 = (Real)1 - u - v;

			if (b0 >= (Real)0 && u >= (Real)0 && v >= (Real)0)
			{
				// Line parameter for the point of intersection.
				Real DdE0 = direction.dot(e0);
				Real DdE1 = direction.dot(e1);
				Real DdDiff = direction.dot(diff);
				Real t = u * DdE0 + v * DdE1 - DdDiff;

				return Segment3D(origin + t * direction, triangle.v[0] + u * e0 + v * e1);
			}
		}

		if (direction.normSquared() < EPSILON)
		{
			Point3D p(origin);
			return p.project(triangle) - p;
		}
		else
		{
			// Treat line and triangle as parallel. Compute
			// closest points by computing distance from
			// line to each triangle edge and taking minimum.
			Segment3D e0(triangle.v[0], triangle.v[1]);
			Segment3D minPQ = proximity(e0);
			Real minDS = minPQ.lengthSquared();

			Segment3D e1(triangle.v[0], triangle.v[2]);
			Segment3D pq1 = proximity(e1);
			if (pq1.lengthSquared() < minDS)
			{
				minPQ = pq1;
				minDS = pq1.lengthSquared();
			}

			Segment3D e2(triangle.v[1], triangle.v[2]);
			Segment3D pq2 = proximity(e2);
			if (pq2.lengthSquared() < minDS)
			{
				minPQ = pq2;
			}

			return minPQ;
		}
	}

	COMM_FUNC Segment3D Line3D::proximity(const Rectangle3D& rectangle) const
	{
		Coord3D N = rectangle.axis[0].cross(rectangle.axis[1]);
		Real NdD = N.dot(direction);
		if (abs(NdD) > REAL_EPSILON)
		{
			// The line and rectangle are not parallel, so the line
			// intersects the plane of the rectangle.
			Coord3D diff = origin - rectangle.center;
			Coord3D W = direction;
			W.normalize();
			Coord3D U = W[0] > W[1] ? Coord3D(-W[2], (Real)0, W[0]) : Coord3D((Real)0, W[2], -W[1]);
			Coord3D V = W.cross(U);
			Real UdD0 = U.dot(rectangle.axis[0]);
			Real UdD1 = U.dot(rectangle.axis[1]);
			Real UdPmC = U.dot(diff);
			Real VdD0 = V.dot(rectangle.axis[0]);
			Real VdD1 = V.dot(rectangle.axis[1]);
			Real VdPmC = V.dot(diff);
			Real invDet = ((Real)1) / (UdD0 * VdD1 - UdD1 * VdD0);

			// Rectangle coordinates for the point of intersection.
			Real s0 = (VdD1 * UdPmC - UdD1 * VdPmC) * invDet;
			Real s1 = (UdD0 * VdPmC - VdD0 * UdPmC) * invDet;

			if (abs(s0) <= rectangle.extent[0] && abs(s1) <= rectangle.extent[1])
			{
				// Line parameter for the point of intersection.
				Real DdD0 = direction.dot(rectangle.axis[0]);
				Real DdD1 = direction.dot(rectangle.axis[1]);
				Real DdDiff = direction.dot(diff);
				Real t = s0 * DdD0 + s1 * DdD1 - DdDiff;

				return Segment3D(origin + t * direction, rectangle.center + s0 * rectangle.axis[0] + s1 * rectangle.axis[1]);
			}
		}

		Real minDS = REAL_MAX;
		Segment3D minPQ;

		for (int i = 0; i < 4; i++)
		{
			Segment3D pq = proximity(rectangle.edge(i));
			if (pq.lengthSquared() < minDS)
			{
				minDS = pq.lengthSquared();
				minPQ = pq;
			}
		}
		
		return minPQ;
	}

	COMM_FUNC Segment3D Line3D::proximity(const Sphere3D& sphere) const
	{
		Coord3D offset = sphere.center - origin;
		Real d2 = direction.normSquared();
		if (d2 < REAL_EPSILON)
		{
			return Point3D(origin).project(sphere) - Point3D(origin);
		}

		Coord3D p = origin + offset.dot(direction) / d2 * direction;

		return Point3D(p).project(sphere) - Point3D(p);
	}

	COMM_FUNC Segment3D Line3D::proximity(const AlignedBox3D& box) const
	{
		Coord3D boxCenter, boxExtent;
		boxCenter = 0.5 * (box.v0 + box.v1);
		boxExtent = 0.5 * (box.v1 - box.v0);
		Coord3D point = origin - boxCenter;
		Coord3D lineDir = direction;

		Real t = Real(0);

		auto case00 = [](Coord3D &pt, Real& t, const Coord3D dir, const Coord3D extent, const int axis) {
			t = (extent[axis] - pt[axis]) / dir[axis];
			pt[axis] = extent[axis];

			pt = clamp(pt, -extent, extent);
		};

		auto case0 = [](Coord3D &pt, Real& t, const Coord3D dir, const Coord3D extent, const int i0, const int i1, const int i2) {
			Real PmE0 = pt[i0] - extent[i0];
			Real PmE1 = pt[i1] - extent[i1];
			Real prod0 = dir[i1] * PmE0;
			Real prod1 = dir[i0] * PmE1;

			if (prod0 >= prod1)
			{
				// line intersects P[i0] = e[i0]
				pt[i0] = extent[i0];

				Real PpE1 = pt[i1] + extent[i1];
				Real delta = prod0 - dir[i0] * PpE1;
				if (delta >= (Real)0)
				{
					Real invLSqr = ((Real)1) / (dir[i0] * dir[i0] + dir[i1] * dir[i1]);
					pt[i1] = -extent[i1];
					t = -(dir[i0] * PmE0 + dir[i1] * PpE1) * invLSqr;
				}
				else
				{
					Real inv = ((Real)1) / dir[i0];
					pt[i1] -= prod0 * inv;
					t = -PmE0 * inv;
				}
			}
			else
			{
				// line intersects P[i1] = e[i1]
				pt[i1] = extent[i1];

				Real PpE0 = pt[i0] + extent[i0];
				Real delta = prod1 - dir[i1] * PpE0;
				if (delta >= (Real)0)
				{
					Real invLSqr = ((Real)1) / (dir[i0] * dir[i0] + dir[i1] * dir[i1]);
					pt[i0] = -extent[i0];
					t = -(dir[i0] * PpE0 + dir[i1] * PmE1) * invLSqr;
				}
				else
				{
					Real inv = ((Real)1) / dir[i1];
					pt[i0] -= prod1 * inv;
					t = -PmE1 * inv;
				}
			}

			if (pt[i2] < -extent[i2])
			{
				pt[i2] = -extent[i2];
			}
			else if (pt[i2] > extent[i2])
			{
				pt[i2] = extent[i2];
			}
		};

		auto face = [](	Coord3D& pnt, Real& final_t,
						const Coord3D& dir, const Coord3D& PmE, const Coord3D& boxExtent,
						int i0, int i1, int i2)
		{
			Coord3D PpE;
			Real lenSqr, inv, tmp, param, t, delta;

			PpE[i1] = pnt[i1] + boxExtent[i1];
			PpE[i2] = pnt[i2] + boxExtent[i2];
			if (dir[i0] * PpE[i1] >= dir[i1] * PmE[i0])
			{
				if (dir[i0] * PpE[i2] >= dir[i2] * PmE[i0])
				{
					// v[i1] >= -e[i1], v[i2] >= -e[i2] (distance = 0)
					pnt[i0] = boxExtent[i0];
					inv = ((Real)1) / dir[i0];
					pnt[i1] -= dir[i1] * PmE[i0] * inv;
					pnt[i2] -= dir[i2] * PmE[i0] * inv;
					final_t = -PmE[i0] * inv;
				}
				else
				{
					// v[i1] >= -e[i1], v[i2] < -e[i2]
					lenSqr = dir[i0] * dir[i0] + dir[i2] * dir[i2];
					tmp = lenSqr * PpE[i1] - dir[i1] * (dir[i0] * PmE[i0] +
						dir[i2] * PpE[i2]);
					if (tmp <= ((Real)2) * lenSqr * boxExtent[i1])
					{
						t = tmp / lenSqr;
						lenSqr += dir[i1] * dir[i1];
						tmp = PpE[i1] - t;
						delta = dir[i0] * PmE[i0] + dir[i1] * tmp + dir[i2] * PpE[i2];
						param = -delta / lenSqr;

						final_t = param;
						pnt[i0] = boxExtent[i0];
						pnt[i1] = t - boxExtent[i1];
						pnt[i2] = -boxExtent[i2];
					}
					else
					{
						lenSqr += dir[i1] * dir[i1];
						delta = dir[i0] * PmE[i0] + dir[i1] * PmE[i1] + dir[i2] * PpE[i2];
						param = -delta / lenSqr;

						final_t = param;
						pnt[i0] = boxExtent[i0];
						pnt[i1] = boxExtent[i1];
						pnt[i2] = -boxExtent[i2];
					}
				}
			}
			else
			{
				if (dir[i0] * PpE[i2] >= dir[i2] * PmE[i0])
				{
					// v[i1] < -e[i1], v[i2] >= -e[i2]
					lenSqr = dir[i0] * dir[i0] + dir[i1] * dir[i1];
					tmp = lenSqr * PpE[i2] - dir[i2] * (dir[i0] * PmE[i0] +
						dir[i1] * PpE[i1]);
					if (tmp <= ((Real)2) * lenSqr * boxExtent[i2])
					{
						t = tmp / lenSqr;
						lenSqr += dir[i2] * dir[i2];
						tmp = PpE[i2] - t;
						delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * tmp;
						param = -delta / lenSqr;

						final_t = param;
						pnt[i0] = boxExtent[i0];
						pnt[i1] = -boxExtent[i1];
						pnt[i2] = t - boxExtent[i2];
					}
					else
					{
						lenSqr += dir[i2] * dir[i2];
						delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * PmE[i2];
						param = -delta / lenSqr;

						final_t = param;
						pnt[i0] = boxExtent[i0];
						pnt[i1] = -boxExtent[i1];
						pnt[i2] = boxExtent[i2];
					}
				}
				else
				{
					// v[i1] < -e[i1], v[i2] < -e[i2]
					lenSqr = dir[i0] * dir[i0] + dir[i2] * dir[i2];
					tmp = lenSqr * PpE[i1] - dir[i1] * (dir[i0] * PmE[i0] +
						dir[i2] * PpE[i2]);
					if (tmp >= (Real)0)
					{
						// v[i1]-edge is closest
						if (tmp <= ((Real)2) * lenSqr * boxExtent[i1])
						{
							t = tmp / lenSqr;
							lenSqr += dir[i1] * dir[i1];
							tmp = PpE[i1] - t;
							delta = dir[i0] * PmE[i0] + dir[i1] * tmp + dir[i2] * PpE[i2];
							param = -delta / lenSqr;

							final_t = param;
							pnt[i0] = boxExtent[i0];
							pnt[i1] = t - boxExtent[i1];
							pnt[i2] = -boxExtent[i2];
						}
						else
						{
							lenSqr += dir[i1] * dir[i1];
							delta = dir[i0] * PmE[i0] + dir[i1] * PmE[i1]
								+ dir[i2] * PpE[i2];
							param = -delta / lenSqr;

							final_t = param;
							pnt[i0] = boxExtent[i0];
							pnt[i1] = boxExtent[i1];
							pnt[i2] = -boxExtent[i2];
						}
						return;
					}

					lenSqr = dir[i0] * dir[i0] + dir[i1] * dir[i1];
					tmp = lenSqr * PpE[i2] - dir[i2] * (dir[i0] * PmE[i0] +
						dir[i1] * PpE[i1]);
					if (tmp >= (Real)0)
					{
						// v[i2]-edge is closest
						if (tmp <= ((Real)2) * lenSqr * boxExtent[i2])
						{
							t = tmp / lenSqr;
							lenSqr += dir[i2] * dir[i2];
							tmp = PpE[i2] - t;
							delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * tmp;
							param = -delta / lenSqr;

							final_t = param;
							pnt[i0] = boxExtent[i0];
							pnt[i1] = -boxExtent[i1];
							pnt[i2] = t - boxExtent[i2];
						}
						else
						{
							lenSqr += dir[i2] * dir[i2];
							delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1]
								+ dir[i2] * PmE[i2];
							param = -delta / lenSqr;

							final_t = param;
							pnt[i0] = boxExtent[i0];
							pnt[i1] = -boxExtent[i1];
							pnt[i2] = boxExtent[i2];
						}
						return;
					}

					// (v[i1],v[i2])-corner is closest
					lenSqr += dir[i2] * dir[i2];
					delta = dir[i0] * PmE[i0] + dir[i1] * PpE[i1] + dir[i2] * PpE[i2];
					param = -delta / lenSqr;

					final_t = param;
					pnt[i0] = boxExtent[i0];
					pnt[i1] = -boxExtent[i1];
					pnt[i2] = -boxExtent[i2];
				}
			}
		};


		// Apply reflections so that direction vector has nonnegative
		// components.
		bool reflect[3];
		for (int i = 0; i < 3; ++i)
		{
			if (lineDir[i] < (Real)0)
			{
				point[i] = -point[i];
				lineDir[i] = -lineDir[i];
				reflect[i] = true;
			}
			else
			{
				reflect[i] = false;
			}
		}

		if (lineDir[0] > REAL_EPSILON)
		{
			if (lineDir[1] > REAL_EPSILON)
			{
				if (lineDir[2] > REAL_EPSILON)  // (+,+,+)
				{
					//CaseNoZeros(point, dir, boxExtent, t);
					Coord3D PmE = point - boxExtent;
					Real prodDxPy = lineDir[0] * PmE[1];
					Real prodDyPx = lineDir[1] * PmE[0];
					Real prodDzPx, prodDxPz, prodDzPy, prodDyPz;

					if (prodDyPx >= prodDxPy)
					{
						prodDzPx = lineDir[2] * PmE[0];
						prodDxPz = lineDir[0] * PmE[2];
						if (prodDzPx >= prodDxPz)
						{
							// line intersects x = e0
							face(point, t, lineDir, PmE, boxExtent, 0, 1, 2);
						}
						else
						{
							// line intersects z = e2
							face(point, t, lineDir, PmE, boxExtent, 2, 0, 1);
						}
					}
					else
					{
						prodDzPy = lineDir[2] * PmE[1];
						prodDyPz = lineDir[1] * PmE[2];
						if (prodDzPy >= prodDyPz)
						{
							// line intersects y = e1
							face(point, t, lineDir, PmE, boxExtent, 1, 2, 0);
						}
						else
						{
							// line intersects z = e2
							face(point, t, lineDir, PmE, boxExtent, 2, 0, 1);
						}
					}
				}
				else  // (+,+,0)
				{
					//Case0(0, 1, 2, point, dir, boxExtent, t);
					case0(point, t, lineDir, boxExtent, 0, 1, 2);
				}
			}
			else
			{
				if (lineDir[2] > REAL_EPSILON)  // (+,0,+)
				{
					//Case0(0, 2, 1, point, dir, boxExtent, t);
					case0(point, t, lineDir, boxExtent, 0, 2, 1);
				}
				else  // (+,0,0)
				{
					//Case00(0, 1, 2, point, dir, boxExtent, t);
					case00(point, t, lineDir, boxExtent, 0);
				}
			}
		}
		else
		{
			if (lineDir[1] > REAL_EPSILON)
			{
				if (lineDir[2] > REAL_EPSILON)  // (0,+,+)
				{
					//Case0(1, 2, 0, point, dir, boxExtent, t);
					case0(point, t, lineDir, boxExtent, 1, 2, 0);
				}
				else  // (0,+,0)
				{
					//Case00(1, 0, 2, point, dir, boxExtent, t);
					case00(point, t, lineDir, boxExtent, 1);
				}
			}
			else
			{
				if (lineDir[2] > REAL_EPSILON)  // (0,0,+)
				{
					//Case00(2, 0, 1, point, dir, boxExtent, t);
					case00(point, t, lineDir, boxExtent, 2);
				}
				else  // (0,0,0)
				{
					point = clamp(point, -boxExtent, boxExtent);
					//Case000(point, boxExtent);
				}
			}
		}

		// Undo the reflections applied previously.
		for (int i = 0; i < 3; ++i)
		{
			if (reflect[i])
			{
				point[i] = -point[i];
			}
		}

		return Segment3D(origin + t * direction, boxCenter + point);
	}

	COMM_FUNC Segment3D Line3D::proximity(const OrientedBox3D& obb) const
	{
		//transform to the local coordinate system of obb
		Coord3D diff = origin - obb.center;
		Coord3D originPrime = Coord3D(diff.dot(obb.u), diff.dot(obb.v), diff.dot(obb.w));
		Coord3D dirPrime = Coord3D(direction.dot(obb.u), direction.dot(obb.v), direction.dot(obb.w));

		Segment3D pqPrime = Line3D(originPrime, dirPrime).proximity(AlignedBox3D(obb.center - obb.extent, obb.center + obb.extent));

		Coord3D pPrime = pqPrime.startPoint();
		Coord3D qPrime = pqPrime.endPoint();

		//transform back to the global coordinate system
		Coord3D p = pPrime[0] * obb.u + pPrime[1] * obb.v + pPrime[2] * obb.w;
		Coord3D q = qPrime[0] * obb.u + qPrime[1] * obb.v + qPrime[2] * obb.w;

		return Segment3D(p, q);
	}

	// 	COMM_FUNC Segment3D Line3D::proximity(const Segment3D& segment) const
// 	{
// 
// 	}

	COMM_FUNC Real Line3D::distance(const Point3D& pt) const
	{
		return pt.distance(*this);
	}

	COMM_FUNC Real Line3D::distance(const Line3D& line) const
	{
		return proximity(line).length();
	}

	COMM_FUNC Real Line3D::distance(const Ray3D& ray) const
	{
		return proximity(ray).length();
	}

	COMM_FUNC Real Line3D::distance(const Segment3D& segment) const
	{
		return proximity(segment).length();
	}

	COMM_FUNC Real Line3D::distance(const AlignedBox3D& box) const
	{
		return proximity(box).length();
	}

	COMM_FUNC Real Line3D::distance(const OrientedBox3D& obb) const
	{
		return proximity(obb).length();
	}

	COMM_FUNC Real Line3D::distanceSquared(const Point3D& pt) const
	{
		return pt.distanceSquared(*this);
	}

	COMM_FUNC Real Line3D::distanceSquared(const Line3D& line) const
	{
		return proximity(line).lengthSquared();
// 		Coord3D u = origin - line.origin;
// 		Real a = direction.normSquared();
// 		Real b = direction.dot(line.direction);
// 		Real c = line.direction.normSquared();
// 		Real d = u.dot(direction);
// 		Real e = u.dot(line.direction);
// 		Real f = u.normSquared();
// 		Real det = a * c - b * b;
// 
// 		if (det < REAL_EPSILON)
// 		{
// 			return f - e * e;
// 		}
// 		else
// 		{
// 			Real invDet = 1 / det;
// 			Real s = (b * e - c * d) * invDet;
// 			Real t = (a * e - b * d) * invDet;
// 			return s * (a * s - b * t + 2*d) - t * (b * s - c * t + 2*e) + f;
// 		}
	}


	COMM_FUNC Real Line3D::distanceSquared(const Ray3D& ray) const
	{
		return proximity(ray).lengthSquared();
	}

	COMM_FUNC Real Line3D::distanceSquared(const Segment3D& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	COMM_FUNC Real Line3D::distanceSquared(const AlignedBox3D& box) const
	{
		return proximity(box).lengthSquared();
	}

	COMM_FUNC Real Line3D::distanceSquared(const OrientedBox3D& obb) const
	{
		return proximity(obb).lengthSquared();
	}

	COMM_FUNC int Line3D::intersect(const Plane3D& plane, Point3D& interPt) const
	{
		Real DdN = direction.dot(plane.normal);
		if (abs(DdN) < REAL_EPSILON)
		{
			return 0;
		}

		Coord3D offset = origin - plane.origin;
		Real t = -offset.dot(plane.normal) / DdN;

		interPt.origin = origin + t * direction;
		return 1;
	}

	COMM_FUNC int Line3D::intersect(const Triangle3D& triangle, Point3D& interPt) const
	{
		Coord3D diff = origin - triangle.v[0];
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Coord3D normal = e0.cross(e1);

		Real DdN = direction.dot(normal);
		Real sign;
		if (DdN >= REAL_EPSILON)
		{
			sign = Real(1);
		}
		else if (DdN <= - REAL_EPSILON)
		{
			sign = (Real)-1;
			DdN = -DdN;
		}
		else
		{
			return 0;
		}

		Real DdQxE1 = sign * direction.dot(diff.cross(e1));
		if (DdQxE1 >= (Real)0)
		{
			Real DdE0xQ = sign * direction.dot(e0.cross(diff));
			if (DdE0xQ >= (Real)0)
			{
				if (DdQxE1 + DdE0xQ <= DdN)
				{
					// Line intersects triangle.
					Real QdN = -sign * diff.dot(normal);
					Real inv = (Real)1 / DdN;

					Real t = QdN * inv;
					interPt.origin = origin + t * direction;
					return 1;
				}
				// else: b1+b2 > 1, no intersection
			}
			// else: b2 < 0, no intersection
		}
		// else: b1 < 0, no intersection

		return 0;
	}

	COMM_FUNC int Line3D::intersect(const Sphere3D& sphere, Segment3D& interSeg) const
	{
		Coord3D diff = origin - sphere.center;
		Real a0 = diff.dot(diff) - sphere.radius * sphere.radius;
		Real a1 = direction.dot(diff);

		// Intersection occurs when Q(t) has real roots.
		Real discr = a1 * a1 - a0;
		if (discr > (Real)0)
		{
			Real root = sqrt(discr);
			interSeg.startPoint() = origin + (-a1 - root)*direction;
			interSeg.endPoint() = origin + (-a1 + root)*direction;
			return 2;
		}
		else if (discr < (Real)0)
		{
			return 0;
		}
		else
		{
			interSeg.startPoint() = origin - a1*direction;
			return 1;
		}
	}

	//TODO:
	COMM_FUNC int Line3D::intersect(const Tet3D& tet, Segment3D& interSeg) const
	{
		return 0;
	}


	COMM_FUNC int Line3D::intersect(const AlignedBox3D& abox, Segment3D& interSeg) const
	{
		if (!isValid())
		{
			return 0;
		}

		Real t0 = -REAL_MAX;
		Real t1 = REAL_MAX;

		auto clip = [](Real denom, Real numer, Real& t0, Real& t1) -> bool
		{
			if (denom > REAL_EPSILON)
			{
				if (numer > denom * t1)
				{
					return false;
				}
				if (numer > denom * t0)
				{
					t0 = numer / denom;
				}
				return true;
			}
			else if (denom < -REAL_EPSILON)
			{
				if (numer > denom * t0)
				{
					return false;
				}
				if (numer > denom * t1)
				{
					t1 = numer / denom;
				}
				return true;
			}
			else
			{
				return numer <= -REAL_EPSILON;
			}
		};

		Coord3D boxCenter = 0.5 * (abox.v0 + abox.v1);
		Coord3D boxExtent = 0.5 * (abox.v1 - abox.v0);

		Coord3D offset = origin - boxCenter;
		Coord3D lineDir = direction;
		lineDir.normalize();

		if (clip(+lineDir[0], -offset[0] - boxExtent[0], t0, t1) &&
			clip(-lineDir[0], +offset[0] - boxExtent[0], t0, t1) &&
			clip(+lineDir[1], -offset[1] - boxExtent[1], t0, t1) &&
			clip(-lineDir[1], +offset[1] - boxExtent[1], t0, t1) &&
			clip(+lineDir[2], -offset[2] - boxExtent[2], t0, t1) &&
			clip(-lineDir[2], +offset[2] - boxExtent[2], t0, t1))
		{
			if (t1 > t0)
			{
				interSeg.v0 = origin + t0 * lineDir;
				interSeg.v1 = origin + t1 * lineDir;
				return 2;
			}
			else
			{
				interSeg.v0 = origin + t0 * lineDir;
				interSeg.v1 = interSeg.v0;
				return 1;
			}
		}

		return 0;
	}

	COMM_FUNC Real Line3D::parameter(const Coord3D& pos) const
	{
		Coord3D l = pos - origin;
		Real d2 = direction.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(direction) / d2;
	}

	COMM_FUNC bool Line3D::isValid() const
	{
		return direction.normSquared() > REAL_EPSILON_SQUARED;
	}

	COMM_FUNC Ray3D::Ray3D()
	{
		origin = Coord3D(0);
		direction = Coord3D(1, 0, 0);
	}

	COMM_FUNC Ray3D::Ray3D(const Coord3D& pos, const Coord3D& dir)
	{
		origin = pos;
		direction = dir;
	}

	COMM_FUNC Ray3D::Ray3D(const Ray3D& ray)
	{
		origin = ray.origin;
		direction = ray.direction;
	}

	COMM_FUNC Segment3D Ray3D::proximity(const Ray3D& ray) const
	{
		Coord3D u = origin - ray.origin;
		Real a = direction.normSquared();
		Real b = direction.dot(ray.direction);
		Real c = ray.direction.normSquared();
		Real d = u.dot(direction);
		Real e = u.dot(ray.direction);
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			Point3D p0(origin);
			Point3D p1(ray.origin);

			Point3D q0 = p0.project(ray);
			Point3D q1 = p1.project(*this);

			return (q0 - p0).lengthSquared() < (q1 - p1).lengthSquared() ? q0 - p0 : p1 - q1;
		}

		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		if (sNum < 0) {
			sNum = 0;
			tNum = e;
			tDenom = c;
		}
		// Check t
		if (tNum < 0) {
			tNum = 0;
			if (-d < 0) {
				sNum = 0;
			}
			else {
				sNum = -d;
				sDenom = a;
			}
		}
		// Parameters of nearest points on restricted domain
		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return Segment3D(origin+ (s * direction), ray.origin + (t * direction));
	}

	COMM_FUNC Segment3D Ray3D::proximity(const Segment3D& segment) const
	{
		Coord3D u = origin - segment.startPoint();
		Real a = direction.normSquared();
		Real b = direction.dot(segment.direction());
		Real c = segment.lengthSquared();
		Real d = u.dot(direction);
		Real e = u.dot(segment.direction());
		Real det = a * c - b * b;

		if (det < REAL_EPSILON)
		{
			if (a < REAL_EPSILON_SQUARED)
			{
				Point3D p0(origin);
				return p0.project(segment) - p0;
			}
			else
			{
				Point3D p1(segment.startPoint());
				Point3D p2(segment.endPoint());

				Point3D q1 = p1.project(*this);
				Point3D q2 = p2.project(*this);

				return (p1 - q1).lengthSquared() < (p2 - q2).lengthSquared() ? (p1 - q1) : (p2 - q2);
			}
		}

		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		if (sNum < 0) {
			sNum = 0;
			tNum = e;
			tDenom = c;
		}

		// Check t
		if (tNum < 0) {
			tNum = 0;
			if (-d < 0) {
				sNum = 0;
			}
			else {
				sNum = -d;
				sDenom = a;
			}
		}
		else if (tNum > tDenom) {
			tNum = tDenom;
			if ((-d + b) < 0) {
				sNum = 0;
			}
			else {
				sNum = -d + b;
				sDenom = a;
			}
		}

		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return Segment3D(origin + (s * direction), segment.startPoint() + (t * segment.direction()));
	}

	COMM_FUNC Segment3D Ray3D::proximity(const Triangle3D& triangle) const
	{
		Line3D line(origin, direction);
		Segment3D pq = line.proximity(triangle);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
		{
			return Point3D(origin).project(triangle) - Point3D(origin);
		}

		return pq;
	}

	COMM_FUNC Segment3D Ray3D::proximity(const Rectangle3D& rectangle) const
	{
		Line3D line(origin, direction);
		Segment3D pq = line.proximity(rectangle);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
		{
			return Point3D(origin).project(rectangle) - Point3D(origin);
		}

		return pq;
	}

	COMM_FUNC Segment3D Ray3D::proximity(const AlignedBox3D& box) const
	{
		Line3D line(origin, direction);
		Segment3D pq = line.proximity(box);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
		{
			return Point3D(origin).project(box) - Point3D(origin);
		}

		return pq;
	}

	COMM_FUNC Segment3D Ray3D::proximity(const OrientedBox3D& obb) const
	{
		Line3D line(origin, direction);
		Segment3D pq = line.proximity(obb);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
		{
			return Point3D(origin).project(obb) - Point3D(origin);
		}

		return pq;
	}

	COMM_FUNC Real Ray3D::distance(const Point3D& pt) const
	{
		return pt.distance(*this);
	}

	COMM_FUNC Real Ray3D::distance(const Segment3D& segment) const
	{
		return proximity(segment).length();
	}

	COMM_FUNC Real Ray3D::distance(const Triangle3D& triangle) const
	{
		return proximity(triangle).length();
	}

	COMM_FUNC Real Ray3D::distanceSquared(const Point3D& pt) const
	{
		return pt.distanceSquared(*this);
	}

	COMM_FUNC Real Ray3D::distanceSquared(const Segment3D& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	COMM_FUNC Real Ray3D::distanceSquared(const Triangle3D& triangle) const
	{
		return proximity(triangle).lengthSquared();
	}

	COMM_FUNC int Ray3D::intersect(const Plane3D& plane, Point3D& interPt) const
	{
		Real DdN = direction.dot(plane.normal);
		if (abs(DdN) < REAL_EPSILON)
		{
			return 0;
		}

		Coord3D offset = origin - plane.origin;
		Real t = -offset.dot(plane.normal) / DdN;

		if (t < Real(0))
		{
			return 0;
		}

		interPt.origin = origin + t * direction;

		return 1;
	}

	COMM_FUNC int Ray3D::intersect(const Triangle3D& triangle, Point3D& interPt) const
	{
		Coord3D diff = origin - triangle.v[0];
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Coord3D normal = e0.cross(e1);

		Real DdN = direction.dot(normal);
		Real sign;
		if (DdN >= REAL_EPSILON)
		{
			sign = Real(1);
		}
		else if (DdN <= -REAL_EPSILON)
		{
			sign = (Real)-1;
			DdN = -DdN;
		}
		else
		{
			return 0;
		}

		Real DdQxE1 = sign * direction.dot(diff.cross(e1));
		if (DdQxE1 >= (Real)0)
		{
			Real DdE0xQ = sign * direction.dot(e0.cross(diff));
			if (DdE0xQ >= (Real)0)
			{
				if (DdQxE1 + DdE0xQ <= DdN)
				{
					// Line intersects triangle.
					Real QdN = -sign * diff.dot(normal);
					Real inv = (Real)1 / DdN;

					Real t = QdN * inv;

					if (t < Real(0))
					{
						return 0;
					}

					interPt.origin = origin + t * direction;
					return 1;
				}
				// else: b1+b2 > 1, no intersection
			}
			// else: b2 < 0, no intersection
		}
		// else: b1 < 0, no intersection

		return 0;
	}

	COMM_FUNC int Ray3D::intersect(const Sphere3D& sphere, Segment3D& interSeg) const
	{
		Coord3D diff = origin - sphere.center;
		Real a0 = diff.dot(diff) - sphere.radius * sphere.radius;
		Real a1 = direction.dot(diff);

		// Intersection occurs when Q(t) has real roots.
		Real discr = a1 * a1 - a0;
		if (discr > (Real)0)
		{
			Real root = sqrt(discr);

			if (-a1 + root < Real(0))
			{
				return 0;
			}
			else if (-a1 + root < Real(0))
			{
				interSeg.startPoint() = origin + (-a1 + root)*direction;
				return 1;
			}
			else
			{
				interSeg.startPoint() = origin + (-a1 - root)*direction;
				interSeg.endPoint() = origin + (-a1 + root)*direction;
				return 2;
			}
		}
		else if (discr < Real(0))
		{
			return 0;
		}
		else
		{
			if (a1 > Real(0))
			{
				return 0;
			}
			interSeg.startPoint() = origin - a1 * direction;
			return 1;
		}
	}

	COMM_FUNC int Ray3D::intersect(const AlignedBox3D& abox, Segment3D& interSeg) const
	{
		int interNum = Line3D(origin, direction).intersect(abox, interSeg);
		if (interNum == 0)
		{
			return 0;
		}

		Real t0 = parameter(interSeg.startPoint());
		Real t1 = parameter(interSeg.endPoint());

		Interval<Real> iRay(0, REAL_MAX, false, true);

		if (iRay.inside(t0))
		{
			interSeg.v0 = origin + iRay.leftLimit()*direction;
			interSeg.v1 = origin + iRay.rightLimit()*direction;
			return 2;
		} 
		else if (iRay.inside(t1))
		{
			interSeg.v0 = origin + iRay.leftLimit()*direction;
			interSeg.v1 = interSeg.v0;
			return 1;
		}
		else
		{
			return 0;
		}
	}

	COMM_FUNC Real Ray3D::parameter(const Coord3D& pos) const
	{
		Coord3D l = pos - origin;
		Real d2 = direction.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(direction) / d2;
	}

	COMM_FUNC bool Ray3D::isValid() const
	{
		return direction.normSquared() > REAL_EPSILON_SQUARED;
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

	COMM_FUNC Segment3D Segment3D::proximity(const Segment3D& segment) const
	{
		Coord3D u = v0 - segment.v0;
		Coord3D dir0 = v1 - v0;
		Coord3D dir1 = segment.v1 - segment.v0;
		Real a = dir0.normSquared();
		Real b = dir0.dot(dir1);
		Real c = dir1.normSquared();
		Real d = u.dot(dir0);
		Real e = u.dot(dir1);
		Real det = a * c - b * b;

		// Check for (near) parallelism
		if (det < REAL_EPSILON) {
			// Arbitrary choice
			Real l0 = lengthSquared();
			Real l1 = segment.lengthSquared();
			Point3D p0 = l0 < l1 ? Point3D(v0) : Point3D(segment.v0);
			Point3D p1 = l0 < l1 ? Point3D(v1) : Point3D(segment.v1);
			Segment3D longerSeg = l0 < l1 ? segment : *this;
			bool bOpposite = l0 < l1 ? false : true;
			Point3D q0 = p0.project(longerSeg);
			Point3D q1 = p1.project(longerSeg);
			Segment3D ret = p0.distance(q0) < p1.distance(q1) ? (q0 - p0) : (q1 - p1);
			return bOpposite ? -ret : ret;
		}

		// Find parameter values of closest points
		// on each segment��s infinite line. Denominator
		// assumed at this point to be ����det����,
		// which is always positive. We can check
		// value of numerators to see if we��re outside
		// the [0, 1] x [0, 1] domain.
		Real sNum = b * e - c * d;
		Real tNum = a * e - b * d;

		Real sDenom = det;
		Real tDenom = det;

		if (sNum < 0) {
			sNum = 0;
			tNum = e;
			tDenom = c;
		}
		else if (sNum > det) {
			sNum = det;
			tNum = e + b;
			tDenom = c;
		}

		// Check t
		if (tNum < 0) {
			tNum = 0;
			if (-d < 0) {
				sNum = 0;
			}
			else if (-d > a) {
				sNum = sDenom;
			}
			else {
				sNum = -d;
				sDenom = a;
			}
		}
		else if (tNum > tDenom) {
			tNum = tDenom;
			if ((-d + b) < 0) {
				sNum = 0;
			}
			else if ((-d + b) > a) {
				sNum = sDenom;
			}
			else {
				sNum = -d + b;
				sDenom = a;
			}
		}

		Real s = sNum / sDenom;
		Real t = tNum / tDenom;

		return Segment3D(v0 + (s * dir0), segment.v0 + (t * dir1));
	}

	COMM_FUNC Segment3D Segment3D::proximity(const Plane3D& plane) const
	{
		Point3D p0(v0);
		Point3D p1(v1);
		Point3D q0 = p0.project(plane);
		Point3D q1 = p1.project(plane);

		Segment3D pq0 = q0 - p0;
		Segment3D pq1 = q1 - p1;

		return pq0.lengthSquared() < pq1.lengthSquared() ? pq0 : pq1;
	}

	COMM_FUNC Segment3D Segment3D::proximity(const Triangle3D& triangle) const
	{
		Line3D line(startPoint(), direction());
		Segment3D pq = line.proximity(triangle);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
			return Point3D(startPoint()).project(triangle) - Point3D(startPoint());

		if (t > Real(1))
			return Point3D(endPoint()).project(triangle) - Point3D(endPoint());

		return pq;
	}

	COMM_FUNC Segment3D Segment3D::proximity(const Rectangle3D& rectangle) const
	{
		Line3D line(startPoint(), direction());
		Segment3D pq = line.proximity(rectangle);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
			return Point3D(startPoint()).project(rectangle) - Point3D(startPoint());

		if (t > Real(1))
			return Point3D(endPoint()).project(rectangle) - Point3D(endPoint());

		return pq;
	}

	COMM_FUNC Segment3D Segment3D::proximity(const AlignedBox3D& box) const
	{
		Line3D line(startPoint(), direction());
		Segment3D pq = line.proximity(box);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
			return Point3D(startPoint()).project(box) - Point3D(startPoint());

		if (t > Real(1))
			return Point3D(endPoint()).project(box) - Point3D(endPoint());

		return pq;
	}

	COMM_FUNC Segment3D Segment3D::proximity(const OrientedBox3D& obb) const
	{
		Line3D line(startPoint(), direction());
		Segment3D pq = line.proximity(obb);

		Real t = parameter(pq.startPoint());

		if (t < Real(0))
			return Point3D(startPoint()).project(obb) - Point3D(startPoint());

		if (t > Real(1))
			return Point3D(endPoint()).project(obb) - Point3D(endPoint());

		return pq;
	}

	COMM_FUNC Real Segment3D::distance(const Segment3D& segment) const
	{
		return proximity(segment).length();
	}

	COMM_FUNC Real Segment3D::distance(const Triangle3D& triangle) const
	{
		return proximity(triangle).length();
	}

	COMM_FUNC Real Segment3D::distanceSquared(const Segment3D& segment) const
	{
		return proximity(segment).lengthSquared();
	}

	COMM_FUNC Real Segment3D::distanceSquared(const Triangle3D& triangle) const
	{
		return proximity(triangle).lengthSquared();
	}

	COMM_FUNC bool Segment3D::intersect(const Plane3D& plane, Point3D& interPt) const
	{
		Coord3D dir = direction();
		Real DdN = dir.dot(plane.normal);
		if (abs(DdN) < REAL_EPSILON)
		{
			return false;
		}

		Coord3D offset = v0 - plane.origin;
		Real t = -offset.dot(plane.normal) / DdN;

		if (t < Real(0) || t > Real(1))
		{
			return false;
		}

		interPt.origin = v0 + t * dir;
		return true;
	}

	COMM_FUNC bool Segment3D::intersect(const Triangle3D& triangle, Point3D& interPt) const
	{
		Coord3D dir = direction();
		Coord3D diff = v0 - triangle.v[0];
		Coord3D e0 = triangle.v[1] - triangle.v[0];
		Coord3D e1 = triangle.v[2] - triangle.v[0];
		Coord3D normal = e0.cross(e1);

		Real DdN = dir.dot(normal);
		Real sign;
		if (DdN >= REAL_EPSILON)
		{
			sign = Real(1);
		}
		else if (DdN <= -REAL_EPSILON)
		{
			sign = (Real)-1;
			DdN = -DdN;
		}
		else
		{
			return false;
		}

		Real DdQxE1 = sign * dir.dot(diff.cross(e1));
		if (DdQxE1 >= (Real)0)
		{
			Real DdE0xQ = sign * dir.dot(e0.cross(diff));
			if (DdE0xQ >= (Real)0)
			{
				if (DdQxE1 + DdE0xQ <= DdN)
				{
					// Line intersects triangle.
					Real QdN = -sign * diff.dot(normal);
					Real inv = (Real)1 / DdN;

					Real t = QdN * inv;

					if (t < Real(0) || t > Real(1))
					{
						return false;
					}

					interPt.origin = v0 + t * dir;
					return true;
				}
				// else: b1+b2 > 1, no intersection
			}
			// else: b2 < 0, no intersection
		}
		// else: b1 < 0, no intersection

		return false;
	}

	COMM_FUNC int Segment3D::intersect(const Sphere3D& sphere, Segment3D& interSeg) const
	{
		Coord3D diff = v0 - sphere.center;
		Coord3D dir = direction();
		Real a0 = diff.dot(diff) - sphere.radius * sphere.radius;
		Real a1 = dir.dot(diff);

		// Intersection occurs when Q(t) has real roots.
		Real discr = a1 * a1 - a0;
		if (discr > (Real)0)
		{
			Real root = sqrt(discr);
			Real t1 = max(-a1 - root, Real(0));
			Real t2 = min(-a1 + root, Real(1));
			if (t1 < t2)
			{
				interSeg.startPoint() = v0 + t1 * dir;
				interSeg.endPoint() = v0 + t2 * dir;
				return 2;
			}
			else if (t1 > t2)
			{
				return 0;
			}
			else
			{
				interSeg.startPoint() = v0 + t1 * dir;
				return 1;
			}
		}
		else if (discr < (Real)0)
		{
			return 0;
		}
		else
		{
			Real t = -a1;
			if (t >= Real(0) && t <= Real(1))
			{
				interSeg.startPoint() = v0 - a1 * dir;
				return 1;
			}
			return 0;
		}
	}

	COMM_FUNC int Segment3D::intersect(const AlignedBox3D& abox, Segment3D& interSeg) const
	{
		Coord3D lineDir = direction();
		int interNum = Line3D(v0, lineDir).intersect(abox, interSeg);
		if (interNum == 0)
		{
			return 0;
		}

		Real t0 = parameter(interSeg.startPoint());
		Real t1 = parameter(interSeg.endPoint());

		Interval<Real> iSeg(0, 1, false, false);

		if (iSeg.inside(t0) && iSeg.inside(t1))
		{
			interSeg.v0 = v0 + t0*lineDir;
			interSeg.v1 = v0 + t1*lineDir;
			return 2;
		}
		else if (iSeg.inside(t1))
		{
			interSeg.v0 = v0 + t1*lineDir;
			interSeg.v1 = interSeg.v0;
			return 1;
		}
		else if (iSeg.inside(t0))
		{
			interSeg.v0 = v0 + t0 * lineDir;
			interSeg.v1 = interSeg.v0;
			return 1;
		}
		else
		{
			return 0;
		}
	}

	COMM_FUNC Real Segment3D::length() const
	{
		return (v1 - v0).norm();
	}

	COMM_FUNC Real Segment3D::lengthSquared() const
	{
		return (v1 - v0).normSquared();
	}

	COMM_FUNC Real Segment3D::parameter(const Coord3D& pos) const
	{
		Coord3D l = pos - v0;
		Coord3D dir = direction();
		Real d2 = dir.normSquared();

		return d2 < REAL_EPSILON_SQUARED ? Real(0) : l.dot(dir) / d2;
	}

	COMM_FUNC Segment3D Segment3D::operator-(void) const
	{
		Segment3D seg;
		seg.v0 = v1;
		seg.v1 = v0;

		return seg;
	}

	COMM_FUNC bool Segment3D::isValid() const
	{
		return lengthSquared() >= REAL_EPSILON_SQUARED;
	}

	COMM_FUNC Plane3D::Plane3D()
	{
		origin = Coord3D(0);
		normal = Coord3D(0, 1, 0);
	}

	COMM_FUNC Plane3D::Plane3D(const Coord3D& pos, const Coord3D n)
	{
		origin = pos;
		normal = n;
	}

	COMM_FUNC Plane3D::Plane3D(const Plane3D& plane)
	{
		origin = plane.origin;
		normal = plane.normal;
	}

	COMM_FUNC bool Plane3D::isValid() const
	{
		return normal.normSquared() >= REAL_EPSILON;
	}

	COMM_FUNC Triangle3D::Triangle3D()
	{
		v[0] = Coord3D(0);
		v[1] = Coord3D(1, 0, 0);
		v[2] = Coord3D(0, 0, 1);
	}

	COMM_FUNC Triangle3D::Triangle3D(const Coord3D& p0, const Coord3D& p1, const Coord3D& p2)
	{
		v[0] = p0;
		v[1] = p1;
		v[2] = p2;
	}

	COMM_FUNC Triangle3D::Triangle3D(const Triangle3D& triangle)
	{
		v[0] = triangle.v[0];
		v[1] = triangle.v[1];
		v[2] = triangle.v[2];
	}

	COMM_FUNC Real Triangle3D::area() const
	{
		return Real(0.5)*((v[1] - v[0]).cross(v[2] - v[0])).norm();
	}

	COMM_FUNC Coord3D Triangle3D::normal() const
	{
		return (v[1] - v[0]).cross(v[2] - v[0]);
	}

	COMM_FUNC bool Triangle3D::computeBarycentrics(const Coord3D& p, Param& bary) const
	{
		if (!isValid())
		{
			bary.u = (Real)0;
			bary.v = (Real)0;
			bary.w = (Real)0;

			return false;
		}

		Coord3D q = Point3D(p).project(Plane3D(v[0], normal())).origin;

		Coord3D d0 = v[1] - v[0];
		Coord3D d1 = v[2] - v[0]; 
		Coord3D d2 = q - v[0];
		Real d00 = d0.dot(d0);
		Real d01 = d0.dot(d1);
		Real d11 = d1.dot(d1);
		Real d20 = d2.dot(d0);
		Real d21 = d2.dot(d1);
		Real denom = d00 * d11 - d01 * d01;
		
		bary.u = (d11 * d20 - d01 * d21) / denom;
		bary.v = (d00 * d21 - d01 * d20) / denom;
		bary.w = 1.0f - bary.v - bary.w;

		return true;
	}

	COMM_FUNC bool Triangle3D::isValid() const
	{
		return abs(area()) > REAL_EPSILON;
	}

	COMM_FUNC Rectangle3D::Rectangle3D()
	{
		center = Coord3D(0);
		axis[0] = Coord3D(1, 0, 0);
		axis[1] = Coord3D(0, 0, 1);
		extent = Coord2D(1, 1);
	}

	COMM_FUNC Rectangle3D::Rectangle3D(const Coord3D& c, const Coord3D& a0, const Coord3D& a1, const Coord2D& ext)
	{
		center = c;
		axis[0] = a0;
		axis[1] = a1;
		extent = ext.maximum(Coord2D(0));
	}

	COMM_FUNC Rectangle3D::Rectangle3D(const Rectangle3D& rectangle)
	{
		center = rectangle.center;
		axis[0] = rectangle.axis[0];
		axis[1] = rectangle.axis[1];
		extent = rectangle.extent;
	}

	COMM_FUNC Point3D Rectangle3D::vertex(const int i) const
	{
		int id = i % 4;
		switch (id)
		{
		case 0:
			return Point3D(center - axis[0] - axis[1]);
			break;
		case 1:
			return Point3D(center + axis[0] - axis[1]);
			break;
		case 2:
			return Point3D(center + axis[0] + axis[1]);
			break;
		default:
			return Point3D(center - axis[0] + axis[1]);
			break;
		}
	}

	COMM_FUNC Segment3D Rectangle3D::edge(const int i) const
	{
		return vertex(i + 1) - vertex(i);
	}

	COMM_FUNC Real Rectangle3D::area() const
	{
		return Real(4) * extent[0] * extent[1];
	}

	COMM_FUNC Coord3D Rectangle3D::normal() const
	{
		return axis[0].cross(axis[1]);
	}

	COMM_FUNC bool Rectangle3D::computeParams(const Coord3D& p, Param& par) const
	{
		if (!isValid())
		{
			par.u = (Real)0;
			par.v = (Real)0;

			return false;
		}

		Coord3D offset = p - center;
		par.u = offset.dot(axis[0]);
		par.v = offset.dot(axis[1]);

		return true;
	}

	COMM_FUNC bool Rectangle3D::isValid() const
	{
		bool bValid = true;
		bValid &= extent[0] >= REAL_EPSILON;
		bValid &= extent[1] >= REAL_EPSILON;

		bValid &= abs(axis[0].normSquared() - Real(1)) < REAL_EPSILON;
		bValid &= abs(axis[1].normSquared() - Real(1)) < REAL_EPSILON;

		bValid &= abs(axis[0].dot(axis[1])) < REAL_EPSILON;

		return bValid;
	}

	COMM_FUNC Disk3D::Disk3D()
	{
		center = Coord3D(0);
		normal = Coord3D(0, 1, 0);
		radius = 1;
	}

	COMM_FUNC Disk3D::Disk3D(const Coord3D& c, const Coord3D& n, const Real& r)
	{
		center = c;
		normal = n.norm() < REAL_EPSILON ? Coord3D(1, 0, 0) : n;
		normal.normalize();
		radius = r < 0 ? 0 : r;
	}

	COMM_FUNC Disk3D::Disk3D(const Disk3D& circle)
	{
		center = circle.center;
		normal = circle.normal;
		radius = circle.radius;
	}

	COMM_FUNC Real Disk3D::area()
	{
		return Real(M_PI) * radius*radius;
	}

	COMM_FUNC bool Disk3D::isValid()
	{
		return radius > REAL_EPSILON && normal.norm() > REAL_EPSILON;
	}

	COMM_FUNC Sphere3D::Sphere3D()
	{
		center = Coord3D(0);
		radius = 1;
	}

	COMM_FUNC Sphere3D::Sphere3D(const Coord3D& c, const Real& r)
	{
		center = c;
		radius = r < 0 ? 0 : r;
	}

	COMM_FUNC Sphere3D::Sphere3D(const Sphere3D& sphere)
	{
		center = sphere.center;
		radius = sphere.radius;
	}

	COMM_FUNC Real Sphere3D::volume()
	{
		return 4 * Real(M_PI) * radius * radius * radius / 3;
	}

	COMM_FUNC bool Sphere3D::isValid()
	{
		return radius >= REAL_EPSILON;
	}

	COMM_FUNC Capsule3D::Capsule3D()
	{
		segment = Segment3D(Coord3D(0), Coord3D(1, 0, 0));
		radius = Real(1);
	}


	COMM_FUNC Capsule3D::Capsule3D(const Coord3D& v0, const Coord3D& v1, const Real& r)
		: segment(v0, v1)
		, radius(r)
	{

	}

	COMM_FUNC Capsule3D::Capsule3D(const Capsule3D& capsule)
	{
		segment = capsule.segment;
		radius = capsule.radius;
	}

	COMM_FUNC Real Capsule3D::volume()
	{
		Real r2 = radius * radius;
		return Real(M_PI)*r2*segment.length() + Real(4.0*M_PI/3.0)*r2*radius;
	}

	COMM_FUNC bool Capsule3D::isValid()
	{
		return radius >= REAL_EPSILON;
	}

	COMM_FUNC Tet3D::Tet3D()
	{
		v[0] = Coord3D(0);
		v[1] = Coord3D(1, 0, 0);
		v[2] = Coord3D(0, 1, 0);
		v[3] = Coord3D(0, 0, 1);
	}

	COMM_FUNC Tet3D::Tet3D(const Coord3D& v0, const Coord3D& v1, const Coord3D& v2, const Coord3D& v3)
	{
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
	}

	COMM_FUNC Tet3D::Tet3D(const Tet3D& tet)
	{
		v[0] = tet.v[0];
		v[1] = tet.v[1];
		v[2] = tet.v[2];
		v[3] = tet.v[3];
	}

	COMM_FUNC Triangle3D Tet3D::face(const int index) const
	{
		switch (index)
		{
		case 0:
			return Triangle3D(v[0], v[2], v[1]);
			break;
		case 1:
			return Triangle3D(v[0], v[3], v[2]);
			break;
		case 2:
			return Triangle3D(v[0], v[1], v[3]);
			break;
		case 3:
			return Triangle3D(v[1], v[2], v[3]);
			break;
		default:
			break;
		}

		//return an ill triangle in case index is out of range
		return Triangle3D(Coord3D(0), Coord3D(0), Coord3D(0));
	}

	COMM_FUNC Real Tet3D::volume() const
	{
		Matrix3D M;
		M.setRow(0, v[1] - v[0]);
		M.setRow(1, v[2] - v[0]);
		M.setRow(2, v[3] - v[0]);

		return M.determinant() / Real(6);
	}

	COMM_FUNC bool Tet3D::isValid()
	{
		return volume() >= REAL_EPSILON;
	}

	AlignedBox3D::AlignedBox3D()
	{
		v0 = Coord3D(Real(-1));
		v1 = Coord3D(Real(1));
	}

	AlignedBox3D::AlignedBox3D(const Coord3D& p0, const Coord3D& p1)
	{
		v0 = p0;
		v1 = p1;
	}

	AlignedBox3D::AlignedBox3D(const AlignedBox3D& box)
	{
		v0 = box.v0;
		v1 = box.v1;
	}

	COMM_FUNC bool AlignedBox3D::intersect(const AlignedBox3D& abox, AlignedBox3D& interBox) const
	{
		for (int i = 0; i < 3; i++)
		{
			if (v1[i] <= abox.v0[i] || v0[i] >= abox.v1[i])
			{
				return false;
			}
		}

		interBox.v0 = v0.maximum(abox.v0);
		interBox.v1 = v1.minimum(abox.v1);

		for (int i = 0; i < 3; i++)
		{
			if (v1[i] <= abox.v1[i])
			{
				interBox.v1[i] = v1[i];
			}
			else
			{
				interBox.v1[i] = abox.v1[i];
			}

			if (v0[i] <= abox.v0[i])
			{
				interBox.v0[i] = abox.v0[i];
			}
			else
			{
				interBox.v0[i] = v0[i];
			}
		}

		return true;
	}

	COMM_FUNC bool AlignedBox3D::isValid()
	{
		return v1[0] > v0[0] && v1[1] > v0[1] && v1[2] > v0[2];
	}

	COMM_FUNC OrientedBox3D::OrientedBox3D()
	{
		center = Coord3D(0);
		u = Coord3D(1, 0, 0);
		v = Coord3D(0, 1, 0);
		w = Coord3D(0, 0, 1);

		extent = Coord3D(1);
	}

	COMM_FUNC OrientedBox3D::OrientedBox3D(const Coord3D c, const Coord3D u_t, const Coord3D v_t, const Coord3D w_t, const Coord3D ext)
	{
		center = c;
		u = u_t;
		v = v_t;
		w = w_t;
		extent = ext;
	}

	COMM_FUNC OrientedBox3D::OrientedBox3D(const OrientedBox3D& obb)
	{
		center = obb.center;
		u = obb.u;
		v = obb.v;
		w = obb.w;
		extent = obb.extent;
	}


	COMM_FUNC Real OrientedBox3D::volume()
	{
		return 8 * extent[0] * extent[1] * extent[2];
	}

	COMM_FUNC bool OrientedBox3D::isValid()
	{
		bool bValid = true;
		bValid &= extent[0] >= REAL_EPSILON;
		bValid &= extent[1] >= REAL_EPSILON;
		bValid &= extent[2] >= REAL_EPSILON;

		bValid &= abs(u.normSquared() - Real(1)) < REAL_EPSILON;
		bValid &= abs(v.normSquared() - Real(1)) < REAL_EPSILON;
		bValid &= abs(w.normSquared() - Real(1)) < REAL_EPSILON;

		bValid &= abs(u.dot(v)) < REAL_EPSILON;
		bValid &= abs(v.dot(w)) < REAL_EPSILON;
		bValid &= abs(w.dot(u)) < REAL_EPSILON;

		return bValid;
	}

}