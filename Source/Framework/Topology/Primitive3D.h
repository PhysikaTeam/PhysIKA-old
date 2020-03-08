/**
 * @file Primitive3D.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief 
 * @version 0.1
 * @date 2020-02-24
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#include "Core/Vector.h"
#include "Core/Matrix.h"

namespace PhysIKA
{
// #ifdef PRECISION_FLOAT
// 	#define REAL_EPSILON 1e-5
// 	#define  REAL_EPSILON_SQUARED 1e-10
// #else
// 	#define REAL_EPSILON 1e-10
// 	#define  REAL_EPSILON_SQUARED 1e-20
// #endif

#ifdef PRECISION_FLOAT
	typedef Vector2f Coord2D;
	typedef Vector3f Coord3D;
	typedef Matrix3f Matrix3D;
#else
	typedef Vector2d Coord2D;
	typedef Vector3d Coord3D;
	typedef Matrix3d Matrix3D;
#endif

	constexpr Real REAL_MAX = (std::numeric_limits<Real>::max)();
	constexpr Real REAL_MIN = (std::numeric_limits<Real>::min)();
	constexpr Real REAL_EPSILON = (std::numeric_limits<Real>::epsilon)();
	constexpr Real REAL_EPSILON_SQUARED = REAL_EPSILON * REAL_EPSILON;

	/**
	 * @brief 0D geometric primitive in three-dimensional space
	 *
	 */
	class Point3D;

	/**
	 * @brief 1D geometric primitives in three-dimensional space
	 * 
	 */
	class Line3D;
	class Ray3D;
	class Segment3D;

	/**
	 * @brief 2D geometric primitives in three-dimensional space
	 * 
	 */
	class Plane3D;
	class Triangle3D;
	class Rectangle3D;
	class Disk3D;

	/**
	 * @brief 3D geometric primitives in three-dimensional space
	 * 
	 */
	class Sphere3D;
	class Capsule3D;
	class Tet3D;
	class AlignedBox3D;
	class OrientedBox3D;

	class Bool
	{
	public:
		Bool(bool v = false) { val = v; }

		COMM_FUNC bool operator! () { return !val; }
		COMM_FUNC bool operator== (bool v) { return val == v; }
		
		COMM_FUNC bool operator= (bool v) { return val = v; }

		COMM_FUNC Bool& operator&= (bool v) { val &= v;  return *this; }
		COMM_FUNC Bool& operator|= (bool v) { val |= v;  return *this; }

	private:
		bool val = false;
	};

	class Point3D
	{
	public:
		COMM_FUNC Point3D();
		COMM_FUNC Point3D(const Real& c0, const Real& c1, const Real& c2);
		COMM_FUNC Point3D(const Point3D& pt);

		COMM_FUNC Point3D operator = (const Coord3D& p);

		explicit COMM_FUNC Point3D(const Real& val);
		explicit COMM_FUNC Point3D(const Coord3D& pos);

		
        /**
         * @brief project a point onto linear components -- lines, rays and segments
         * 
         * @param line/ray/segment linear components 
         * @return closest point
         */
		COMM_FUNC Point3D project(const Line3D& line) const;
		COMM_FUNC Point3D project(const Ray3D& ray) const;
		COMM_FUNC Point3D project(const Segment3D& segment) const;
        /**
         * @brief project a point onto planar components -- planes, triangles and disks
         * 
         * @param plane/triangle/disk planar components
         * @return closest point
         */
		COMM_FUNC Point3D project(const Plane3D& plane) const;
		COMM_FUNC Point3D project(const Triangle3D& triangle) const;
		COMM_FUNC Point3D project(const Rectangle3D& rectangle) const;
		COMM_FUNC Point3D project(const Disk3D& disk) const;
        /**
         * @brief project a point onto polyhedra components -- tetrahedra, spheres and oriented bounding boxes
         * 
         * @param sphere/capsule/tet/abox/obb polyhedra components
         * @return closest point
         */
		COMM_FUNC Point3D project(const Sphere3D& sphere, Bool& bInside = Bool(false)) const;
		COMM_FUNC Point3D project(const Capsule3D& capsule, Bool& bInside = Bool(false)) const;
		COMM_FUNC Point3D project(const Tet3D& tet, Bool& bInside = Bool(false)) const;
		COMM_FUNC Point3D project(const AlignedBox3D& abox, Bool& bInside = Bool(false)) const;
		COMM_FUNC Point3D project(const OrientedBox3D& obb, Bool& bInside = Bool(false)) const;



		COMM_FUNC Real distance(const Point3D& pt) const;
		COMM_FUNC Real distance(const Line3D& line) const;
		COMM_FUNC Real distance(const Ray3D& ray) const;
		COMM_FUNC Real distance(const Segment3D& segment) const;
		/**
		 * @brief compute the signed distance to 2D geometric primitives
		 * 
		 * @param plane/triangle/rectangle/disk planar components
		 * @return positive if point resides in the positive side of 2D geometric primitives
		 */
		COMM_FUNC Real distance(const Plane3D& plane) const;
		COMM_FUNC Real distance(const Triangle3D& triangle) const;
		COMM_FUNC Real distance(const Rectangle3D& rectangle) const;
		COMM_FUNC Real distance(const Disk3D& disk) const;
		/**
		 * @brief compute signed distance to 3D geometric primitives
		 * 
		 * @param sphere/tet/abox/obb 3D geometric primitives 
		 * @return Real negative distance if a point is inside the 3D geometric primitive, otherwise return a positive value
		 */
		COMM_FUNC Real distance(const Sphere3D& sphere) const;
		COMM_FUNC Real distance(const Capsule3D& capsule) const;
		COMM_FUNC Real distance(const Tet3D& tet) const;
		COMM_FUNC Real distance(const AlignedBox3D& abox) const;
		COMM_FUNC Real distance(const OrientedBox3D& obb) const;



		COMM_FUNC Real distanceSquared(const Point3D& pt) const;
		COMM_FUNC Real distanceSquared(const Line3D& line) const;
		COMM_FUNC Real distanceSquared(const Ray3D& ray) const;
		COMM_FUNC Real distanceSquared(const Segment3D& segment) const;
		/**
		 * @brief return squared distance from a point to 3D geometric primitives
		 * 
		 * @param plane/triangle/rectangle/disk planar components
		 * @return COMM_FUNC distanceSquared 
		 */
		COMM_FUNC Real distanceSquared(const Plane3D& plane) const;
		COMM_FUNC Real distanceSquared(const Triangle3D& triangle) const;
		COMM_FUNC Real distanceSquared(const Rectangle3D& rectangle) const;
		COMM_FUNC Real distanceSquared(const Disk3D& disk) const;
		/**
		 * @brief return squared distance from a point to 3D geometric primitives
		 * 
		 * @param sphere/capsule/tet/abox/obb 3D geometric primitives
		 * @return Real squared distance
		 */
		COMM_FUNC Real distanceSquared(const Sphere3D& sphere) const;
		COMM_FUNC Real distanceSquared(const Capsule3D& capsule) const;
		COMM_FUNC Real distanceSquared(const Tet3D& tet) const;
		COMM_FUNC Real distanceSquared(const AlignedBox3D& abox) const;
		COMM_FUNC Real distanceSquared(const OrientedBox3D& obb) const;


		/**
		 * @brief check whether a point strictly lies inside (excluding boundary) a 1D geometric primitive
		 * 
		 * @param line/ray/segment 1D geometric primitives
		 * @return true if a point is inside the geometric primitive, otherwise return false
		 */
		COMM_FUNC bool inside(const Line3D& line) const;
		COMM_FUNC bool inside(const Ray3D& ray) const;
		COMM_FUNC bool inside(const Segment3D& segment) const;
		/**
		 * @brief check whether a point strictly lies inside (excluding boundary) a 2D geometric primitive
		 * 
		 * @param plane/triangle/rectangle/disk 2D geometric primitives
		 * @return true if a point is inside the geometric primitive, otherwise return false
		 */
		COMM_FUNC bool inside(const Plane3D& plane) const;
		COMM_FUNC bool inside(const Triangle3D& triangle) const;
		COMM_FUNC bool inside(const Rectangle3D& rectangle) const;
		COMM_FUNC bool inside(const Disk3D& disk) const;
		/**
		 * @brief check whether a point strictly lies inside (excluding boundary) a 3D geometric primitive
		 * 
		 * @param sphere/tet/abox/obb 3D geometric primitives
		 * @return true if a point is inside the geometric primitive, otherwise return false
		 */
		COMM_FUNC bool inside(const Sphere3D& sphere) const;
		COMM_FUNC bool inside(const Capsule3D& capsule) const;
		COMM_FUNC bool inside(const Tet3D& tet) const;
		COMM_FUNC bool inside(const AlignedBox3D& box) const;
		COMM_FUNC bool inside(const OrientedBox3D& obb) const;

		COMM_FUNC const Segment3D operator-(const Point3D& pt) const;

		Coord3D origin;
	};

	class Line3D
	{
	public:
		COMM_FUNC Line3D();
        /**
         * @brief 
         * 
         * @param pos 
         * @param dir = 0 indicate the line degenerates into a point
         */
		COMM_FUNC Line3D(const Coord3D& pos, const Coord3D& dir);
		COMM_FUNC Line3D(const Line3D& line);

		COMM_FUNC Segment3D proximity(const Line3D& line) const;
		COMM_FUNC Segment3D proximity(const Ray3D& ray) const;
		COMM_FUNC Segment3D proximity(const Segment3D& segment) const;

		COMM_FUNC Segment3D proximity(const Triangle3D& triangle) const;
		COMM_FUNC Segment3D proximity(const Rectangle3D& rectangle) const;

		COMM_FUNC Segment3D proximity(const Sphere3D& sphere) const;
		COMM_FUNC Segment3D proximity(const AlignedBox3D& box) const;
		COMM_FUNC Segment3D proximity(const OrientedBox3D& obb) const;


		COMM_FUNC Real distance(const Point3D& pt) const;
		COMM_FUNC Real distance(const Line3D& line) const;
		COMM_FUNC Real distance(const Ray3D& ray) const;
		COMM_FUNC Real distance(const Segment3D& segment) const;
		
		COMM_FUNC Real distance(const AlignedBox3D& box) const;
		COMM_FUNC Real distance(const OrientedBox3D& obb) const;


		COMM_FUNC Real distanceSquared(const Point3D& pt) const;
		COMM_FUNC Real distanceSquared(const Line3D& line) const;
		COMM_FUNC Real distanceSquared(const Ray3D& ray) const;
		COMM_FUNC Real distanceSquared(const Segment3D& segment) const;
		/**
		 * @brief compute signed distance to 3D geometric primitives
		 * 
		 * @param box/obb 
		 * @return 0 if intersecting the 3D geometric primitives
		 */
		COMM_FUNC Real distanceSquared(const AlignedBox3D& box) const;
		COMM_FUNC Real distanceSquared(const OrientedBox3D& obb) const;
		

		COMM_FUNC int intersect(const Plane3D& plane, Point3D& interPt) const;
		COMM_FUNC int intersect(const Triangle3D& triangle, Point3D& interPt) const;

		COMM_FUNC int intersect(const Sphere3D& sphere, Segment3D& interSeg) const;
		COMM_FUNC int intersect(const Tet3D& tet, Segment3D& interSeg) const;
		COMM_FUNC int intersect(const AlignedBox3D& abox, Segment3D& interSeg) const;


		COMM_FUNC Real parameter(const Coord3D& pos) const;

		COMM_FUNC bool isValid() const;

		Coord3D origin;

		//direction will be normalized during construction
		Coord3D direction;
	};

	class Ray3D
	{
	public:
		COMM_FUNC Ray3D();

		struct Param
		{
			Real t;
		};

        /**
         * @brief 
         * 
         * @param pos 
         * @param ||dir|| = 0 indicates the ray degenerates into a point
         * @return COMM_FUNC 
         */
		COMM_FUNC Ray3D(const Coord3D& pos, const Coord3D& dir);
		COMM_FUNC Ray3D(const Ray3D& ray);

		COMM_FUNC Segment3D proximity(const Ray3D& ray) const;
		COMM_FUNC Segment3D proximity(const Segment3D& segment) const;

		COMM_FUNC Segment3D proximity(const Triangle3D& triangle) const;
		COMM_FUNC Segment3D proximity(const Rectangle3D& rectangle) const;

		COMM_FUNC Segment3D proximity(const AlignedBox3D& box) const;
		COMM_FUNC Segment3D proximity(const OrientedBox3D& obb) const;

		COMM_FUNC Real distance(const Point3D& pt) const;
		COMM_FUNC Real distance(const Segment3D& segment) const;
		COMM_FUNC Real distance(const Triangle3D& triangle) const;

		COMM_FUNC Real distanceSquared(const Point3D& pt) const;
		COMM_FUNC Real distanceSquared(const Segment3D& segment) const;
		COMM_FUNC Real distanceSquared(const Triangle3D& triangle) const;

		COMM_FUNC int intersect(const Plane3D& plane, Point3D& interPt) const;
		COMM_FUNC int intersect(const Triangle3D& triangle, Point3D& interPt) const;

		COMM_FUNC int intersect(const Sphere3D& sphere, Segment3D& interSeg) const;

		COMM_FUNC int intersect(const AlignedBox3D& abox, Segment3D& interSeg) const;

		COMM_FUNC Real parameter(const Coord3D& pos) const;

		COMM_FUNC bool isValid() const;

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

		COMM_FUNC Segment3D proximity(const Segment3D& segment) const;

		COMM_FUNC Segment3D proximity(const Plane3D& plane) const;
		COMM_FUNC Segment3D proximity(const Triangle3D& triangle) const;
		COMM_FUNC Segment3D proximity(const Rectangle3D& rectangle) const;

		COMM_FUNC Segment3D proximity(const AlignedBox3D& box) const;
		COMM_FUNC Segment3D proximity(const OrientedBox3D& obb) const;


		COMM_FUNC Real distance(const Segment3D& segment) const;

		COMM_FUNC Real distance(const Triangle3D& triangle) const;

		COMM_FUNC Real distanceSquared(const Segment3D& segment) const;

		COMM_FUNC Real distanceSquared(const Triangle3D& triangle) const;

		COMM_FUNC bool intersect(const Plane3D& plane, Point3D& interPt) const;
		COMM_FUNC bool intersect(const Triangle3D& triangle, Point3D& interPt) const;

		COMM_FUNC int intersect(const Sphere3D& sphere, Segment3D& interSeg) const;
		COMM_FUNC int intersect(const AlignedBox3D& abox, Segment3D& interSeg) const;

		COMM_FUNC Real length() const;
		COMM_FUNC Real lengthSquared() const;

		COMM_FUNC Real parameter(const Coord3D& pos) const;

		inline COMM_FUNC Coord3D& startPoint() { return v0; }
		inline COMM_FUNC Coord3D& endPoint() { return v1; }

		inline COMM_FUNC Coord3D startPoint() const { return v0; }
		inline COMM_FUNC Coord3D endPoint() const { return v1; }

		inline COMM_FUNC Coord3D direction() const { return v1 - v0; }

		inline COMM_FUNC Segment3D operator-(void) const;

		COMM_FUNC bool isValid() const;
		
		Coord3D v0;
		Coord3D v1;
	};

	class Plane3D
	{
	public:
		COMM_FUNC Plane3D();
		COMM_FUNC Plane3D(const Coord3D& pos, const Coord3D n);
		COMM_FUNC Plane3D(const Plane3D& plane);

		COMM_FUNC bool isValid() const;

		Coord3D origin;

		/**
		 * @brief the plane will be treated as a single point if its normal is zero
		 */
		Coord3D normal;
	};

	class Triangle3D
	{
	public:
		COMM_FUNC Triangle3D();
		COMM_FUNC Triangle3D(const Coord3D& p0, const Coord3D& p1, const Coord3D& p2);
		COMM_FUNC Triangle3D(const Triangle3D& triangle);

		struct Param
		{
			Real u;
			Real v;
			Real w;
		};

		COMM_FUNC Real area() const;
		COMM_FUNC Coord3D normal() const;

		COMM_FUNC bool computeBarycentrics(const Coord3D& p, Param& bary) const;

		COMM_FUNC bool isValid() const;

		Coord3D v[3];
	};

	class Rectangle3D
	{
	public:
		COMM_FUNC Rectangle3D();
		COMM_FUNC Rectangle3D(const Coord3D& c, const Coord3D& a0, const Coord3D& a1, const Coord2D& ext);
		COMM_FUNC Rectangle3D(const Rectangle3D& rectangle);

		struct Param
		{
			Real u;
			Real v;
		};

		COMM_FUNC Point3D vertex(const int i) const;
		COMM_FUNC Segment3D edge(const int i) const;

		COMM_FUNC Real area() const;
		COMM_FUNC Coord3D normal() const;

		COMM_FUNC bool computeParams(const Coord3D& p, Param& par) const;

		COMM_FUNC bool isValid() const;

		Coord3D center;
		/**
		 * @brief two orthonormal unit axis
		 * 
		 */
		Coord3D axis[2];
		Coord2D extent;
	};

	class Disk3D
	{
	public:
		COMM_FUNC Disk3D();
		COMM_FUNC Disk3D(const Coord3D& c, const Coord3D& n, const Real& r);
		COMM_FUNC Disk3D(const Disk3D& circle);

		COMM_FUNC Real area();

		COMM_FUNC bool isValid();

		Real radius;
		Coord3D center;
		Coord3D normal;
	};

	class Sphere3D
	{
	public:
		COMM_FUNC Sphere3D();
		COMM_FUNC Sphere3D(const Coord3D& c, const Real& r);
		COMM_FUNC Sphere3D(const Sphere3D& sphere);

		COMM_FUNC Real volume();

		COMM_FUNC bool isValid();

		Real radius;
		Coord3D center;
	};

	class Capsule3D
	{
	public:
		COMM_FUNC Capsule3D();
		COMM_FUNC Capsule3D(const Coord3D& v0, const Coord3D& v1, const Real& r);
		COMM_FUNC Capsule3D(const Capsule3D& capsule);

		COMM_FUNC Real volume();

		COMM_FUNC bool isValid();

		Real radius;
		Segment3D segment;
	};

    /**
     * @brief vertices are ordered so that the normal vectors for the triangular faces point outwards
     * 
     */
	class Tet3D
	{
	public:
		COMM_FUNC Tet3D();
		COMM_FUNC Tet3D(const Coord3D& v0, const Coord3D& v1, const Coord3D& v2, const Coord3D& v3);
		COMM_FUNC Tet3D(const Tet3D& tet);

		COMM_FUNC Triangle3D face(const int index) const;

		COMM_FUNC Real volume() const;

		COMM_FUNC bool isValid();

		Coord3D v[4];
	};

	class AlignedBox3D
	{
	public:
		AlignedBox3D();
		AlignedBox3D(const Coord3D& p0, const Coord3D& p1);
		AlignedBox3D(const AlignedBox3D& box);

		COMM_FUNC bool intersect(const AlignedBox3D& abox, AlignedBox3D& interBox) const;

		COMM_FUNC bool isValid();

		Coord3D v0;
		Coord3D v1;
	};

	class OrientedBox3D
	{
	public:
		COMM_FUNC OrientedBox3D();

        /**
         * @brief construct an oriented bounding box, gurantee u_t, v_t and w_t are unit vectors and form right-handed orthornormal basis
         * 
         * @param c  centerpoint 
         * @param u_t 
         * @param v_t 
         * @param w_t 
         * @param ext half the dimension in each of the u, v, and w directions
         * @return COMM_FUNC 
         */
		COMM_FUNC OrientedBox3D(const Coord3D c, const Coord3D u_t, const Coord3D v_t, const Coord3D w_t, const Coord3D ext);

		COMM_FUNC OrientedBox3D(const OrientedBox3D& obb);

		COMM_FUNC Real volume();

		COMM_FUNC bool isValid();

        /**
         * @brief centerpoint
         * 
         */
		Coord3D center;

        /**
         * @brief three unit vectors u, v and w forming a right-handed orthornormal basis
         * 
         */
		Coord3D u, v, w;

        /**
         * @brief half the dimension in each of the u, v, and w directions
         */
		Coord3D extent;
	};
}

