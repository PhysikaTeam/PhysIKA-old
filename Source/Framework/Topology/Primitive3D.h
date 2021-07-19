#ifndef PHYSIKA_PRIMITIVE_3D
#define PHYSIKA_PRIMITIVE_3D
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

namespace PhysIKA {
// #ifdef PRECISION_FLOAT
//     #define REAL_EPSILON 1e-5
//     #define  REAL_EPSILON_SQUARED 1e-10
// #else
//     #define REAL_EPSILON 1e-10
//     #define  REAL_EPSILON_SQUARED 1e-20
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

constexpr Real REAL_MAX             = (std::numeric_limits<Real>::max)();
constexpr Real REAL_MIN             = (std::numeric_limits<Real>::min)();
constexpr Real REAL_EPSILON         = (std::numeric_limits<Real>::epsilon)();
constexpr Real REAL_EPSILON_SQUARED = REAL_EPSILON * REAL_EPSILON;

#ifdef PRECISION_FLOAT
typedef Vector2f Coord2D;
typedef Vector3f Coord3D;
typedef Matrix3f Matrix3D;
#else
typedef Vector2d Coord2D;
typedef Vector3d Coord3D;
typedef Matrix3d Matrix3D;
#endif

/**
     * @brief 0D geometric primitive in three-dimensional space
     *
     */
template <typename Real>
class TPoint3D;

/**
     * @brief 1D geometric primitives in three-dimensional space
     *
     */
template <typename Real>
class TLine3D;
template <typename Real>
class TRay3D;
template <typename Real>
class TSegment3D;

/**
     * @brief 2D geometric primitives in three-dimensional space
     *
     */
template <typename Real>
class TPlane3D;
template <typename Real>
class TTriangle3D;
template <typename Real>
class TRectangle3D;
template <typename Real>
class TDisk3D;

/**
     * @brief 3D geometric primitives in three-dimensional space
     *
     */
template <typename Real>
class TSphere3D;
template <typename Real>
class TCapsule3D;
template <typename Real>
class TTet3D;
template <typename Real>
class TAlignedBox3D;
template <typename Real>
class TOrientedBox3D;

class Bool
{
public:
    COMM_FUNC Bool(bool v = false)
    {
        val = v;
    }

    COMM_FUNC bool operator!()
    {
        return !val;
    }
    COMM_FUNC bool operator==(bool v)
    {
        return val == v;
    }

    COMM_FUNC bool operator=(bool v)
    {
        return val = v;
    }

    COMM_FUNC Bool& operator&=(bool v)
    {
        val &= v;
        return *this;
    }
    COMM_FUNC Bool& operator|=(bool v)
    {
        val |= v;
        return *this;
    }

private:
    bool val = false;
};

template <typename Real>
class TPoint3D
{
public:
    COMM_FUNC TPoint3D();
    COMM_FUNC TPoint3D(const Real& c0, const Real& c1, const Real& c2);
    COMM_FUNC TPoint3D(const TPoint3D& pt);

    COMM_FUNC TPoint3D operator=(const Coord3D& p);

    explicit COMM_FUNC TPoint3D(const Real& val);
    explicit COMM_FUNC TPoint3D(const Coord3D& pos);

    /**
         * @brief project a point onto linear components -- lines, rays and segments
         *
         * @param line/ray/segment linear components
         * @return closest point
         */
    COMM_FUNC TPoint3D<Real> project(const TLine3D<Real>& line) const;
    COMM_FUNC TPoint3D<Real> project(const TRay3D<Real>& ray) const;
    COMM_FUNC TPoint3D<Real> project(const TSegment3D<Real>& segment) const;
    /**
         * @brief project a point onto planar components -- planes, triangles and disks
         *
         * @param plane/triangle/disk planar components
         * @return closest point
         */
    COMM_FUNC TPoint3D<Real> project(const TPlane3D<Real>& plane) const;
    COMM_FUNC TPoint3D<Real> project(const TTriangle3D<Real>& triangle) const;
    COMM_FUNC TPoint3D<Real> project(const TRectangle3D<Real>& rectangle) const;
    COMM_FUNC TPoint3D<Real> project(const TDisk3D<Real>& disk) const;
    /**
         * @brief project a point onto polyhedra components -- tetrahedra, spheres and oriented bounding boxes
         *
         * @param sphere/capsule/tet/abox/obb polyhedra components
         * @return closest point
         */
    COMM_FUNC TPoint3D<Real> project(const TSphere3D<Real>& sphere, Bool& bInside = Bool(false)) const;
    COMM_FUNC TPoint3D<Real> project(const TCapsule3D<Real>& capsule, Bool& bInside = Bool(false)) const;
    COMM_FUNC TPoint3D<Real> project(const TTet3D<Real>& tet, Bool& bInside = Bool(false)) const;
    COMM_FUNC TPoint3D<Real> project(const TAlignedBox3D<Real>& abox, Bool& bInside = Bool(false)) const;
    COMM_FUNC TPoint3D<Real> project(const TOrientedBox3D<Real>& obb, Bool& bInside = Bool(false)) const;

    COMM_FUNC Real distance(const TPoint3D<Real>& pt) const;
    COMM_FUNC Real distance(const TLine3D<Real>& line) const;
    COMM_FUNC Real distance(const TRay3D<Real>& ray) const;
    COMM_FUNC Real distance(const TSegment3D<Real>& segment) const;
    /**
         * @brief compute the signed distance to 2D geometric primitives
         *
         * @param plane/triangle/rectangle/disk planar components
         * @return positive if point resides in the positive side of 2D geometric primitives
         */
    COMM_FUNC Real distance(const TPlane3D<Real>& plane) const;
    COMM_FUNC Real distance(const TTriangle3D<Real>& triangle) const;
    COMM_FUNC Real distance(const TRectangle3D<Real>& rectangle) const;
    COMM_FUNC Real distance(const TDisk3D<Real>& disk) const;

    COMM_FUNC Real areaTriangle(const TTriangle3D<Real>& triangle, const Real& r) const;
    COMM_FUNC Real areaTrianglePrint(const TTriangle3D<Real>& triangle, const Real& r) const;

    /**
         * @brief compute signed distance to 3D geometric primitives
         *
         * @param sphere/tet/abox/obb 3D geometric primitives
         * @return Real negative distance if a point is inside the 3D geometric primitive, otherwise return a positive value
         */
    COMM_FUNC Real distance(const TSphere3D<Real>& sphere) const;
    COMM_FUNC Real distance(const TCapsule3D<Real>& capsule) const;
    COMM_FUNC Real distance(const TTet3D<Real>& tet) const;
    COMM_FUNC Real distance(const TAlignedBox3D<Real>& abox) const;
    COMM_FUNC Real distance(const TOrientedBox3D<Real>& obb) const;

    COMM_FUNC Real distanceSquared(const TPoint3D<Real>& pt) const;
    COMM_FUNC Real distanceSquared(const TLine3D<Real>& line) const;
    COMM_FUNC Real distanceSquared(const TRay3D<Real>& ray) const;
    COMM_FUNC Real distanceSquared(const TSegment3D<Real>& segment) const;
    /**
         * @brief return squared distance from a point to 3D geometric primitives
         *
         * @param plane/triangle/rectangle/disk planar components
         * @return COMM_FUNC distanceSquared
         */
    COMM_FUNC Real distanceSquared(const TPlane3D<Real>& plane) const;
    COMM_FUNC Real distanceSquared(const TTriangle3D<Real>& triangle) const;
    COMM_FUNC Real distanceSquared(const TRectangle3D<Real>& rectangle) const;
    COMM_FUNC Real distanceSquared(const TDisk3D<Real>& disk) const;
    /**
         * @brief return squared distance from a point to 3D geometric primitives
         *
         * @param sphere/capsule/tet/abox/obb 3D geometric primitives
         * @return Real squared distance
         */
    COMM_FUNC Real distanceSquared(const TSphere3D<Real>& sphere) const;
    COMM_FUNC Real distanceSquared(const TCapsule3D<Real>& capsule) const;
    COMM_FUNC Real distanceSquared(const TTet3D<Real>& tet) const;
    COMM_FUNC Real distanceSquared(const TAlignedBox3D<Real>& abox) const;
    COMM_FUNC Real distanceSquared(const TOrientedBox3D<Real>& obb) const;

    /**
         * @brief check whether a point strictly lies inside (excluding boundary) a 1D geometric primitive
         *
         * @param line/ray/segment 1D geometric primitives
         * @return true if a point is inside the geometric primitive, otherwise return false
         */
    COMM_FUNC bool inside(const TLine3D<Real>& line) const;
    COMM_FUNC bool inside(const TRay3D<Real>& ray) const;
    COMM_FUNC bool inside(const TSegment3D<Real>& segment) const;
    /**
         * @brief check whether a point strictly lies inside (excluding boundary) a 2D geometric primitive
         *
         * @param plane/triangle/rectangle/disk 2D geometric primitives
         * @return true if a point is inside the geometric primitive, otherwise return false
         */
    COMM_FUNC bool inside(const TPlane3D<Real>& plane) const;
    COMM_FUNC bool inside(const TTriangle3D<Real>& triangle) const;
    COMM_FUNC bool inside(const TRectangle3D<Real>& rectangle) const;
    COMM_FUNC bool inside(const TDisk3D<Real>& disk) const;
    /**
         * @brief check whether a point strictly lies inside (excluding boundary) a 3D geometric primitive
         *
         * @param sphere/tet/abox/obb 3D geometric primitives
         * @return true if a point is inside the geometric primitive, otherwise return false
         */
    COMM_FUNC bool inside(const TSphere3D<Real>& sphere) const;
    COMM_FUNC bool inside(const TCapsule3D<Real>& capsule) const;
    COMM_FUNC bool inside(const TTet3D<Real>& tet) const;
    COMM_FUNC bool inside(const TAlignedBox3D<Real>& box) const;
    COMM_FUNC bool inside(const TOrientedBox3D<Real>& obb) const;

    COMM_FUNC const TSegment3D<Real> operator-(const TPoint3D<Real>& pt) const;

    Coord3D origin;
};

template <typename Real>
class TLine3D
{
public:
    COMM_FUNC TLine3D();
    /**
         * @brief
         *
         * @param pos
         * @param dir = 0 indicate the line degenerates into a point
         */
    COMM_FUNC TLine3D(const Coord3D& pos, const Coord3D& dir);
    COMM_FUNC TLine3D(const TLine3D<Real>& line);

    COMM_FUNC TSegment3D<Real> proximity(const TLine3D<Real>& line) const;
    COMM_FUNC TSegment3D<Real> proximity(const TRay3D<Real>& ray) const;
    COMM_FUNC TSegment3D<Real> proximity(const TSegment3D<Real>& segment) const;

    COMM_FUNC TSegment3D<Real> proximity(const TTriangle3D<Real>& triangle) const;
    COMM_FUNC TSegment3D<Real> proximity(const TRectangle3D<Real>& rectangle) const;

    COMM_FUNC TSegment3D<Real> proximity(const TSphere3D<Real>& sphere) const;
    COMM_FUNC TSegment3D<Real> proximity(const TAlignedBox3D<Real>& box) const;
    COMM_FUNC TSegment3D<Real> proximity(const TOrientedBox3D<Real>& obb) const;

    COMM_FUNC Real distance(const TPoint3D<Real>& pt) const;
    COMM_FUNC Real distance(const TLine3D<Real>& line) const;
    COMM_FUNC Real distance(const TRay3D<Real>& ray) const;
    COMM_FUNC Real distance(const TSegment3D<Real>& segment) const;

    COMM_FUNC Real distance(const TAlignedBox3D<Real>& box) const;
    COMM_FUNC Real distance(const TOrientedBox3D<Real>& obb) const;

    //         COMM_FUNC Line3D(const Coord3D& pos, const Coord3D& dir);
    //         COMM_FUNC Line3D(const Line3D& line);

    COMM_FUNC Real distanceSquared(const TPoint3D<Real>& pt) const;
    COMM_FUNC Real distanceSquared(const TLine3D<Real>& line) const;
    COMM_FUNC Real distanceSquared(const TRay3D<Real>& ray) const;
    COMM_FUNC Real distanceSquared(const TSegment3D<Real>& segment) const;
    /**
         * @brief compute signed distance to 3D geometric primitives
         *
         * @param box/obb
         * @return 0 if intersecting the 3D geometric primitives
         */
    COMM_FUNC Real distanceSquared(const TAlignedBox3D<Real>& box) const;
    COMM_FUNC Real distanceSquared(const TOrientedBox3D<Real>& obb) const;

    COMM_FUNC int intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const;
    COMM_FUNC int intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const;

    COMM_FUNC int intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const;
    COMM_FUNC int intersect(const TTet3D<Real>& tet, TSegment3D<Real>& interSeg) const;
    COMM_FUNC int intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const;

    COMM_FUNC Real parameter(const Coord3D& pos) const;

    COMM_FUNC bool isValid() const;

    Coord3D origin;

    //direction will be normalized during construction
    Coord3D direction;
};

template <typename Real>
class TRay3D
{
public:
    COMM_FUNC TRay3D();

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
    COMM_FUNC TRay3D(const Coord3D& pos, const Coord3D& dir);
    COMM_FUNC TRay3D(const TRay3D<Real>& ray);

    COMM_FUNC TSegment3D<Real> proximity(const TRay3D<Real>& ray) const;
    COMM_FUNC TSegment3D<Real> proximity(const TSegment3D<Real>& segment) const;

    COMM_FUNC TSegment3D<Real> proximity(const TTriangle3D<Real>& triangle) const;
    COMM_FUNC TSegment3D<Real> proximity(const TRectangle3D<Real>& rectangle) const;

    COMM_FUNC TSegment3D<Real> proximity(const TAlignedBox3D<Real>& box) const;
    COMM_FUNC TSegment3D<Real> proximity(const TOrientedBox3D<Real>& obb) const;

    COMM_FUNC Real distance(const TPoint3D<Real>& pt) const;
    COMM_FUNC Real distance(const TSegment3D<Real>& segment) const;
    COMM_FUNC Real distance(const TTriangle3D<Real>& triangle) const;

    COMM_FUNC Real distanceSquared(const TPoint3D<Real>& pt) const;
    COMM_FUNC Real distanceSquared(const TSegment3D<Real>& segment) const;
    COMM_FUNC Real distanceSquared(const TTriangle3D<Real>& triangle) const;

    COMM_FUNC int intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const;
    COMM_FUNC int intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const;

    COMM_FUNC int intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const;

    COMM_FUNC int intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const;

    COMM_FUNC Real parameter(const Coord3D& pos) const;

    COMM_FUNC bool isValid() const;

    Coord3D origin;

    //guarantee direction is a unit vector
    Coord3D direction;
};

template <typename Real>
class TSegment3D
{
public:
    COMM_FUNC TSegment3D();
    COMM_FUNC TSegment3D(const Coord3D& p0, const Coord3D& p1);
    COMM_FUNC TSegment3D(const TSegment3D<Real>& segment);

    COMM_FUNC TSegment3D<Real> proximity(const TSegment3D<Real>& segment) const;

    COMM_FUNC TSegment3D<Real> proximity(const TPlane3D<Real>& plane) const;
    COMM_FUNC TSegment3D<Real> proximity(const TTriangle3D<Real>& triangle) const;
    COMM_FUNC TSegment3D<Real> proximity(const TRectangle3D<Real>& rectangle) const;

    COMM_FUNC TSegment3D<Real> proximity(const TAlignedBox3D<Real>& box) const;
    COMM_FUNC TSegment3D<Real> proximity(const TOrientedBox3D<Real>& obb) const;

    COMM_FUNC Real distance(const TSegment3D<Real>& segment) const;

    COMM_FUNC Real distance(const TTriangle3D<Real>& triangle) const;

    COMM_FUNC Real distanceSquared(const TSegment3D<Real>& segment) const;

    COMM_FUNC Real distanceSquared(const TTriangle3D<Real>& triangle) const;

    COMM_FUNC bool intersect(const TPlane3D<Real>& plane, TPoint3D<Real>& interPt) const;
    COMM_FUNC bool intersect(const TTriangle3D<Real>& triangle, TPoint3D<Real>& interPt) const;

    COMM_FUNC int intersect(const TSphere3D<Real>& sphere, TSegment3D<Real>& interSeg) const;
    COMM_FUNC int intersect(const TAlignedBox3D<Real>& abox, TSegment3D<Real>& interSeg) const;
    COMM_FUNC int intersect(const TOrientedBox3D<Real>& obb, TSegment3D<Real>& interSeg) const;

    COMM_FUNC Real length() const;
    COMM_FUNC Real lengthSquared() const;

    COMM_FUNC Real parameter(const Coord3D& pos) const;

    inline COMM_FUNC Coord3D& startPoint()
    {
        return v0;
    }
    inline COMM_FUNC Coord3D& endPoint()
    {
        return v1;
    }

    inline COMM_FUNC Coord3D startPoint() const
    {
        return v0;
    }
    inline COMM_FUNC Coord3D endPoint() const
    {
        return v1;
    }

    inline COMM_FUNC Coord3D direction() const
    {
        return v1 - v0;
    }

    inline COMM_FUNC TSegment3D<Real> operator-(void) const;

    COMM_FUNC bool isValid() const;

    Coord3D v0;
    Coord3D v1;
};

template <typename Real>
class TPlane3D
{
public:
    COMM_FUNC TPlane3D();
    COMM_FUNC TPlane3D(const Coord3D& pos, const Coord3D n);
    COMM_FUNC TPlane3D(const TPlane3D& plane);

    COMM_FUNC bool isValid() const;

    Coord3D origin;

    /**
         * @brief the plane will be treated as a single point if its normal is zero
         */
    Coord3D normal;
};

template <typename Real>
class TTriangle3D
{
public:
    COMM_FUNC TTriangle3D();
    COMM_FUNC TTriangle3D(const Coord3D& p0, const Coord3D& p1, const Coord3D& p2);
    COMM_FUNC TTriangle3D(const TTriangle3D& triangle);

    struct Param
    {
        Real u;
        Real v;
        Real w;
    };

    COMM_FUNC Real    area() const;
    COMM_FUNC Coord3D normal() const;

    COMM_FUNC bool    computeBarycentrics(const Coord3D& p, Param& bary) const;
    COMM_FUNC Coord3D computeLocation(const Param& bary) const;

    COMM_FUNC Real maximumEdgeLength() const;

    COMM_FUNC bool isValid() const;

    COMM_FUNC TAlignedBox3D<Real> aabb();

    Coord3D v[3];
};

template <typename Real>
class TRectangle3D
{
public:
    COMM_FUNC TRectangle3D();
    COMM_FUNC TRectangle3D(const Coord3D& c, const Coord3D& a0, const Coord3D& a1, const Coord2D& ext);
    COMM_FUNC TRectangle3D(const TRectangle3D<Real>& rectangle);

    struct Param
    {
        Real u;
        Real v;
    };

    COMM_FUNC TPoint3D<Real> vertex(const int i) const;
    COMM_FUNC TSegment3D<Real> edge(const int i) const;

    COMM_FUNC Real    area() const;
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

template <typename Real>
class TDisk3D
{
public:
    COMM_FUNC TDisk3D();
    COMM_FUNC TDisk3D(const Coord3D& c, const Coord3D& n, const Real& r);
    COMM_FUNC TDisk3D(const TDisk3D<Real>& circle);

    COMM_FUNC Real area();

    COMM_FUNC bool isValid();

    Real    radius;
    Coord3D center;
    Coord3D normal;
};

template <typename Real>
class TSphere3D
{
public:
    COMM_FUNC TSphere3D();
    COMM_FUNC TSphere3D(const Coord3D& c, const Real& r);
    COMM_FUNC TSphere3D(const TSphere3D<Real>& sphere);

    COMM_FUNC Real volume();

    COMM_FUNC bool isValid();

    COMM_FUNC TAlignedBox3D<Real> aabb();

    Real    radius;
    Coord3D center;
};

template <typename Real>
class TCapsule3D
{
public:
    COMM_FUNC TCapsule3D();
    COMM_FUNC TCapsule3D(const Coord3D& v0, const Coord3D& v1, const Real& r);
    COMM_FUNC TCapsule3D(const TCapsule3D<Real>& capsule);

    COMM_FUNC Real volume();

    COMM_FUNC bool isValid();

    Real             radius;
    TSegment3D<Real> segment;
};

/**
     * @brief vertices are ordered so that the normal vectors for the triangular faces point outwards
     *
     */
template <typename Real>
class TTet3D
{
public:
    COMM_FUNC TTet3D();
    COMM_FUNC TTet3D(const Coord3D& v0, const Coord3D& v1, const Coord3D& v2, const Coord3D& v3);
    COMM_FUNC TTet3D(const TTet3D<Real>& tet);

    COMM_FUNC TTriangle3D<Real> face(const int index) const;

    COMM_FUNC Real volume() const;

    COMM_FUNC bool isValid();

    COMM_FUNC bool intersect(const TTet3D<Real>& tet, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2, int need_distance = 1) const;

    COMM_FUNC TAlignedBox3D<Real> aabb();

    Coord3D v[4];
};

template <typename Real>
class TAlignedBox3D
{
public:
    COMM_FUNC TAlignedBox3D();
    COMM_FUNC TAlignedBox3D(const Coord3D& p0);
    COMM_FUNC TAlignedBox3D(const Coord3D& p0, const Coord3D& p1);
    COMM_FUNC TAlignedBox3D(const TAlignedBox3D<Real>& box);

    COMM_FUNC bool intersect(const TAlignedBox3D<Real>& abox, TAlignedBox3D<Real>& interBox) const;
    COMM_FUNC bool meshInsert(const TTriangle3D<Real>& tri) const;
    COMM_FUNC bool overlaps(const TAlignedBox3D<Real>& box) const;
    COMM_FUNC bool isValid();
    COMM_FUNC TAlignedBox3D<Real>& operator+=(const Vector<Real, 3>& p);
    COMM_FUNC TAlignedBox3D<Real>& operator+=(const TAlignedBox3D<Real>& p);
    COMM_FUNC Vector3f             center() const
    {
        return (v0 + v1) * double(0.5);
    }
    COMM_FUNC TAlignedBox3D<Real> operator+(const TAlignedBox3D<Real>& v) const;

    COMM_FUNC Real width() const;
    COMM_FUNC Real height() const;
    COMM_FUNC Real depth() const;

    COMM_FUNC TOrientedBox3D<Real> rotate(const Matrix3D& mat);

    COMM_FUNC inline Real length(unsigned int i) const
    {
        return v1[i] - v0[i];
    }

    // v0 min, v1 max
    Coord3D v0;
    Coord3D v1;
};

template <typename Real>
class TOrientedBox3D
{
public:
    COMM_FUNC TOrientedBox3D();

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
    COMM_FUNC TOrientedBox3D(const Coord3D c, const Coord3D u_t, const Coord3D v_t, const Coord3D w_t, const Coord3D ext);

    COMM_FUNC TOrientedBox3D(const TOrientedBox3D<Real>& obb);

    COMM_FUNC Real volume();

    COMM_FUNC bool isValid();

    COMM_FUNC TOrientedBox3D<Real> rotate(const Matrix3D& mat);

    COMM_FUNC TAlignedBox3D<Real> aabb();

    COMM_FUNC bool point_intersect(const TOrientedBox3D<Real>& OBB, Coord3D& interNorm, Real& interDist, Coord3D& p1, Coord3D& p2) const;
    //COMM_FUNC bool triangle_intersect(const TTriangle3D<Real>& Tri) const;
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

#ifdef PRECISION_FLOAT
template class TPoint3D<float>;
template class TLine3D<float>;
template class TRay3D<float>;
template class TSegment3D<float>;
template class TPlane3D<float>;
template class TTriangle3D<float>;
template class TRectangle3D<float>;
template class TDisk3D<float>;
template class TSphere3D<float>;
template class TCapsule3D<float>;
template class TTet3D<float>;
template class TAlignedBox3D<float>;
template class TOrientedBox3D<float>;
//convenient typedefs
typedef TPoint3D<float>       Point3D;
typedef TLine3D<float>        Line3D;
typedef TRay3D<float>         Ray3D;
typedef TSegment3D<float>     Segment3D;
typedef TPlane3D<float>       Plane3D;
typedef TTriangle3D<float>    Triangle3D;
typedef TRectangle3D<float>   Rectangle3D;
typedef TDisk3D<float>        Disk3D;
typedef TSphere3D<float>      Sphere3D;
typedef TCapsule3D<float>     Capsule3D;
typedef TTet3D<float>         Tet3D;
typedef TAlignedBox3D<float>  AlignedBox3D;
typedef TOrientedBox3D<float> OrientedBox3D;
#else
template class TPoint3D<double>;
template class TLine3D<double>;
template class TRay3D<double>;
template class TSegment3D<double>;
template class TPlane3D<double>;
template class TTriangle3D<double>;
template class TRectangle3D<double>;
template class TDisk3D<double>;
template class TSphere3D<double>;
template class TCapsule3D<double>;
template class TTet3D<double>;
template class TAlignedBox3D<double>;
template class TOrientedBox3D<double>;
//convenient typedefs
typedef TPoint3D<double>       Point3D;
typedef TLine3D<double>        Line3D;
typedef TRay3D<double>         Ray3D;
typedef TSegment3D<double>     Segment3D;
typedef TPlane3D<double>       Plane3D;
typedef TTriangle3D<double>    Triangle3D;
typedef TRectangle3D<double>   Rectangle3D;
typedef TDisk3D<double>        Disk3D;
typedef TSphere3D<double>      Sphere3D;
typedef TCapsule3D<double>     Capsule3D;
typedef TTet3D<double>         Tet3D;
typedef TAlignedBox3D<double>  AlignedBox3D;
typedef TOrientedBox3D<double> OrientedBox3D;
#endif

}  // namespace PhysIKA

#include "Primitive3D.inl"

#endif
