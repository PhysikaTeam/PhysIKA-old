/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: collision detection api entry point
 * @version    : 1.0
 */

#pragma once

#include "Framework/Framework/CollidableObject.h"
#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Framework/Framework/ModuleTopology.h"

namespace PhysIKA {
/**
     * Data structure to store collision results.
     *
     * Sample usage:
     * TrianglePair result = ...
     * auto meshID = result.id0();
     * auto triangleID = result.id1();
     */
class TrianglePair
{
    unsigned int _id[2];  //< ! mesh index - triangle index pair

public:
    /**
         * get mesh index
         *
         * @return mesh index
         */
    unsigned int id0() const
    {
        return _id[0];
    }

    /**
         * get triangle index
         *
         * @return triangle index
         */
    unsigned int id1() const
    {
        return _id[1];
    }

    /**
         * constructor
         *
         * @param[in] id1 mesh id
         * @param[in] id2 triangle id
         */
    TrianglePair(unsigned int id1, unsigned int id2)
    {
        if (id1 < id2)
        {
            _id[0] = id1;
            _id[1] = id2;
        }
        else
        {
            _id[0] = id2;
            _id[1] = id1;
        }
    }
    /**
         * get mesh index and triangle index
         *
         * @param[out] id1 mesh index
         * @param[out] id2 triangle index
         */
    void get(unsigned int& id1, unsigned int& id2)
    {
        id1 = _id[0];
        id2 = _id[1];
    }

    /**
         * operator < to define partial order of TrianglePair
         *
         * @param[in] other the TrianglePair to be compared with
         */
    bool operator<(const TrianglePair& other) const
    {
        if (_id[0] == other._id[0])
            return _id[1] < other._id[1];
        else
            return _id[0] < other._id[0];
    }
};

template <typename TDataType>
class CollidableTriangle : public CollidableObject
{
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;
    CollidableTriangle();
    virtual ~CollidableTriangle();

    // very robust triangle intersection test
    // uses no divisions
    // works on coplanar triangles
    static inline double fmax(double a, double b, double c)
    {
        double t = a;
        if (b > t)
            t = b;
        if (c > t)
            t = c;
        return t;
    }

    static inline double fmin(double a, double b, double c)
    {
        double t = a;
        if (b < t)
            t = b;
        if (c < t)
            t = c;
        return t;
    }
    static inline int project3(const Vector3f& ax,
                               const Vector3f& p1,
                               const Vector3f& p2,
                               const Vector3f& p3)
    {
        double P1 = ax.dot(p1);
        double P2 = ax.dot(p2);
        double P3 = ax.dot(p3);

        double mx1 = fmax(P1, P2, P3);
        double mn1 = fmin(P1, P2, P3);

        if (mn1 > 0)
            return 0;
        if (0 > mx1)
            return 0;
        return 1;
    }
    static inline int project6(Vector3f& ax,
                               Vector3f& p1,
                               Vector3f& p2,
                               Vector3f& p3,
                               Vector3f& q1,
                               Vector3f& q2,
                               Vector3f& q3)
    {
        double P1 = ax.dot(p1);
        double P2 = ax.dot(p2);
        double P3 = ax.dot(p3);
        double Q1 = ax.dot(q1);
        double Q2 = ax.dot(q2);
        double Q3 = ax.dot(q3);

        double mx1 = fmax(P1, P2, P3);
        double mn1 = fmin(P1, P2, P3);
        double mx2 = fmax(Q1, Q2, Q3);
        double mn2 = fmin(Q1, Q2, Q3);

        if (mn1 > mx2)
            return 0;
        if (mn2 > mx1)
            return 0;
        return 1;
    }
    static bool
    tri_contact(Vector3f& P1, Vector3f& P2, Vector3f& P3, Vector3f& Q1, Vector3f& Q2, Vector3f& Q3)
    {
        Vector3f p1;
        Vector3f p2 = P2 - P1;
        Vector3f p3 = P3 - P1;
        Vector3f q1 = Q1 - P1;
        Vector3f q2 = Q2 - P1;
        Vector3f q3 = Q3 - P1;

        Vector3f e1 = p2 - p1;
        Vector3f e2 = p3 - p2;
        Vector3f e3 = p1 - p3;

        Vector3f f1 = q2 - q1;
        Vector3f f2 = q3 - q2;
        Vector3f f3 = q1 - q3;

        Vector3f n1 = e1.cross(e2);
        Vector3f m1 = f1.cross(f2);

        Vector3f g1 = e1.cross(n1);
        Vector3f g2 = e2.cross(n1);
        Vector3f g3 = e3.cross(n1);

        Vector3f h1 = f1.cross(m1);
        Vector3f h2 = f2.cross(m1);
        Vector3f h3 = f3.cross(m1);

        Vector3f ef11 = e1.cross(f1);
        Vector3f ef12 = e1.cross(f2);
        Vector3f ef13 = e1.cross(f3);
        Vector3f ef21 = e2.cross(f1);
        Vector3f ef22 = e2.cross(f2);
        Vector3f ef23 = e2.cross(f3);
        Vector3f ef31 = e3.cross(f1);
        Vector3f ef32 = e3.cross(f2);
        Vector3f ef33 = e3.cross(f3);

        // now begin the series of tests
        if (!project3(n1, q1, q2, q3))
            return false;
        if (!project3(m1, -q1, p2 - q1, p3 - q1))
            return false;

        if (!project6(ef11, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(ef12, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(ef13, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(ef21, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(ef22, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(ef23, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(ef31, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(ef32, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(ef33, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(g1, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(g2, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(g3, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(h1, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(h2, p1, p2, p3, q1, q2, q3))
            return false;
        if (!project6(h3, p1, p2, p3, q1, q2, q3))
            return false;

        return true;
    }

    static bool checkSelfIJ(TriangleMesh<DataType3f>* ma, int fa, TriangleMesh<DataType3f>* mb, int fb)
    {
        Triangle& a = ma->triangleSet->getHTriangles()[fa];
        Triangle& b = mb->triangleSet->getHTriangles()[fb];

        if (ma == mb)
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    if (a[k] == b[l])
                    {
                        //printf("covertex triangle found!\n");
                        return false;
                    }

        Vector3f p0 = ma->triangleSet->gethPoints()[a[0]];
        Vector3f p1 = ma->triangleSet->gethPoints()[a[1]];
        Vector3f p2 = ma->triangleSet->gethPoints()[a[2]];
        Vector3f q0 = mb->triangleSet->gethPoints()[b[0]];
        Vector3f q1 = mb->triangleSet->gethPoints()[b[1]];
        Vector3f q2 = mb->triangleSet->gethPoints()[b[2]];

        return tri_contact(p0, p1, p2, q0, q1, q2);
    }

private:
};
}  // namespace PhysIKA
