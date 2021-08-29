#pragma once

#include "Dynamics/RigidBody/ForwardDynamicsSolver.h"

namespace PhysIKA {
struct ContactPointX
{
    //public:
    float  depth      = 0;
    int    beFriction = -1;
    float3 point;
    float3 normal;
    //float point[3];
    //float normal[3];

    //COMM_FUNC ContactPointX()
    //{
    //    depth = 0;
    //}

    //COMM_FUNC ContactPointX(const ContactPointX& cp)
    //{
    //    depth = cp.depth;
    //    point = cp.point;
    //    normal = cp.normal;
    //}

    //COMM_FUNC ContactPointX(const ContactPointX&& cp)
    //{
    //    depth = cp.depth;
    //    point = cp.point;
    //    normal = cp.normal;
    //}

    COMM_FUNC bool operator<(const ContactPointX& p) const
    {
        return depth < p.depth;
    }
    COMM_FUNC bool operator>(const ContactPointX& p) const
    {
        return depth > p.depth;
    }

    COMM_FUNC ContactPointX& operator=(const ContactPointX& p)
    {
        depth      = p.depth;
        point      = p.point;
        normal     = p.normal;
        beFriction = p.beFriction;
        return *this;
    }
    //COMM_FUNC ContactPointX operator /(float val) {
    //    ContactPointX p = *this;
    //    p.depth /= val;
    //    return p;
    //}

    //COMM_FUNC ContactPointX operator +(const ContactPointX& p)const {
    //    ContactPointX p2 = *this;
    //    p2.depth += p.depth;
    //    return p2;
    //}
};

//class FeathContactSolver
//{
//public:

//    void solve
//};
}  // namespace PhysIKA