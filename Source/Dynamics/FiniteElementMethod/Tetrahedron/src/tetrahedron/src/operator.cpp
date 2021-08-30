#include "../inc/vector.h"
using namespace cxz;

cxz::MyVector3 cxz::operator+(const cxz::MyVector3& v1, const cxz::MyVector3& v2)
{
    cxz::MyVector3 v3;
    v3.coord[0] = v1.coord[0] + v2.coord[0];
    v3.coord[1] = v1.coord[1] + v2.coord[1];
    v3.coord[2] = v1.coord[2] + v2.coord[2];
    return v3;
}

cxz::MyVector3 cxz::operator-(const cxz::MyVector3& v1, const cxz::MyVector3& v2)
{
    cxz::MyVector3 v3;
    v3.coord[0] = v1.coord[0] - v2.coord[0];
    v3.coord[1] = v1.coord[1] - v2.coord[1];
    v3.coord[2] = v1.coord[2] - v2.coord[2];
    return v3;
}

cxz::MyVector3 cxz::operator^(const cxz::MyVector3& v1, const cxz::MyVector3& v2)
{
    cxz::MyVector3 v3;
    v3.coord[0] = v1.coord[1] * v2.coord[2] - v1.coord[2] * v2.coord[1];
    v3.coord[1] = v1.coord[2] * v2.coord[0] - v1.coord[0] * v2.coord[2];
    v3.coord[2] = v1.coord[0] * v2.coord[1] - v1.coord[1] * v2.coord[0];
    return v3;
}

double cxz::operator*(const cxz::MyVector3& v1, const cxz::MyVector3& v2)
{
    return (v1.coord[0] * v2.coord[0] + v1.coord[1] * v2.coord[1] + v1.coord[2] * v2.coord[2]);
}

cxz::MyVector3 cxz::operator*(const double k, const cxz::MyVector3& v)
{
    cxz::MyVector3 temp;
    temp.coord[0] = v.coord[0] * k;
    temp.coord[1] = v.coord[1] * k;
    temp.coord[2] = v.coord[2] * k;
    return temp;
}

bool cxz::operator==(const cxz::MyVector3& v1, const cxz::MyVector3& v2)
{
    if (length(v1 - v2) < CXZ_ZERO)
        return 1;
    return 0;
}

bool cxz::if_zero(const MyVector3& v1)
{
    if (length(v1) < CXZ_ZERO)
        return 1;
    return 0;
}

double cxz::length(const MyVector3& v)
{
    return pow(v.coord[0] * v.coord[0] + v.coord[1] * v.coord[1] + v.coord[2] * v.coord[2], 0.5);
}

double cxz::area(const MyVector3& p_0, const MyVector3& p_1, const MyVector3& p_2)
{
    return 0.5 * length((p_1 - p_0) ^ (p_2 - p_0));
}

double cxz::volume(const MyVector3& p_0, const MyVector3& p_1, const MyVector3& p_2, const MyVector3& p_3)
{
    return 1.0 / 6 * fabs(((p_1 - p_0) ^ (p_2 - p_0)) * (p_3 - p_0));
}

MyVector3 cxz::unitize(const MyVector3& v)
{
    MyVector3 v1;
    double    len = length(v);
    if (len < CXZ_ZERO)
        v1.coord[0] = v1.coord[1] = v1.coord[2] = 0;
    else
    {
        v1.coord[0] = v.coord[0] / len;
        v1.coord[1] = v.coord[1] / len;
        v1.coord[2] = v.coord[2] / len;
    }
    return v1;
}

bool cxz::point_in_tet(const MyVector3& p, const MyVector3 (&t)[4])
{
    if ((volume(p, t[0], t[1], t[2]) + volume(p, t[1], t[2], t[3]) + volume(p, t[2], t[3], t[0])) < 1.00001 * volume(t[0], t[1], t[2], t[3]))
        return 1;
    return 0;
}