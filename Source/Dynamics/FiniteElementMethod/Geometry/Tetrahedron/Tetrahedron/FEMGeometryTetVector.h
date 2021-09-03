#pragma once

#define CXZ_ZERO 0.000001
#define CXZ_ZERO2 0.00001

#include <vector>
#include <cstddef>
#include <math.h>
namespace cxz {

struct MyVector3
{
    double coord[3];
    MyVector3()
    {
        coord[0] = 0;
        coord[1] = 0;
        coord[2] = 0;
    }
    MyVector3(const double& a0, const double& a1, const double& a2)
    {
        coord[0] = a0;
        coord[1] = a1;
        coord[2] = a2;
    }
};

double    length(const MyVector3& v);
double    area(const MyVector3& p_0, const MyVector3& p_1, const MyVector3& p_2);
double    volume(const MyVector3& p_0, const MyVector3& p_1, const MyVector3& p_2, const MyVector3& p_3);
bool      point_in_tet(const MyVector3& p, const MyVector3 (&t)[4]);
MyVector3 unitize(const MyVector3& v);
MyVector3 operator+(const MyVector3& v1, const MyVector3& v2);
MyVector3 operator-(const MyVector3& v1, const MyVector3& v2);
MyVector3 operator^(const MyVector3& v1, const MyVector3& v2);
double    operator*(const MyVector3& v1, const MyVector3& v2);
MyVector3 operator*(const double k, const MyVector3& v);
bool      operator==(const MyVector3& v1, const MyVector3& v2);
bool      if_zero(const MyVector3& v1);
}  // namespace cxz
