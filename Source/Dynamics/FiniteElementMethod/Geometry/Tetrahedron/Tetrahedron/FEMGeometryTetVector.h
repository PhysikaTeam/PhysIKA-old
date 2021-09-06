#pragma once

#define CXZ_ZERO 0.000001
#define CXZ_ZERO2 0.00001

#include <vector>
#include <cstddef>
#include <math.h>
namespace cxz {

/**
 * @brief FEM Geometry MyVector3
 * 
 */
struct MyVector3
{
    double coord[3];
    /**
     * @brief Construct a new My Vector 3 object
     * 
     */
    MyVector3()
    {
        coord[0] = 0;
        coord[1] = 0;
        coord[2] = 0;
    }

    /**
     * @brief Construct a new My Vector 3 object
     * 
     * @param a0 
     * @param a1 
     * @param a2 
     */
    MyVector3(const double& a0, const double& a1, const double& a2)
    {
        coord[0] = a0;
        coord[1] = a1;
        coord[2] = a2;
    }
};

/**
 * @brief Get the length of the MyVector object
 * 
 * @param v 
 * @return double 
 */
double    length(const MyVector3& v);

/**
 * @brief Get the area of the MyVector object
 * 
 * @param p_0 
 * @param p_1 
 * @param p_2 
 * @return double 
 */
double    area(const MyVector3& p_0, const MyVector3& p_1, const MyVector3& p_2);

/**
 * @brief Get the volume of the MyVector object
 * 
 * @param p_0 
 * @param p_1 
 * @param p_2 
 * @param p_3 
 * @return double 
 */
double    volume(const MyVector3& p_0, const MyVector3& p_1, const MyVector3& p_2, const MyVector3& p_3);

/**
 * @brief Get the point in the tetrahedron object
 * 
 * @param p 
 * @param t 
 * @return true 
 * @return false 
 */
bool      point_in_tet(const MyVector3& p, const MyVector3 (&t)[4]);

/**
 * @brief Unitize the MyVector object
 * 
 * @param v 
 * @return MyVector3 
 */
MyVector3 unitize(const MyVector3& v);
MyVector3 operator+(const MyVector3& v1, const MyVector3& v2);
MyVector3 operator-(const MyVector3& v1, const MyVector3& v2);
MyVector3 operator^(const MyVector3& v1, const MyVector3& v2);
double    operator*(const MyVector3& v1, const MyVector3& v2);
MyVector3 operator*(const double k, const MyVector3& v);
bool      operator==(const MyVector3& v1, const MyVector3& v2);

/**
 * @brief Determine whether the object is zero
 * 
 * @param v1 
 * @return true 
 * @return false 
 */
bool      if_zero(const MyVector3& v1);
}  // namespace cxz
