/*
 * @file tetrahedron_tetrahedron_intersection.cpp
 * @brief detect intersection between two tetrahedra, based on the paper
 *            "Fast Tetrahedron Tetrahedron Overlap Algorithm" 
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Geometry_Intersections/tetrahedron_tetrahedron_intersection.h"

namespace Physika{

namespace GeometryIntersections{

//forward declaration of  internal functions
template <typename Scalar>
bool findSeparatingPlaneForTetAFace(const std::vector<Vector<Scalar,3> > &teta_to_tetb_vec, const Vector<Scalar,3> &tet_a_face_normal,
                                                          std::vector<Scalar> &coord_i, unsigned char &mask_i);
template <typename Scalar>
bool findSeparatingPlaneForTetAEdge(const std::vector<std::vector<Scalar> > &coord, const std::vector<unsigned char> &masks,
                                                           unsigned int edge_face_1, unsigned int edge_face_2);
template <typename Scalar>
bool findSeparatingPlaneForTetBFace(const std::vector<Vector<Scalar,3> > &tetb_to_teta_vec, const Vector<Scalar,3> &tet_b_face_normal);

//the intersection function
template <typename Scalar>
bool intersectTetrahedra(const std::vector<Vector<Scalar,3> > &tet_a, const std::vector<Vector<Scalar,3> > &tet_b)
{
    PHYSIKA_ASSERT(tet_a.size() == 4);
    PHYSIKA_ASSERT(tet_b.size() == 4);

    std::vector<Vector<Scalar,3> > teta_to_tetb_vec(4); //vector from v_a0 to vb_i
    for(unsigned int i = 0; i < teta_to_tetb_vec.size(); ++i)
        teta_to_tetb_vec[i] = tet_b[i] - tet_a[0];
    std::vector<unsigned char> masks(4);
    std::vector<std::vector<Scalar> > coord(4, std::vector<Scalar>(4));
    //test face 0 of tet a
    Vector<Scalar,3> tet_a_face_normal = (tet_a[2] - tet_a[0]).cross(tet_a[1] - tet_a[0]);
    if(tet_a_face_normal.dot(tet_a[3] - tet_a[0]) > 0) //flip normal if not outward
        tet_a_face_normal *= -1;
    if(findSeparatingPlaneForTetAFace(teta_to_tetb_vec,tet_a_face_normal,coord[0],masks[0]))
        return false;
    //test face 1 of tet a
    tet_a_face_normal = (tet_a[1] - tet_a[0]).cross(tet_a[3] - tet_a[0]);
    if(tet_a_face_normal.dot(tet_a[2] - tet_a[0]) > 0) //flip normal if not outward
        tet_a_face_normal *= -1;
    if(findSeparatingPlaneForTetAFace(teta_to_tetb_vec,tet_a_face_normal,coord[1],masks[1]))
        return false;
    //test the common edge of face 0 and face 1
    if(findSeparatingPlaneForTetAEdge(coord,masks,0,1))
        return false;
    //test face 2 of tet a
    tet_a_face_normal = (tet_a[3] - tet_a[0]).cross(tet_a[2] - tet_a[0]);
    if(tet_a_face_normal.dot(tet_a[1] - tet_a[0]) > 0) //flip normal if not outward
        tet_a_face_normal *= -1;
    if(findSeparatingPlaneForTetAFace(teta_to_tetb_vec,tet_a_face_normal,coord[2],masks[2]))
        return false;
    //test edge 0-2 and edge 1-2
    if(findSeparatingPlaneForTetAEdge(coord,masks,0,2))
        return false;
    if(findSeparatingPlaneForTetAEdge(coord,masks,1,2))
        return false;
    //test face 3 of tet a
    tet_a_face_normal = (tet_a[2] - tet_a[1]).cross(tet_a[3] - tet_a[1]);
    if(tet_a_face_normal.dot(tet_a[1] - tet_a[0]) < 0)
        tet_a_face_normal *= -1;
    //update teta_to_tetb_vec before testing
    for(unsigned int i = 0; i < teta_to_tetb_vec.size(); ++i)
        teta_to_tetb_vec[i] = tet_b[i] - tet_a[1];
    if(findSeparatingPlaneForTetAFace(teta_to_tetb_vec,tet_a_face_normal,coord[3],masks[3]))
        return false;
    //test edge 0-3,1-3,2-3
    if(findSeparatingPlaneForTetAEdge(coord,masks,0,3))
        return false;
    if(findSeparatingPlaneForTetAEdge(coord,masks,1,3))
        return false;
    if(findSeparatingPlaneForTetAEdge(coord,masks,2,3))
        return false;
    //test if vertices of tet b is inside tet a, if any vertex is inside, return true
    if((masks[0] | masks[1] | masks[2] | masks[3]) != 0x0F)
        return true;
    
    //test face 0 of tet b
    std::vector<Vector<Scalar,3> > tetb_to_teta_vec(4); //vector from v_b0 to va_i
    for(unsigned int i = 0; i < tetb_to_teta_vec.size(); ++i)
        tetb_to_teta_vec[i] = tet_a[i] - tet_b[0];
    Vector<Scalar,3> tet_b_face_normal = (tet_b[2] - tet_b[0]).cross(tet_b[1] - tet_b[0]);
    if(tet_b_face_normal.dot(tet_b[3] - tet_b[0]) > 0) //flip normal if not outward
        tet_b_face_normal *= -1;
    if(findSeparatingPlaneForTetBFace(tetb_to_teta_vec,tet_b_face_normal))
        return false;
    //test face 1 of tet b
    tet_b_face_normal = (tet_b[1] - tet_b[0]).cross(tet_b[3] - tet_b[0]);
    if(tet_b_face_normal.dot(tet_b[2] - tet_b[0]) > 0) //flip normal if not outward
        tet_b_face_normal *= -1;
    if(findSeparatingPlaneForTetBFace(tetb_to_teta_vec,tet_b_face_normal))
        return false;
    //test face 2 of tet b
    tet_b_face_normal = (tet_b[3] - tet_b[0]).cross(tet_b[2] - tet_b[0]);
    if(tet_b_face_normal.dot(tet_b[1] - tet_b[0]) > 0) //flip normal if not outward
        tet_b_face_normal *= -1;
    if(findSeparatingPlaneForTetBFace(tetb_to_teta_vec,tet_b_face_normal))
        return false;
    //test face 3 of tet b
    tet_b_face_normal = (tet_b[2] - tet_b[1]).cross(tet_b[3] - tet_b[1]);
    if(tet_b_face_normal.dot(tet_b[1] - tet_b[0]) < 0)
        tet_b_face_normal *= -1;
    //update tetb_to_teta_vec before testing
    for(unsigned int i = 0; i < tetb_to_teta_vec.size(); ++i)
        tetb_to_teta_vec[i] = tet_a[i] - tet_b[1];
    if(findSeparatingPlaneForTetBFace(tetb_to_teta_vec,tet_b_face_normal))
        return false;
    //if no separation plane found, return true
    return true;
}

//internal functions used in intersectTetrahedra()
template <typename Scalar>
bool findSeparatingPlaneForTetAFace(const std::vector<Vector<Scalar,3> > &teta_to_tetb_vec, const Vector<Scalar,3> &tet_a_face_normal,
                                                          std::vector<Scalar> &coord_i, unsigned char &mask_i)
{
    mask_i = 0x00;
    const unsigned int shifts[4] = {1,2,4,8};
    for(unsigned int i = 0; i < 4; ++i)
    {
        coord_i[i] = teta_to_tetb_vec[i].dot(tet_a_face_normal);
        if(coord_i[i] > 0)
            mask_i |= shifts[i];
    }
    return mask_i == 0x0F;
}

template <typename Scalar>
bool findSeparatingPlaneForTetAEdge(const std::vector<std::vector<Scalar> > &coord, const std::vector<unsigned char> &masks,
                                                           unsigned int edge_face_1, unsigned int edge_face_2)
{
    const std::vector<Scalar> &coord_face_1 = coord[edge_face_1];
    const std::vector<Scalar> &coord_face_2 = coord[edge_face_2];
    unsigned char mask_face_1 = masks[edge_face_1];
    unsigned char mask_face_2 = masks[edge_face_2];
    
    if ((mask_face_1 | mask_face_2) != 0x0F) // if there is a vertex of b
        return false; // included in (-,-) return false

    mask_face_1 &= (mask_face_1 ^ mask_face_2); // exclude the vertices in (+,+)
    mask_face_2 &= (mask_face_1 ^ mask_face_2);

    // edge 0: 0--1
    if ((mask_face_1 & 1) && // the vertex 0 of b is in (-,+)
        (mask_face_2 & 2)) // the vertex 1 of b is in (+,-)
        if ((coord_face_1[1]*coord_face_2[0] - coord_face_1[0]*coord_face_2[1]) > 0)
            // the edge of b (0,1) intersect (-,-) (see the paper)
            return false;

    if ((mask_face_1 & 2) &&
        (mask_face_2 & 1))
        if ((coord_face_1[1]*coord_face_2[0] - coord_face_1[0]*coord_face_2[1]) < 0)
            return false;

    // edge 1: 0--2
    if ((mask_face_1 & 1) &&
        (mask_face_2 & 4))
        if ((coord_face_1[2]*coord_face_2[0] - coord_face_1[0]*coord_face_2[2]) > 0)
            return false;

    if ((mask_face_1 & 4) &&
        (mask_face_2 & 1))
        if ((coord_face_1[2]*coord_face_2[0] - coord_face_1[0]*coord_face_2[2]) < 0)
            return false;

    // edge 2: 0--3
    if ((mask_face_1 & 1) &&
        (mask_face_2 & 8))
        if ((coord_face_1[3]*coord_face_2[0] - coord_face_1[0]*coord_face_2[3]) > 0)
            return false;

    if ((mask_face_1 & 8) &&
        (mask_face_2 & 1))
        if ((coord_face_1[3]*coord_face_2[0] - coord_face_1[0]*coord_face_2[3]) < 0)
            return false;

    // edge 3: 1--2
    if ((mask_face_1 & 2) &&
        (mask_face_2 & 4))
        if ((coord_face_1[2]*coord_face_2[1] - coord_face_1[1]*coord_face_2[2]) > 0)
            return false;

    if ((mask_face_1 & 4) &&
        (mask_face_2 & 2))
        if ((coord_face_1[2]*coord_face_2[1] - coord_face_1[1]*coord_face_2[2]) < 0)
            return false;

    // edge 4: 1--3
    if ((mask_face_1 & 2) &&
        (mask_face_2 & 8))
        if ((coord_face_1[3]*coord_face_2[1] - coord_face_1[1]*coord_face_2[3]) > 0)
            return false;

    if ((mask_face_1 & 8) &&
        (mask_face_2 & 2))
        if ((coord_face_1[3]*coord_face_2[1] - coord_face_1[1]*coord_face_2[3]) < 0)
            return false;

    // edge 5: 2--3
    if ((mask_face_1 & 4) &&
        (mask_face_2 & 8))
        if ((coord_face_1[3]*coord_face_2[2] - coord_face_1[2]*coord_face_2[3]) > 0)
            return false;

    if ((mask_face_1 & 8) &&
        (mask_face_2 & 4))
        if ((coord_face_1[3]*coord_face_2[2] - coord_face_1[2]*coord_face_2[3]) < 0)
            return false;

    //no separating plane found
    return true;
}

template <typename Scalar>
bool findSeparatingPlaneForTetBFace(const std::vector<Vector<Scalar,3> > &tetb_to_teta_vec, const Vector<Scalar,3> &tet_b_face_normal)
{
    unsigned char mask = 0x00;
    const unsigned int shifts[4] = {1,2,4,8};
    for(unsigned int i = 0; i < 4; ++i)
    {
        if(tetb_to_teta_vec[i].dot(tet_b_face_normal) > 0)
            mask |= shifts[i];
    }
    return mask == 0x0F;
}

//explicit instantiation
template bool intersectTetrahedra<float>(const std::vector<Vector<float,3> > &tet_a, const std::vector<Vector<float,3> > &tet_b);
template bool intersectTetrahedra<double>(const std::vector<Vector<double,3> > &tet_a, const std::vector<Vector<double,3> > &tet_b);
}

}  //end of namespace Physika
