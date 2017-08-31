/*
 * @file geometry_intersection_test.cpp 
 * @brief test the methods in Physika_Geometry/Geometry_Intersections/.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <vector>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Geometry_Intersections/tetrahedron_tetrahedron_intersection.h"

using Physika::Vector;
using Physika::GeometryIntersections::intersectTetrahedra;
using namespace std;

int main()
{
    //test tetrahedron-tetrahedron intersection
    Vector<float,3> point_a(0,0,0), point_b(1,0,0), point_c(0,1,0), point_d(0,0,1);
    vector<Vector<float,3> > tet_a;
    tet_a.push_back(point_a);
    tet_a.push_back(point_b);
    tet_a.push_back(point_c);
    tet_a.push_back(point_d);
    Vector<float,3> point_e(1,0,1), point_f(0,1,1), point_g(1,1,0), point_h(1,1,1);
    vector<Vector<float,3> > tet_b;
    tet_b.push_back(point_e);
    tet_b.push_back(point_f);
    tet_b.push_back(point_g);
    tet_b.push_back(point_h);
    cout<<"Points: a(0,0,0), b (1,0,0), c(0,1,0), d(0,0,1), e(1,0,1), f(0,1,1), g(1,1,0), h(1,1,1).\n";
    cout<<"abcd intersect with efgh? ";
    if(intersectTetrahedra(tet_a,tet_b))
        cout<<"Yes\n";
    else
        cout<<"No\n";
    vector<Vector<float,3> > tet_c;
    tet_c.push_back(point_a);
    tet_c.push_back(point_f);
    tet_c.push_back(point_g);
    tet_c.push_back(point_h);
    cout<<"abcd intersect with afgh? ";
    if(intersectTetrahedra(tet_a,tet_c))
        cout<<"Yes\n";
    else
        cout<<"No\n";
    return 0;
}
