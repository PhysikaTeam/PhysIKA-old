/*
 * @file tetrahedron_tetrahedron_intersection.h
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

#ifndef PHYSIKA_GEOMETRY_GEOMETRY_INTERSECTIONS_TETRAHEDRON_TETRAHEDRON_INTERSECTION_H_
#define PHYSIKA_GEOMETRY_GEOMETRY_INTERSECTIONS_TETRAHEDRON_TETRAHEDRON_INTERSECTION_H_

#include <vector>

namespace Physika{

template <typename Scalar, int Dim> class Vector;

namespace GeometryIntersections{

template <typename Scalar>
bool intersectTetrahedra(const std::vector<Vector<Scalar,3> > &tet_a, const std::vector<Vector<Scalar,3> > &tet_b);

}  //end of namespace GeometryIntersections

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_GEOMETRY_INTERSECTIONS_TETRAHEDRON_TETRAHEDRON_INTERSECTION_H_
