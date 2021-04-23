// Copyright (c) 2010 GeometryFactory (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Intersections_3/include/CGAL/Intersections_3/Iso_cuboid_3_Plane_3.h $
// $Id: Iso_cuboid_3_Plane_3.h 90d2e03 2020-01-15T13:32:11+01:00 Maxime Gimeno
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Sebastien Loriot
//

#ifndef CGAL_INTERSECTIONS_3_ISO_CUBOID_3_PLANE_3_H
#define CGAL_INTERSECTIONS_3_ISO_CUBOID_3_PLANE_3_H

#include <CGAL/Iso_cuboid_3.h>
#include <CGAL/Plane_3.h>
#include <CGAL/Intersections_3/internal/Bbox_3_Plane_3_do_intersect.h>
#include <CGAL/Intersections_3/internal/Iso_cuboid_3_Plane_3_intersection.h>


namespace CGAL {
  CGAL_DO_INTERSECT_FUNCTION(Iso_cuboid_3, Plane_3, 3)
  CGAL_INTERSECTION_FUNCTION(Iso_cuboid_3, Plane_3, 3)
}

#endif // CGAL_INTERSECTIONS_3_ISO_CUBOID_3_PLANE_3_H
