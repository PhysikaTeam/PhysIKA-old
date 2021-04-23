// Copyright (c) 1999  Max-Planck-Institute Saarbruecken (Germany).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Convex_hull_2/include/CGAL/convex_hull_traits_2.h $
// $Id: convex_hull_traits_2.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Stefan Schirra

// This file's name must begin with a lower-case letter for backward
// compatability.  Unfortunately, you can't have a file that differs only
// in capitalization on the Windows platforms.

#ifndef CGAL_CONVEX_HULL_TRAITS_2_H
#define CGAL_CONVEX_HULL_TRAITS_2_H

#include <CGAL/license/Convex_hull_2.h>

#include <CGAL/Point_2.h>
#include <CGAL/predicates_on_points_2.h>
#include <CGAL/distance_predicates_2.h>

namespace CGAL {

template <class K_>
class Convex_hull_traits_2 : public K_
{
public:
  Convex_hull_traits_2() { }
  Convex_hull_traits_2(const K_& k) : K_(k) { }
};

template <class K_>
class convex_hull_traits_2 : public Convex_hull_traits_2<K_>
{
  convex_hull_traits_2() { }
  convex_hull_traits_2(const K_& k) : K_(k) { }
};

} //namespace CGAL

#endif // CGAL_CONVEX_HULL_TRAITS_2_H
