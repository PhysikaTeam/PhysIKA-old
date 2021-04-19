// Copyright (c) 1997-2001
// ETH Zurich (Switzerland).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Bounding_volumes/include/CGAL/Min_ellipse_2_traits_2.h $
// $Id: Min_ellipse_2_traits_2.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Sven Schoenherr <sven@inf.ethz.ch>, Bernd Gaertner

#ifndef CGAL_MIN_ELLIPSE_2_TRAITS_2_H
#define CGAL_MIN_ELLIPSE_2_TRAITS_2_H

#include <CGAL/license/Bounding_volumes.h>


#include <CGAL/Min_ellipse_2/Optimisation_ellipse_2.h>

namespace CGAL {

// Class declarations
// ==================
template < class Traits_ >
class Min_ellipse_2;

template < class K_ >
class Min_ellipse_2_traits_2;

// Class interface and implementation
// ==================================
template < class K_ >
class Min_ellipse_2_traits_2 {
  public:
    // types
    typedef  K_                               K;
    typedef  typename K::Point_2              Point;
    typedef  CGAL::Optimisation_ellipse_2<K>  Ellipse;

private:
    // data members
    Ellipse  ellipse;                                // current ellipse

    // friends
    friend  class CGAL::Min_ellipse_2< CGAL::Min_ellipse_2_traits_2<K> >;

  public:
    // creation (use default implementations)
    // Min_ellipse_2_traits_2( );
    // Min_ellipse_2_traits_2( Min_ellipse_2_traits_2<K> const&);
};

} //namespace CGAL

#endif // CGAL_MIN_ELLIPSE_2_TRAITS_2_H

// ===== EOF ==================================================================
