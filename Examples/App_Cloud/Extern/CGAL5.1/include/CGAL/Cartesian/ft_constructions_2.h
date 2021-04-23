// Copyright (c) 2000
// Utrecht University (The Netherlands),
// ETH Zurich (Switzerland),
// INRIA Sophia-Antipolis (France),
// Max-Planck-Institute Saarbruecken (Germany),
// and Tel-Aviv University (Israel).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Cartesian_kernel/include/CGAL/Cartesian/ft_constructions_2.h $
// $Id: ft_constructions_2.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Herve Bronnimann

#ifndef CGAL_CARTESIAN_FT_CONSTRUCTIONS_2_H
#define CGAL_CARTESIAN_FT_CONSTRUCTIONS_2_H

namespace CGAL {

template < class K >
inline
typename K::FT
squared_distance(const PointC2<K> &p,
                 const PointC2<K> &q)
{
  return squared_distanceC2(p.x(), p.y(), q.x(), q.y());
}

template < class K >
inline
typename K::FT
scaled_distance_to_line(const LineC2<K> &l,
                        const PointC2<K> &p)
{
  return scaled_distance_to_lineC2(l.a(), l.b(), l.c(), p.x(), p.y());
}

template < class K >
inline
typename K::FT
scaled_distance_to_line(const PointC2<K> &p,
                        const PointC2<K> &q,
                        const PointC2<K> &r)
{
  return scaled_distance_to_lineC2(p.x(), p.y(), q.x(), q.y(), r.x(), r.y());
}

template < class K >
inline
typename K::FT
line_y_at_x(const LineC2<K> &l, const typename K::FT &x)
{
  return line_y_at_xC2(l.a(), l.b(), l.c(), x);
}

template < class K >
inline
typename K::FT
line_x_at_y(const LineC2<K> &l, const typename K::FT &y)
{
  return line_y_at_xC2(l.b(), l.a(), l.c(), y);
}

} //namespace CGAL

#endif // CGAL_CARTESIAN_FT_CONSTRUCTIONS_2_H
