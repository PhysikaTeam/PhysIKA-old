// Copyright (c) 2006 Foundation for Research and Technology-Hellas (Greece).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Voronoi_diagram_2/include/CGAL/Voronoi_diagram_2/Edge_less.h $
// $Id: Edge_less.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Menelaos Karavelas <mkaravel@iacm.forth.gr>

#ifndef CGAL_VORONOI_DIAGRAM_2_EDGE_LESS_H
#define CGAL_VORONOI_DIAGRAM_2_EDGE_LESS_H 1

#include <CGAL/license/Voronoi_diagram_2.h>


#include <CGAL/Voronoi_diagram_2/basic.h>

namespace CGAL {

namespace VoronoiDiagram_2 { namespace Internal {

template<class Edge_t>
struct Edge_less
{
  typedef Edge_t        Edge;
  typedef bool          result_type;

  bool operator()(const Edge& e1, const Edge& e2) const {
    if ( e1.first != e2.first ) { return e1.first < e2.first; }
    return e1.second < e2.second;
  }
};


} } //namespace VoronoiDiagram_2::Internal

} //namespace CGAL

#endif // CGAL_VORONOI_DIAGRAM_2_EDGE_LESS_H
