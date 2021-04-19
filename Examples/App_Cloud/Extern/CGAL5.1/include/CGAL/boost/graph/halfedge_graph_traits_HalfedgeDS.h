// Copyright (c) 2007  GeometryFactory (France).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/HalfedgeDS/include/CGAL/boost/graph/halfedge_graph_traits_HalfedgeDS.h $
// $Id: halfedge_graph_traits_HalfedgeDS.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Andreas Fabri, Fernando Cacciola

#ifndef CGAL_BOOST_GRAPH_HALFEDGE_GRAPH_TRAITS_HALFEDGEDS_H
#define CGAL_BOOST_GRAPH_HALFEDGE_GRAPH_TRAITS_HALFEDGEDS_H

#include <boost/config.hpp>
#include <boost/iterator_adaptors.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>

#include <CGAL/basic.h>

namespace CGAL {

template < class HDS >
class HDS_all_undirected_edges_const_iterator
  : public HDS_all_edges_iterator_base<HDS,typename HDS::Edge_const_iterator,typename HDS::Halfedge_const_handle>
{
  typedef HDS_all_edges_iterator_base<HDS,typename HDS::Edge_const_iterator,typename HDS::Halfedge_const_handle> Base ;

public:

  typedef typename HDS::Edge_const_iterator Iterator;

  HDS_all_undirected_edges_const_iterator() {}
  HDS_all_undirected_edges_const_iterator( Iterator j) : Base(j) {}
};

template < class HDS >
class HDS_all_undirected_edges_iterator
  : public HDS_all_edges_iterator_base<HDS,typename HDS::Edge_iterator,typename HDS::Halfedge_handle>
{
  typedef HDS_all_edges_iterator_base<HDS,typename HDS::Edge_iterator,typename HDS::Halfedge_handle> Base ;

public:

  typedef typename HDS::Edge_iterator Iterator;

  HDS_all_undirected_edges_iterator() {}
  HDS_all_undirected_edges_iterator( Iterator j) : Base(j) {}
};


template <class HDS_>
struct HDS_halfedge_graph_traits
{
public :

  typedef HDS_ HDS;

  typedef HDS_all_undirected_edges_iterator<HDS> undirected_edge_iterator;

  typedef typename HDS::Vertex::Point Point ;
};

} //namespace CGAL

#endif // CGAL_BOOST_GRAPH_HALFEDGE_GRAPH_TRAITS_HALFEDGEDS_H
