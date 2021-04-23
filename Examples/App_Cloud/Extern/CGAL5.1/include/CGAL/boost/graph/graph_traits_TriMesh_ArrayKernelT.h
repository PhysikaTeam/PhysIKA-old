// Copyright (c) 2007  GeometryFactory (France).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/BGL/include/CGAL/boost/graph/graph_traits_TriMesh_ArrayKernelT.h $
// $Id: graph_traits_TriMesh_ArrayKernelT.h 52164b1 2019-10-19T15:34:59+02:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s)     : Andreas Fabri, Philipp Moeller

#ifndef CGAL_BOOST_GRAPH_GRAPH_TRAITS_TRIMESH_ARRAYKERNELT_H
#define CGAL_BOOST_GRAPH_GRAPH_TRAITS_TRIMESH_ARRAYKERNELT_H

// http://openmesh.org/Documentation/OpenMesh-Doc-Latest/classOpenMesh_1_1Concepts_1_1KernelT.html
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <CGAL/boost/graph/properties_TriMesh_ArrayKernelT.h>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#define OPEN_MESH_CLASS OpenMesh::TriMesh_ArrayKernelT<K>
#include <CGAL/boost/graph/graph_traits_OpenMesh.h>

#endif // CGAL_BOOST_GRAPH_TRAITS_TRIMESH_ARRAYKERNELT_H
