// Copyright (c) 1997
// Utrecht University (The Netherlands),
// ETH Zurich (Switzerland),
// INRIA Sophia-Antipolis (France),
// Max-Planck-Institute Saarbruecken (Germany),
// and Tel-Aviv University (Israel).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Stream_support/include/CGAL/IO/File_writer_OFF_impl.h $
// $Id: File_writer_OFF_impl.h 52164b1 2019-10-19T15:34:59+02:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Lutz Kettner  <kettner@mpi-sb.mpg.de>

#ifdef CGAL_HEADER_ONLY
#define CGAL_INLINE_FUNCTION inline
#else
#define CGAL_INLINE_FUNCTION
#endif

#include <CGAL/basic.h>
#include <iostream>
#include <CGAL/IO/File_writer_OFF.h>

namespace CGAL {

CGAL_INLINE_FUNCTION
void
File_writer_OFF::
write_header( std::ostream& o,
              std::size_t   vertices,
              std::size_t   halfedges,
              std::size_t   facets,
              bool          normals) {
    m_out = &o;
    m_header.set_vertices(  vertices);
    // Don't. This halfdges aren't trusted:
    // m_header.set_halfedges( halfedges);
    (void)halfedges;
    m_header.set_facets(  facets);
    m_header.set_normals( normals);
    // Print header.
    out() << m_header;
}
} //namespace CGAL
// EOF //
