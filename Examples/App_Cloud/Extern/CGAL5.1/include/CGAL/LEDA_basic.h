// Copyright (c) 1999
// Utrecht University (The Netherlands),
// ETH Zurich (Switzerland),
// INRIA Sophia-Antipolis (France),
// Max-Planck-Institute Saarbruecken (Germany),
// and Tel-Aviv University (Israel).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/LEDA/include/CGAL/LEDA_basic.h $
// $Id: LEDA_basic.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Matthias Baesken



#ifndef CGAL_LEDA_BASIC_H
#define CGAL_LEDA_BASIC_H

#include <CGAL/config.h>

#ifdef CGAL_USE_LEDA
// The following is needed for LEDA 4.4 due to min/max problems...
#  define LEDA_NO_MIN_MAX_TEMPL

#include <LEDA/system/basic.h>

#ifdef LEDA_NAMESPACE
#  define CGAL_LEDA_SCOPE  leda
#else
#  define CGAL_LEDA_SCOPE
#endif


#endif


#endif
