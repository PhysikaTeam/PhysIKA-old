// Copyright (c) 1999,2007,2012
// Utrecht University (The Netherlands),
// ETH Zurich (Switzerland),
// INRIA Sophia-Antipolis (France),
// Max-Planck-Institute Saarbruecken (Germany),
// and Tel-Aviv University (Israel).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Number_types/include/CGAL/number_type_config.h $
// $Id: number_type_config.h 52164b1 2019-10-19T15:34:59+02:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Stefan Schirra, Michael Hemmer

#ifndef CGAL_NUMBER_TYPE_CONFIG_H
#define CGAL_NUMBER_TYPE_CONFIG_H

#include <CGAL/config.h>

#define CGAL_PI 3.14159265358979323846


#ifdef CGAL_USE_NTS_NAMESPACE

#define CGAL_NTS_BEGIN_NAMESPACE namespace NTS {
#define CGAL_NTS_END_NAMESPACE }
#define CGAL_NTS ::CGAL::NTS::

#else

#define CGAL_NTS_BEGIN_NAMESPACE
#define CGAL_NTS_END_NAMESPACE
#define CGAL_NTS ::CGAL::

#endif

#endif // CGAL_NUMBER_TYPE_CONFIG_H
