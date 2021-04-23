// Copyright (c) 2005
// Utrecht University (The Netherlands),
// ETH Zurich (Switzerland),
// INRIA Sophia-Antipolis (France),
// Max-Planck-Institute Saarbruecken (Germany),
// and Tel-Aviv University (Israel).  All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Kernel_23/include/CGAL/Kernel/Same_uncertainty.h $
// $Id: Same_uncertainty.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Sylvain Pion

#ifndef CGAL_SAME_UNCERTAINTY_H
#define CGAL_SAME_UNCERTAINTY_H

#include <CGAL/config.h>

namespace CGAL {

template < typename T1, typename T2 >
struct Same_uncertainty
{
  typedef T1 type;
};

template < typename T > class Uncertain;
template < typename T > struct Sgn;

template < typename T1, typename T2 >
struct Same_uncertainty < T1, Uncertain<T2> >
{
  typedef Uncertain<T1> type;
};

// Short cut to extract uncertainty from a number type directly.
template < typename T1, typename NT >
struct Same_uncertainty_nt
  : Same_uncertainty <T1, typename Sgn<NT>::result_type > {};

} //namespace CGAL

#endif // CGAL_SAME_UNCERTAINTY_H
