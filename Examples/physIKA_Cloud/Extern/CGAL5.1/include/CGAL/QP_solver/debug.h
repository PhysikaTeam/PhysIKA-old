// Copyright (c) 1997-2007  ETH Zurich (Switzerland).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/QP_solver/include/CGAL/QP_solver/debug.h $
// $Id: debug.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Sven Schoenherr
//                 Bernd Gaertner <gaertner@inf.ethz.ch>
//                 Franz Wessendorp
//                 Kaspar Fischer

#ifndef CGAL_QP_DEBUG_H
#define CGAL_QP_DEBUG_H

#include <CGAL/license/QP_solver.h>


// macro definitions
// =================

// debug
// -----
#if (    defined( CGAL_QP_NO_DEBUG)\
      || defined( CGAL_QP_NO_ASSERTIONS)\
      || defined( CGAL_NO_ASSERTIONS)\
      || defined( CGAL_NO_DEBUG) || defined( NDEBUG))
#  define  CGAL_qpe_debug  if ( 0)
#else
#  define  CGAL_qpe_debug  if ( 1)
#endif // qpe debug

#endif // CGAL_QP_DEBUG_H

// ===== EOF ==================================================================
