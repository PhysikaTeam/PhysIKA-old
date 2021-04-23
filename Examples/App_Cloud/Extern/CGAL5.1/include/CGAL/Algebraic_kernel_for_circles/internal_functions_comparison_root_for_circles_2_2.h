// Copyright (c) 2003-2006  INRIA Sophia-Antipolis (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Algebraic_kernel_for_circles/include/CGAL/Algebraic_kernel_for_circles/internal_functions_comparison_root_for_circles_2_2.h $
// $Id: internal_functions_comparison_root_for_circles_2_2.h 0779373 2020-03-26T13:31:46+01:00 Sébastien Loriot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s)     : Monique Teillaud, Sylvain Pion, Julien Hazebrouck

// Partially supported by the IST Programme of the EU as a Shared-cost
// RTD (FET Open) Project under Contract No  IST-2000-26473
// (ECG - Effective Computational Geometry for Curves and Surfaces)
// and a STREP (FET Open) Project under Contract No  IST-006413
// (ACS -- Algorithms for Complex Shapes)

#ifndef CGAL_ALGEBRAIC_KERNEL_FOR_CIRCLES_COMPARISON_ROOT_FOR_CIRCLES_2_2_H
#define CGAL_ALGEBRAIC_KERNEL_FOR_CIRCLES_COMPARISON_ROOT_FOR_CIRCLES_2_2_H

#include <CGAL/license/Circular_kernel_2.h>



namespace CGAL {
  namespace AlgebraicFunctors{

    template <typename RT>
      Comparison_result
      compare_x(const CGAL::Root_for_circles_2_2<RT>& r1, const CGAL::Root_for_circles_2_2<RT>& r2){
      return compare(r1.x(), r2.x());
    }

    template <typename RT>
      Comparison_result
      compare_y(const CGAL::Root_for_circles_2_2<RT>& r1, const CGAL::Root_for_circles_2_2<RT>& r2){
      return compare(r1.y(), r2.y());
    }

    template <typename RT>
      Comparison_result
      compare_xy(const CGAL::Root_for_circles_2_2<RT>& r1, const CGAL::Root_for_circles_2_2<RT>& r2){
      Comparison_result compx = compare_x(r1, r2);
      if(compx != 0)
        return compx;
      return compare_y(r1, r2);
    }

  } // namespace AlgebraicFunctors
} // namespace CGAL

#endif // CGAL_ALGEBRAIC_KERNEL_FOR_CIRCLES_COMPARISON_ROOT_FOR_CIRCLES_2_2_H
