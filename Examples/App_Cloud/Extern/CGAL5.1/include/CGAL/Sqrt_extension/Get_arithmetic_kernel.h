// Copyright (c) 2006-2008 Max-Planck-Institute Saarbruecken (Germany).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.1/Number_types/include/CGAL/Sqrt_extension/Get_arithmetic_kernel.h $
// $Id: Get_arithmetic_kernel.h 52164b1 2019-10-19T15:34:59+02:00 Sébastien Loriot
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Michael Hemmer   <hemmer@mpi-inf.mpg.de>


#ifndef CGAL_SQRT_EXTENSION_GET_ARITHMETIC_KERNEL_H
#define CGAL_SQRT_EXTENSION_GET_ARITHMETIC_KERNEL_H

#include <CGAL/basic.h>
#include <CGAL/Get_arithmetic_kernel.h>

namespace CGAL {

template <class COEFF, class ROOT, class ACDE_TAG,class FP_TAG>
struct Get_arithmetic_kernel<Sqrt_extension<COEFF,ROOT,ACDE_TAG,FP_TAG> >{
  typedef Get_arithmetic_kernel<COEFF> GET;
  typedef typename GET::Arithmetic_kernel Arithmetic_kernel;
};

} //namespace CGAL

#endif
