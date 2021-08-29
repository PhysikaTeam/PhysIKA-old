/*
 * @file cxx11_support.h 
 * @brief macros indicating whether specific c++0x(c++11) features are supported by compiler
 * @author FeiZhu
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_CXX11_SUPPORT_H_
#define PHYSIKA_CORE_UTILITIES_CXX11_SUPPORT_H_

/*
 * As we know, no compiler can fully support c++11 features by now.
 * So:
 *    We must manually check whether the specific feature we want to use is supported by all
 *    target compilers of PhysIKA (GNU g++ && MSVC). If it's not, we must provide workaround
 *    code so that PhysIKA works with all target compilers.
 * Here is one site where we can check the c++0x(c++11) features support in popular compilers:
 *    https://wiki.apache.org/stdcxx/C%2B%2B0xCompilerSupport
 * 
 * If you don't want this trouble, just avoid using c++11 features.
 */

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
/* GNU GCC/G++. --------------------------------------------- */

//static_assert since g++ 4.3
#if ((defined(__GNUC__) && (__GNUC__ >= 4)) || (defined(__GNUG__) && (__GNUG__ >= 4))) \
    && (__GNUC_MINOR__ >= 3)
#define SUPPORT_STATIC_ASSERT
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

#elif defined(_MSC_VER)
/* Microsoft Visual Studio. --------------------------------- */

//static assert since msvc 10.0
#if _MSC_VER >= 1000
#define SUPPORT_STATIC_ASSERT
#endif

#endif

#endif  //PHYSIKA_CORE_UTILITIES_CXX11_SUPPORT_H_
