/*
 * @file physika_assert.h 
 * @brief Customized assert macro for Physika.
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_PHYSIKA_ASSERT_H_
#define PHYSIKA_CORE_UTILITIES_PHYSIKA_ASSERT_H_

#include "Physika_Core/Utilities/global_config.h"
#include <cassert>

//for now, PHYSIKA_ASSERT() is just the assert from standard library
#define PHYSIKA_ASSERT(x) assert(x)

#endif//PHYSIKA_CORE_UTILITIES_PHYSIKA_ASSERT_H_
