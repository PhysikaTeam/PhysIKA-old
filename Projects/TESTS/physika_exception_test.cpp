/*
 * @file physika_exception_test.cpp
 * @brief Test Physika exception.
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

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"
using Physika::PhysikaException;
using namespace std;

int main()
{
    throw PhysikaException("Test PhysikaException");
    return 0;
}
