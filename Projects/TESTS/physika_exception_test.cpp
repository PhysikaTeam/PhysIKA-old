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
#include "Physika_Core/Vectors/vector_1d.h"
#include "Physika_Core/Utilities/physika_exception.h"
using Physika::PhysikaException;
using Physika::Vector;
using namespace std;

void testFunction()
{
    Vector<float,1> vector_1d(0);
    std::cout<<vector_1d[1]<<"\n";
}

int main()
{
    testFunction();
    //throw PhysikaException("Test PhysikaException");
    return 0;
}
