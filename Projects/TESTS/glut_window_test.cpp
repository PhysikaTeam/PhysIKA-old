/*
 * @file glut_window_test.cpp 
 * @brief Test GlutWindow of Physika.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_GUI/Glut_Window/glut_window.h"
using namespace std;
using Physika::GlutWindow;

void displayFunction()
{
    cout<<"Custom display function\n";
}

int main()
{
    GlutWindow window;
    cout<<"Window name: "<<window.name()<<"\n";
    cout<<"Window size: "<<window.width()<<"x"<<window.height()<<"\n";
    window.setDisplayFunction(displayFunction);
    window.createWindow();
    return 0;
}
