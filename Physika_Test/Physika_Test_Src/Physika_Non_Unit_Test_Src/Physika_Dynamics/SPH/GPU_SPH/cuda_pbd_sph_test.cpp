/*
* @file cuda_pbd_sph_test.cpp
* @brief Test CudaPBDSPH of Physika.
* @author WeiChen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include "cuda_runtime.h"
#include "Physika_Core/Utilities/cuda_utilities.h"

#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"

#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph.h"
#include "Physika_Dynamics/SPH/GPU_SPH/scene_loader.h"
#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph_plugin_render.h"

using namespace std;
using namespace Physika;

int main()
{

    CudaPBDSPH * cuda_pbd_sph = SceneLoader::Load("Ball");

    
    GlutWindow glut_window("cuda_pbd_sph_test", 1280, 720);
    glut_window.enableEventMode();

    CudaPBDSPHPluginRender render_plugin;
    render_plugin.setWindow(&glut_window);
    cuda_pbd_sph->addPlugin(&render_plugin);


    cout << "Window name: " << glut_window.name() << "\n";
    cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";

    glut_window.setCameraPosition(Vector<double, 3>(0, 0.1, 5));
    glut_window.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
    glut_window.setCameraNearClip(0.01);
    glut_window.setCameraFarClip(1.0e3);
    glut_window.createWindow();


    cuda_pbd_sph->run();
    

    return 0;
}