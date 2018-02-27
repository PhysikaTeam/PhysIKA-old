/*
 * @file scene_config.cpp
 * @Brief class SceneConfig
 * @author Wei Chen
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <string>
#include <fstream>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/SPH/GPU_SPH/scene_config.h"

namespace Physika {

    float SceneConfig::timestep = 0.001f;
    float SceneConfig::viscosity = 10000.0f;
    float SceneConfig::surfacetension = 0.1f;
    float SceneConfig::density = 1000.0f;
    float SceneConfig::gravity = -9.81f;
    float SceneConfig::incompressibility = 800000.0f;
    float SceneConfig::samplingdistance = 0.01f;
    float SceneConfig::smoothinglength = 2.5f;
    bool  SceneConfig::ghosts = 1;
    int   SceneConfig::width = 512;
    int   SceneConfig::height = 512;
    float SceneConfig::controlviscosity = 1;
    int   SceneConfig::dumpimageevery = 1;
    int   SceneConfig::dumppovrayevery = 1;
    int   SceneConfig::computesurfaceevery = 1;
    int   SceneConfig::fastmarchingevery = 1;
    float SceneConfig::totaltime = 5;
    bool  SceneConfig::offline = 0;
    float SceneConfig::tangentialfriction = 0;
    float SceneConfig::normalfriction = 0;
    float SceneConfig::rotation_angle = 0;

    std::string SceneConfig::scene = "";
    std::string SceneConfig::constraint = "";
    
    Vector3f SceneConfig::rotation_axis;
    Vector3f SceneConfig::rotation_center;
    Vector3f SceneConfig::scenelowerbound;
    Vector3f SceneConfig::sceneupperbound;

    bool  SceneConfig::multires = 0;
    bool  SceneConfig::vorticity = 0;
    int   SceneConfig::initiallevel = 1;
    float SceneConfig::xsph = 0.25f;
    float SceneConfig::alpha = 4;
    float SceneConfig::beta = 6;
    float SceneConfig::gamma = 1.2f;

    std::string SceneConfig::pov = "pov";
    std::string SceneConfig::rendered = "rendered";
    std::string SceneConfig::restorefile = "norestore";

    float SceneConfig::controlvolume = 1.0f;
    int   SceneConfig::dimX = 0;
    int   SceneConfig::dimY = 0;
    int   SceneConfig::dimZ = 0;
    float SceneConfig::diffusion = 0.0f;

    std::string SceneConfig::_file = "";

    void SceneConfig::reload() 
    {
        reload(_file);
    }

    void SceneConfig::reload(std::string file) 
    {
        _file = file;
        std::ifstream input(file.c_str(), std::ios::in);
        std::string param;
        while (input >> param) 
        {
            if      (param == std::string("timestep"))            input >> timestep;
            else if (param == std::string("viscosity"))           input >> viscosity;
            else if (param == std::string("incompressibility"))   input >> incompressibility;
            else if (param == std::string("surfacetension"))      input >> surfacetension;
            else if (param == std::string("density"))             input >> density;
            else if (param == std::string("gravity"))             input >> gravity;
            else if (param == std::string("samplingdistance"))    input >> samplingdistance;
            else if (param == std::string("smoothinglength"))     input >> smoothinglength;
            else if (param == std::string("ghosts"))              input >> ghosts;
            else if (param == std::string("width"))               input >> width;
            else if (param == std::string("height"))              input >> height;
            else if (param == std::string("controlviscosity"))    input >> controlviscosity;
            else if (param == std::string("dumpimageevery"))      input >> dumpimageevery;
            else if (param == std::string("dumppovrayevery"))     input >> dumppovrayevery;
            else if (param == std::string("computesurfaceevery")) input >> computesurfaceevery;
            else if (param == std::string("fastmarchingevery"))   input >> fastmarchingevery;
            else if (param == std::string("totaltime"))           input >> totaltime;
            else if (param == std::string("offline"))             input >> offline;
            else if (param == std::string("tangentialfriction"))  input >> tangentialfriction;
            else if (param == std::string("normalfriction"))      input >> normalfriction;
            else if (param == std::string("scene"))               input >> scene;
            else if (param == std::string("constraint"))          input >> constraint;
            else if (param == std::string("rotation_angle"))      input >> rotation_angle;
            else if (param == std::string("vorticity"))           input >> vorticity;
            else if (param == std::string("restorefile"))         input >> restorefile;
            else if (param == std::string("dimX"))                input >> dimX;
            else if (param == std::string("dimY"))                input >> dimY;
            else if (param == std::string("dimZ"))                input >> dimZ;
            else if (param == std::string("diffusion"))           input >> diffusion;
            else if (param == std::string("multires"))            input >> multires;
            else if (param == std::string("initiallevel"))        input >> initiallevel;
            else if (param == std::string("xsph"))                input >> xsph;
            else if (param == std::string("alpha"))               input >> alpha;
            else if (param == std::string("beta"))                input >> beta;
            else if (param == std::string("gamma"))               input >> gamma;
            else if (param == "pov")                              input >> pov;
            else if (param == "rendered")                         input >> rendered;
            else if (param == std::string("rotation_axis")) 
            {
                input >> rotation_axis[0];
                input >> rotation_axis[1];
                input >> rotation_axis[2];
            }
            else if (param == std::string("rotation_center")) 
            {
                input >> rotation_center[0];
                input >> rotation_center[1];
                input >> rotation_center[2];
            }
            else if (param == std::string("scenelowerbound")) 
            {
                input >> scenelowerbound[0];
                input >> scenelowerbound[1];
                input >> scenelowerbound[2];
            }
            else if (param == std::string("sceneupperbound")) 
            {
                input >> sceneupperbound[0];
                input >> sceneupperbound[1];
                input >> sceneupperbound[2];
            }
        }

        preComputing();
    }

void SceneConfig::preComputing()
{
    float R = samplingdistance*smoothinglength;
    #ifndef KERNEL_3D
    controlvolume = PI*R*R;
    #else
    controlvolume = 4.0 / 3.0*PI*R*R*R;
    #endif
}


}// end of namespace  Physika