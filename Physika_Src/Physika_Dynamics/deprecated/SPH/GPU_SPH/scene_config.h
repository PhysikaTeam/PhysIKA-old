/*
 * @file scene_config.h
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


#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_SCENE_CONFIG_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_SCENE_CONFIG_H_


#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

class SceneConfig 
{
public:

    static void reload(std::string file);
    static void reload();
    static void preComputing();

public:
    static float timestep;
    static float viscosity;
    static float incompressibility;
    static float surfacetension;
    static float density;
    static float gravity;
    static float samplingdistance;
    static float smoothinglength;
    static bool ghosts;

    static int width;
    static int height;

    static float controlviscosity;
    static int dumpimageevery;
    static int computesurfaceevery;
    static int fastmarchingevery;
    static int dumppovrayevery;

    static float totaltime;
    static bool offline;

    static float tangentialfriction;
    static float normalfriction;
    static float rotation_angle;

    static std::string scene;
    static std::string constraint;
    
    static Vector3f rotation_axis;
    static Vector3f rotation_center;
    static Vector3f scenelowerbound;
    static Vector3f sceneupperbound;
    static bool multires;
    static bool vorticity;

    static int dimX;
    static int dimY;
    static int dimZ;

    static float diffusion;

    static std::string _file;
    static std::string pov;
    static std::string rendered;

    static int initiallevel;
    static float xsph;

    static float alpha;
    static float beta;
    static float gamma;

    static std::string restorefile;

    static float controlvolume;
};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_SCENE_CONFIG_H_