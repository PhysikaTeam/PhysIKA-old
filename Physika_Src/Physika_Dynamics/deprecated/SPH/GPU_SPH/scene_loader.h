/*
 * @file scene_loader.h
 * @Brief class SceneLoader
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


#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_SCENE_LOADER_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_SCENE_LOADER_H_

#include <string>
#include <vector_functions.h>

#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph.h"
#include "Physika_Dynamics/SPH/GPU_SPH/scene_config.h"


namespace Physika{

class SceneLoader
{
public:

    static CudaPBDSPH* Load(std::string name)
    {
        SceneConfig::reload("config3d.txt");

        if (name.compare("Ball") == 0)
        {
            Settings settings;

            const float radius = 0.0125f;
            const float restDistance = 0.005f;
            Ball* db = new Ball(make_float3(0.65f, 0.86f, 0.5f), make_float3(-0.65f, 0.86f, 0.5f), 0.12f);
            db->build(restDistance);

            float3 lower = make_float3(0.0f, 0.1f, 0.0f);

            settings.smoothingLength = radius;
            settings.samplingDistance = restDistance;
            settings.iterNum = 4;
            settings.pNum = db->size();
            settings.gravity = make_float3(0, -9.8f, 0);
            settings.upBound = make_float3(1.0f, 1.0f, 1.0f);
            settings.lowBound = make_float3(0.0f, 0.0f, 0.0f);

            settings.nbrMaxSize = 50;
            settings.mass = 1.0f;

            settings.restDensity = 1000.0f;

            settings.surfaceTension = SceneConfig::surfacetension;
            settings.viscosity = SceneConfig::viscosity;

            settings.normalFriction = SceneConfig::normalfriction;
            settings.tangentialFriction = SceneConfig::tangentialfriction;

            CudaPBDSPH* sim = new CudaPBDSPH(settings);
            sim->addModel(db);

            sim->initialize("");
            return sim;
        }
        return NULL;
    }
};



} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_SCENE_LOADER_H_