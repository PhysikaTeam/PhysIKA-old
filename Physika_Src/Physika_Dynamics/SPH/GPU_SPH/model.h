/*
 * @file model.h
 * @Brief class Model
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


#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_MODEL_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_MODEL_H_

#include <string>
#include <vector>
#include "vector_types.h"
#include "vector_functions.h"

namespace Physika{

class Model 
{
public:
    Model(std::string name) : name(name) {}
    virtual void build(float s) = 0;
    int size() { return positions.size(); }

public:
    std::string name;

    std::vector<float3> positions;
    std::vector<float3> velocities;
};

class Ball : public Model 
{

public:
    Ball(float3 center1, float3 center2, float R) : Model("Square") { m_center1 = center1; m_center2 = center2; m_R = R; }

    virtual void build(float s)
    {
        for (float x = m_center1.x - m_R; x < m_center1.x + m_R; x += s) {
            for (float y = m_center1.y - m_R; y < m_center1.y + m_R; y += s) {
                for (float z = m_center1.z - m_R; z < m_center1.z + m_R; z += s) {
                    float dist = (x - m_center1.x)*(x - m_center1.x) + (y - m_center1.y)*(y - m_center1.y) + (z - m_center1.z)*(z - m_center1.z);
                    if (sqrt(dist) < m_R)
                    {
                        float3 pos = make_float3(float(x), float(y), float(z));
                        positions.push_back(pos);
                        velocities.push_back(make_float3(-0.5f, -0.0f, -0.2f));
                    }
                }
            }
        }

        /*
        for (float x = m_center2.x - m_R; x < m_center2.x + m_R; x += s) {
            for (float y = m_center2.y - m_R; y < m_center2.y + m_R; y += s) {
             	for (float z = m_center2.z - m_R; z < m_center2.z + m_R; z += s) {
             		float dist = (x - m_center2.x)*(x - m_center2.x) + (y - m_center2.y)*(y - m_center2.y) + (z - m_center2.z)*(z - m_center2.z);
             		if (sqrt(dist) < m_R)
             		{
             			float3 pos = make_float3(float(x), float(y), float(z));
             			positions.push_back(pos);
             			velocities.push_back(make_float3(0.5f, -0.0f, -0.2f));
             		}
             	}
            }
        }
        */

    }

private:
    float3 m_center1;
    float3 m_center2;
    float m_R;
};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_MODEL_H_