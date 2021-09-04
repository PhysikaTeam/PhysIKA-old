/**
 * @author     : syby119 (syby119@126.com)
 * @date       : 2021-05-30
 * @version    : 1.0
 * @description: extended visual module to display mesh in wireframe
 */

#pragma once

#include "Rendering/SurfaceMeshRender.h"

class SurfaceLineRender : public PhysIKA::SurfaceMeshRender
{
    void display() override
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        SurfaceMeshRender::display();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
};
