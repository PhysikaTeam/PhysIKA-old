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
