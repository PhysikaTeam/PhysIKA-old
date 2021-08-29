#pragma once

#include "Framework/Framework/ModuleVisual.h"

class vtkActor;
class vtkPolyDataMapper;
class PVTKPointSetSource;

namespace PhysIKA {
class PVTKPointSetRender : public VisualModule
{
    DECLARE_CLASS(PVTKPointSetRender)
public:
    PVTKPointSetRender();
    virtual ~PVTKPointSetRender();

    vtkActor* getVTKActor();

protected:
    bool initializeImpl() override;

    void updateRenderingContext() override;

private:
    vtkActor*           m_actor;
    vtkPolyDataMapper*  mapper;
    PVTKPointSetSource* pointsetSource;
};

}  // namespace PhysIKA