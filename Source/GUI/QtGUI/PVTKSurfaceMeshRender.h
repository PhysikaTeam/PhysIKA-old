#pragma once

#include "Framework/Framework/ModuleVisual.h"

class vtkActor;
class vtkPolyDataMapper;
class PVTKPolyDataSource;

namespace PhysIKA {
class PVTKSurfaceMeshRender : public VisualModule
{
    DECLARE_CLASS(PVTKSurfaceMeshRender)
public:
    PVTKSurfaceMeshRender();
    virtual ~PVTKSurfaceMeshRender();

    vtkActor* getVTKActor();

protected:
    bool initializeImpl() override;

    void updateRenderingContext() override;

private:
    vtkActor*           m_actor;
    vtkPolyDataMapper*  mapper;
    PVTKPolyDataSource* polydataSource;
};

}  // namespace PhysIKA