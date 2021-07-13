#pragma once
#include "vtkPolyDataAlgorithm.h"  // For export macro
#include "Framework/Topology/PointSet.h"

class vtkDataSet;
class vtkPointSet;

class PHYSIKA_EXPORT PVTKPointSetSource : public vtkPolyDataAlgorithm
{
public:
    static PVTKPointSetSource* New();
    vtkTypeMacro(PVTKPointSetSource, vtkPolyDataAlgorithm);

    void setData(std::shared_ptr<PhysIKA::PointSet<PhysIKA::DataType3f>> data)
    {
        m_point_set = data;
    }

protected:
    PVTKPointSetSource();
    ~PVTKPointSetSource() override;

    /**
    * This is called by the superclass.
    * This is the method you should override.
    */
    int RequestData(vtkInformation*        request,
                    vtkInformationVector** inputVector,
                    vtkInformationVector*  outputVector) override;

private:
    PVTKPointSetSource(const PVTKPointSetSource&) = delete;
    void operator=(const PVTKPointSetSource&) = delete;

    std::shared_ptr<PhysIKA::PointSet<PhysIKA::DataType3f>> m_point_set;
};
