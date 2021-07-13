#pragma once
#include "vtkPolyDataAlgorithm.h"  // For export macro
#include "Framework/Topology/TriangleSet.h"

class vtkDataSet;
class vtkPolyData;

class PHYSIKA_EXPORT PVTKPolyDataSource : public vtkPolyDataAlgorithm
{
public:
    static PVTKPolyDataSource* New();
    vtkTypeMacro(PVTKPolyDataSource, vtkPolyDataAlgorithm);

    void setData(std::shared_ptr<PhysIKA::TriangleSet<PhysIKA::DataType3f>> data)
    {
        m_tri_set = data;
    }

protected:
    PVTKPolyDataSource();
    ~PVTKPolyDataSource() override;

    /**
    * This is called by the superclass.
    * This is the method you should override.
    */
    int RequestData(vtkInformation*        request,
                    vtkInformationVector** inputVector,
                    vtkInformationVector*  outputVector) override;

    int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
    PVTKPolyDataSource(const PVTKPolyDataSource&) = delete;
    void operator=(const PVTKPolyDataSource&) = delete;

    std::shared_ptr<PhysIKA::TriangleSet<PhysIKA::DataType3f>> m_tri_set;
};
