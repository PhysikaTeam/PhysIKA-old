#include "PVTKPointSetSource.h"

#include "Core/Utility/Function1Pt.h"

#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkPointSet.h"
#include "vtkPoints.h"

vtkStandardNewMacro(PVTKPointSetSource);

//----------------------------------------------------------------------------
PVTKPointSetSource::PVTKPointSetSource()
{
	this->SetNumberOfInputPorts(0);
	this->SetNumberOfOutputPorts(1);
}

//----------------------------------------------------------------------------
PVTKPointSetSource::~PVTKPointSetSource() = default;

int PVTKPointSetSource::RequestData(
	vtkInformation* vtkNotUsed(request),
	vtkInformationVector** vtkNotUsed(inputVector),
	vtkInformationVector* outputVector)
{

	if (m_point_set == nullptr)
	{
		return 0;
	}

	// get the info object
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	// get the output
	vtkPointSet *output = vtkPointSet::SafeDownCast(
		outInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkPoints* pts = vtkPoints::New();

	auto device_pts = m_point_set->getPoints();

	int num_of_points = device_pts.size();
	PhysIKA::HostArray<PhysIKA::Vector3f> host_pts;
	host_pts.resize(num_of_points);
	PhysIKA::Function1Pt::copy(host_pts, device_pts);

	pts->Allocate(num_of_points);

	for(int i = 0; i < num_of_points; i++)
	{
		pts->InsertPoint(i, host_pts[i][0], host_pts[i][1], host_pts[i][2]);
	}

	pts->Squeeze();
	output->SetPoints(pts);
	pts->Delete();

	host_pts.release();

	return 1;
}


