#include "PVTKOpenGLWidget.h"

//VTK
#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkRenderWindowInteractor.h>
#include <QVTKOpenGLWidget.h>

//QT
#include <QGridLayout>

namespace PhysIKA
{
	PVTKOpenGLWidget::PVTKOpenGLWidget(QWidget *parent) :
		QWidget(parent)
	{
		m_MainLayout = new QGridLayout();
		this->setLayout(m_MainLayout);

		m_OpenGLWidget = new QVTKOpenGLWidget();
		m_MainLayout->addWidget(m_OpenGLWidget, 0, 0);

		m_renderer = vtkRenderer::New();

		vtkSmartPointer<vtkRenderWindow> renderWindow = m_OpenGLWidget->GetRenderWindow();

		renderWindow->AddRenderer(m_renderer);

		vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
		sphereSource->SetCenter(0.0, 0.0, 0.0);
		sphereSource->SetRadius(1.0);
		sphereSource->Update();

		vtkPolyData* polydata = sphereSource->GetOutput();

		// Create a mapper
		vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputData(polydata);

		// Create an actor
		vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
		actor->SetMapper(mapper);

		// Add the actors to the scene
		m_renderer->AddActor(actor);
		m_renderer->SetBackground(.2, .3, .4);


		vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();

		m_axisWidget = vtkOrientationMarkerWidget::New();
		m_axisWidget->SetOutlineColor(0.9300, 0.5700, 0.1300);
		m_axisWidget->SetOrientationMarker(axes);
		m_axisWidget->SetCurrentRenderer(m_renderer);
		m_axisWidget->SetInteractor(m_OpenGLWidget->GetInteractor());
		m_axisWidget->SetViewport(0.0, 0.0, 0.2, 0.2);
		m_axisWidget->SetEnabled(1);
		m_axisWidget->InteractiveOn();

		m_renderer->ResetCamera();
	}

	PVTKOpenGLWidget::~PVTKOpenGLWidget()
	{
		delete m_MainLayout;
		m_MainLayout = nullptr;

		m_renderer->Delete();
		m_renderer = nullptr;

		m_axisWidget->Delete();
		m_axisWidget = nullptr;
	}

	void PVTKOpenGLWidget::showAxisWidget()
	{

	}

}