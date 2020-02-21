/*=========================================================================

  Program:   VTK-based Visualization Widget
  Module:    VTKOpenGLWidget.h

  Copyright (c) Xiaowei He
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#ifndef VTKOPENGLWIDGET_H
#define VTKOPENGLWIDGET_H

#include <QWidget>

//VTK
#include <vtkSmartPointer.h>

QT_FORWARD_DECLARE_CLASS(QGridLayout)

class vtkRenderer;
class vtkOrientationMarkerWidget;
class QVTKOpenGLWidget;

namespace PhysIKA
{
	class PVTKOpenGLWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit PVTKOpenGLWidget(QWidget *parent = nullptr);
		~PVTKOpenGLWidget();

	signals:

	public slots:
		void showAxisWidget();

	public:
		QGridLayout*		m_MainLayout;

		vtkRenderer*		m_renderer;
		QVTKOpenGLWidget*						m_OpenGLWidget;
		vtkOrientationMarkerWidget*				m_axisWidget;
	};

}

#endif // VTKOPENGLWIDGET_H
