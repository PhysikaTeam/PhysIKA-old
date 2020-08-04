/*=========================================================================

  Program:   VTK-based Visualization Widget
  Module:    VTKOpenGLWidget.h

  Copyright (c) Xiaowei He
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#ifndef PFLOWSCENEWIDGET_H
#define PFLOWSCENEWIDGET_H

#include <QWidget>


QT_FORWARD_DECLARE_CLASS(QGridLayout)

namespace PhysIKA
{
	class PFlowSceneWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit PFlowSceneWidget(QWidget *parent = nullptr);
		~PFlowSceneWidget();

		//void addActor(vtkActor *actor);

	signals:

	public:
		QGridLayout*		m_MainLayout;

	};

}

#endif // VTKOPENGLWIDGET_H
