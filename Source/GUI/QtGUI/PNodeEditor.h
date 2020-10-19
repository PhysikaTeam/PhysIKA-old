#pragma once

#include <QWidget>

namespace PhysIKA
{
	class QtNodeWidget;

	class PNodeEditor :
		public QWidget
	{
		Q_OBJECT
	public:
		PNodeEditor(QtNodeWidget* node_widget);


	private:
		QtNodeWidget* node_widget = nullptr;
	};

}

