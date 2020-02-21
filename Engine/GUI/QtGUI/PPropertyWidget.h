#ifndef QNODEPROPERTYWIDGET_H
#define QNODEPROPERTYWIDGET_H

#include <QToolBox>
#include <QListWidget>

namespace PhysIKA
{
	class Base;
	class Node;
	class Module;

	class PPropertyWidget : public QListWidget
	{
		Q_OBJECT
	public:
		explicit PPropertyWidget(QWidget *parent = nullptr);

//		void clear();

	//signals:

	public slots:
		void showProperty(Module* module);
		void showProperty(Node* node);

	private:
		void updateContext(Base* base);
	};

}

#endif // QNODEPROPERTYWIDGET_H
