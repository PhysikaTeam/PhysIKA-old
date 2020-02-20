#ifndef PMODULELISTWIDGET_H
#define PMODULELISTWIDGET_H

#include <QListWidget>

namespace PhysIKA
{
	class Module;
	class Node;

	class PModuleListItem : public QListWidgetItem
	{
	public:
		PModuleListItem(Module* module, QListWidget *listview = nullptr);

	private:
		Module* m_module;
	};

	class PModuleListWidget : public QListWidget
	{
		Q_OBJECT

	public:
		PModuleListWidget();

	public slots:
		void updateModule(Node* node);
	};
}

#endif // PMODULELISTWIDGET_H
