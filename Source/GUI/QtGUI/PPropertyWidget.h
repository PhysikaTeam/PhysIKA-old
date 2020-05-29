#ifndef QNODEPROPERTYWIDGET_H
#define QNODEPROPERTYWIDGET_H

#include <QToolBox>
#include <QWidget>
#include <QGroupBox>
#include <QScrollArea>
#include <QGridLayout>
#include <QVBoxLayout>

#include <vector>

namespace PhysIKA
{
	class Base;
	class Node;
	class Module;
	class Field;

	class QBoolFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QBoolFieldWidget(Field* field);
		~QBoolFieldWidget() {};

	public slots:
		void changeValue(int status);

	private:
		Field* m_field = nullptr;
	};

	class QIntegerFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QIntegerFieldWidget(Field* field);
		~QIntegerFieldWidget() {};

	public slots:
		void changeValue(int);

	private:
		Field* m_field = nullptr;
	};


	class PPropertyWidget : public QWidget
	{
		Q_OBJECT
	public:
		explicit PPropertyWidget(QWidget *parent = nullptr);
		~PPropertyWidget();

		virtual QSize sizeHint() const;

//		void clear();

	//signals:
		QWidget* addWidget(QWidget* widget);
		void removeAllWidgets();


	public slots:
		void showProperty(Module* module);
		void showProperty(Node* node);

	private:
		void updateContext(Base* base);

		void addScalarFieldWidget(Field* field);
		void addArrayFieldWidget(Field* field);

		QVBoxLayout* m_main_layout;
		QScrollArea* m_scroll_area;
		QWidget * m_scroll_widget;
		QGridLayout* m_scroll_layout;

		std::vector<QWidget*> m_widgets;
	};

}

#endif // QNODEPROPERTYWIDGET_H
