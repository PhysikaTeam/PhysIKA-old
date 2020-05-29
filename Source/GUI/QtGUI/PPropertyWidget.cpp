#include "PPropertyWidget.h"
#include "Framework/Module.h"
#include "Framework/Node.h"

#include <QGroupBox>
#include <QLabel>
#include <QCheckBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>

namespace PhysIKA
{
	QBoolFieldWidget::QBoolFieldWidget(Field* field)
		: QGroupBox()
	{
		m_field = field;

		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(160, 18);
		name->setText(QString::fromStdString(field->getObjectName()));
		QCheckBox* checkbox = new QCheckBox();
		checkbox->setFixedSize(40, 18);
		layout->addWidget(name, 0, 0);
		layout->addWidget(checkbox, 0, 1, Qt::AlignRight);

		connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(changeValue(int)));
	}


	void QBoolFieldWidget::changeValue(int status)
	{
		VarField<bool>* f = TypeInfo::CastPointerDown<VarField<bool>>(m_field);
		if (f == nullptr)
		{
			return;
		}

		if (status == Qt::Checked)
		{
			f->setValue(true);
		}
		else if (status == Qt::PartiallyChecked)
		{
			//m_pLabel->setText("PartiallyChecked");
		}
		else
		{
			f->setValue(false);
			//m_pLabel->setText("Unchecked");
		}
	}

	QIntegerFieldWidget::QIntegerFieldWidget(Field* field)
		: QGroupBox()
	{
		m_field = field;

		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(QString::fromStdString(field->getObjectName()));

		QSpinBox* spinner = new QSpinBox;
		spinner->setFixedSize(120, 18);

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner, 0, 1, Qt::AlignRight);

		this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
	}

	void QIntegerFieldWidget::changeValue(int value)
	{

	}

	//QWidget-->QVBoxLayout-->QScrollArea-->QWidget-->QGridLayout
	PPropertyWidget::PPropertyWidget(QWidget *parent)
		: QWidget(parent)
		, m_main_layout()
	{
		m_main_layout = new QVBoxLayout;
		m_scroll_area = new QScrollArea;

		m_main_layout->setContentsMargins(0, 0, 0, 0);
		m_main_layout->setSpacing(0);
		m_main_layout->addWidget(m_scroll_area);

		m_scroll_area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		m_scroll_area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		m_scroll_area->setWidgetResizable(true);

		m_scroll_layout = new QGridLayout;
		m_scroll_layout->setAlignment(Qt::AlignLeft | Qt::AlignTop);

		QWidget * m_scroll_widget = new QWidget;
		m_scroll_widget->setLayout(m_scroll_layout);
		
		m_scroll_area->setWidget(m_scroll_widget);


		setMinimumWidth(250);
		setLayout(m_main_layout);
	}

	PPropertyWidget::~PPropertyWidget()
	{
		m_widgets.clear();
	}

	QSize PPropertyWidget::sizeHint() const
	{
		return QSize(20, 20);
	}

	QWidget* PPropertyWidget::addWidget(QWidget* widget)
	{
		m_scroll_layout->addWidget(widget);
		m_widgets.push_back(widget);

		return widget;
	}

	void PPropertyWidget::removeAllWidgets()
	{
		//TODO: check whether m_widgets[i] should be explicitly deleted
		for (int i = 0; i < m_widgets.size(); i++)
		{
			m_scroll_layout->removeWidget(m_widgets[i]);
			delete m_widgets[i];
		}
		m_widgets.clear();
	}

	void PPropertyWidget::showProperty(Module* module)
	{
//		clear();

		updateContext(module);
	}

	void PPropertyWidget::showProperty(Node* node)
	{
//		clear();

		updateContext(node);
	}

	void PPropertyWidget::updateContext(Base* base)
	{
		if (base == nullptr)
		{
			return;
		}

		this->removeAllWidgets();

		std::vector<Field*>& fields = base->getAllFields();

		for each (Field* var in fields)
		{
			if (var != nullptr)
			{
				if (var->getClassName() == std::string("Variable"))
				{
					addScalarFieldWidget(var);
				}
				else if (var->getClassName() == std::string("ArrayBuffer"))
				{
				}
				//addItem(new QListWidgetItem(var->getObjectName().c_str(), this));
			}
		}
	}

	void PPropertyWidget::addScalarFieldWidget(Field* field)
	{
		std::string template_name = field->getTemplateName();
		if (template_name == std::string(typeid(bool).name()))
		{
			this->addWidget(new QBoolFieldWidget(field));
		}
		else if (template_name == std::string(typeid(int).name()))
		{
			this->addWidget(new QIntegerFieldWidget(field));
//			this->addWidget(new QIntegerFieldWidget(new VarField<int>()));
		}
	}

	void PPropertyWidget::addArrayFieldWidget(Field* field)
	{

	}

}
