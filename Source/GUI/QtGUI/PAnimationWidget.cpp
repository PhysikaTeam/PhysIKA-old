#include "PAnimationWidget.h"


#include <QGridLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QSlider>
#include <QScrollBar>

namespace PhysIKA
{
	PAnimationWidget::PAnimationWidget(QWidget *parent) : 
		QWidget(parent),
		m_startSim(nullptr),
		m_resetSim(nullptr)
	{
		QHBoxLayout* layout = new QHBoxLayout();
		setLayout(layout);

		QGridLayout* frameLayout	= new QGridLayout();
		QLineEdit* startFrame		= new QLineEdit();
		QLineEdit* endFrame			= new QLineEdit();
		QScrollBar*	scrollBar		= new QScrollBar(Qt::Horizontal, this);

		startFrame->setFixedSize(30, 25);
		endFrame->setFixedSize(30, 25);
		scrollBar->setFixedHeight(25);
		frameLayout->addWidget(startFrame, 0, 0);
		frameLayout->addWidget(scrollBar, 0, 1);
		frameLayout->addWidget(endFrame, 0, 2);

		QGridLayout* operationLayout = new QGridLayout();

		m_startSim = new QPushButton("Start");
		m_resetSim = new QPushButton("Reset");
		operationLayout->addWidget(m_startSim, 0, 0);
		operationLayout->addWidget(m_resetSim, 0, 1);

		layout->addLayout(frameLayout, 10);
		layout->addLayout(operationLayout, 1);
	}

}
