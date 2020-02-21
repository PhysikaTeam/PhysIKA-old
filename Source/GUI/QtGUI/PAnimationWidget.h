#ifndef PANIMATIONWIDGET_H
#define PANIMATIONWIDGET_H

#include <QWidget>

QT_FORWARD_DECLARE_CLASS(QPushButton)

namespace PhysIKA
{
	class PAnimationWidget : public QWidget
	{
		Q_OBJECT

	public:
		explicit PAnimationWidget(QWidget *parent = nullptr);

	signals:

	public slots:

	public:
		QPushButton*	m_startSim;
		QPushButton*	m_resetSim;
	};
}

#endif // PANIMATIONWIDGET_H
