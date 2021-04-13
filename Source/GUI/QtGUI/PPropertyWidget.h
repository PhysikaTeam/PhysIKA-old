#ifndef QNODEPROPERTYWIDGET_H
#define QNODEPROPERTYWIDGET_H

#include <QToolBox>
#include <QWidget>
#include <QGroupBox>
#include <QLineedit>
#include <QScrollArea>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QComboBox>
#include <QMessageBox>
#include <QProcess>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QDebug>

#include "Nodes/QtBlock.h"

#include <vector>

namespace PhysIKA
{
	class Base;
	class Node;
	class Module;
	class Field;
	class QDoubleSpinner;
	class PVTKOpenGLWidget;

	// add by HNU
	class vtk2ObjThread :public QObject
	{
		Q_OBJECT

	public:
		void doWork(QString);

	Q_SIGNALS:
		void resultReady(QString);
	};

	// 20210315 add by HNU
	class QTextFieldWidget:public QGroupBox
	{
		Q_OBJECT
	public:
		 QTextFieldWidget(Field* field);
		~QTextFieldWidget() {};

		// files conversion *.k => *.obj
		static QString k2Obj(const QString);
		static QString vtk2Obj(const QString);

		int multiK2Obj(QString);
		int startAnimating(QString);

	Q_SIGNALS:
		void loadFileSignal(const QString);

		void multiThread(QString);

	public slots :
		void initSolverUi(QGridLayout*);
		void initOutputUi(QGridLayout*);
		void btnClicked();

		QString updatePreview();


		// Path saver define
		static QString* getInputPathSaver()
		{
			if (m_InputPathSaver == nullptr)
				m_InputPathSaver = new QString;
			return m_InputPathSaver;
		}

		static QString* getSolverPathSaver()
		{
			if (m_solverPathSaver == nullptr)
				m_solverPathSaver   = new QString;
			return m_solverPathSaver;
		}

		static QString* getWorkDirPathSaver()
		{
			if ( m_workDirPathSaver == nullptr)
				m_workDirPathSaver = new QString;
			return m_workDirPathSaver;
		}

		static QString* getParviewSaver()
		{
			if (m_previewSaver== nullptr)
				m_previewSaver  = new QString;
			return m_previewSaver;
		}

		void emitLoadFileSignal(const QString str)
		{
			//qDebug() << "load File Signal emit";
			emit loadFileSignal(str);
		}


	private:
		Field* m_field = nullptr;
		
		QPushButton* m_btnManager[6];

		QComboBox* m_inputCombo = new QComboBox;
		QComboBox* m_solverCombo= new QComboBox;
		QComboBox* m_workdirCombo = new QComboBox;
		QComboBox* m_outputCombo = new QComboBox;


		QLineEdit* m_numLineedit = new QLineEdit;
		QLineEdit* m_IDLineedit = new QLineEdit;
		QLineEdit* m_DTLineedit = new QLineEdit;
		QLineEdit* m_OTLineedit = new QLineEdit;
		QLineEdit* m_searchStepLineedit = new QLineEdit;

		QLineEdit* m_previewLineedit = new QLineEdit;

		QString m_Preview;


		QProcess* proCmd = new QProcess(this);
		
		void btnAddPicHelper(QPushButton*);

		static QString* m_InputPathSaver;
		static QString* m_solverPathSaver;
		static QString* m_workDirPathSaver;
		static QString* m_previewSaver;
	};



	class QBoolFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QBoolFieldWidget(Field* field);
		~QBoolFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int status);


		/* �����ؼ����ж� */
		//void on_openBtn_clicked();

	private:
		Field* m_field = nullptr;

		/* �����ؼ����ж� */
		//QLineEdit* lineedit = new QLineEdit();
	};

	class QIntegerFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QIntegerFieldWidget(Field* field);
		~QIntegerFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int);

	private:
		Field* m_field = nullptr;
	};

	class QRealFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QRealFieldWidget(Field* field);
		~QRealFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(double);

	private:
		Field* m_field = nullptr;
	};


	class QVector3FieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QVector3FieldWidget(Field* field);
		~QVector3FieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(double);

	private:
		Field* m_field = nullptr;

		QDoubleSpinner* spinner1;
		QDoubleSpinner* spinner2;
		QDoubleSpinner* spinner3;
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


		void sendQTextFieldInitSignal();

	public slots:
		void showProperty(Module* module);
		void showProperty(Node* node);

		void showBlockProperty(QtNodes::QtBlock& block);

		void updateDisplay();

		QTextFieldWidget* getTextFieldWidget();

	Q_SIGNALS:
		void QTextFieldInitSignal();

	private:
		void updateContext(Base* base);

		void addScalarFieldWidget(Field* field);
		void addArrayFieldWidget(Field* field);

		QVBoxLayout* m_main_layout;
		QScrollArea* m_scroll_area;
		QWidget * m_scroll_widget;
		QGridLayout* m_scroll_layout;

		std::vector<QWidget*> m_widgets;
		
		QTextFieldWidget* m_TextFieldWidget;
	};
}

#endif // QNODEPROPERTYWIDGET_H
