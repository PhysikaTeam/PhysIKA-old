#include <QMap>
#include <QDebug>
#include "QtApp.h"
#include "PMainWindow.h"
#include "Framework/Log.h"
//#include "Rendering/OpenGLContext.h"

namespace PhysIKA {
	QtApp::QtApp(int argc, char **argv)
	{
		m_mainWindow = nullptr;
		m_app = std::make_shared<QApplication>(argc, argv);

		//To resolver the error "Cannot queue arguments of type of Log::Message" for multi-thread applications
		qRegisterMetaType<Log::Message>("Log::Message");
	}

	QtApp::~QtApp()
	{

	}

	void QtApp::createWindow(int width, int height)
	{
		m_mainWindow = std::make_shared<PMainWindow>();
		m_mainWindow->resize(1024, 768);
	}

	void QtApp::mainLoop()
	{
		//         OpenGLContext::getInstance().initialize();
		SceneGraph::getInstance().initialize();

		m_mainWindow->show();
		m_app->exec();
	}

	// add by HNU
	std::shared_ptr<PMainWindow> QtApp::getMainWindow() const
	{
		return m_mainWindow;
	}
}
