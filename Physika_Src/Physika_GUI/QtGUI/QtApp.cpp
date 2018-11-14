#include <QMap>
#include <QDebug>
#include "QtApp.h"
#include "QtWindow.h"
#include "Physika_Render/OpenGLContext.h"

QtApp::QtApp(int argc, char **argv)
{
	m_mainWindow = nullptr;
	m_app = std::make_shared<QApplication>(argc, argv);
}

QtApp::~QtApp()
{

}

void QtApp::createWindow(int width, int height)
{
	m_mainWindow = std::make_shared<QtWindow>();
	m_mainWindow->resize(1024, 768);
}

void QtApp::mainLoop()
{
	OpenGLContext::getInstance().initialize();
	SceneGraph::getInstance().initialize();

	m_mainWindow->show();
	m_app->exec();
}
