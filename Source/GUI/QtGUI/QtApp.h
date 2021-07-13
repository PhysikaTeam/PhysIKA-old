#pragma once
#include <memory>
#include "GUI/AppBase.h"

#include <QApplication>

namespace PhysIKA {

class PMainWindow;

class QtApp : public AppBase
{
public:
    QtApp(int argc = 0, char** argv = NULL);
    ~QtApp();

    void createWindow(int width, int height) override;
    void mainLoop() override;

    // add by HNU
    std::shared_ptr<PMainWindow> getMainWindow() const;

private:
    std::shared_ptr<QApplication> m_app;

    std::shared_ptr<PMainWindow> m_mainWindow;
};

}  // namespace PhysIKA