#pragma once
#include <memory>
#include "Physika_GUI/AppBase.h"

class QApplication;
class QtWindow;

namespace Physika {

    class QtApp : public AppBase
    {
    public:
        QtApp(int argc = 0, char **argv = NULL);
        ~QtApp();

        void createWindow(int width, int height) override;
        void mainLoop() override;

    private:
        std::shared_ptr<QApplication> m_app;
        std::shared_ptr<QtWindow> m_mainWindow;
    };

}