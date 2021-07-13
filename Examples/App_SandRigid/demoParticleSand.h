

#include "GUI/GlutGUI/GLApp.h"
#include "Dynamics/RigidBody/RigidBody2.h"
#include "Rendering/RigidMeshRender.h"

#include "Dynamics/RigidBody/Vehicle/PBDCar.h"

#include "demoCallbacks.h"

using namespace PhysIKA;

class DemoParticleSand : public GLApp
{

private:
    DemoParticleSand()
    {
        setKeyboardFunction(DemoParticleSand::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSand* m_instance;

public:
    static DemoParticleSand* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSand;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;

                case 'f':
                    if (m_instance->pheightSaver)
                    {
                        m_instance->pheightSaver->handle(0);
                    }
                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::shared_ptr<ParticleHeightOnZ> pheightSaver;

    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};

class DemoParticleSandRigid_Sphere : public GLApp
{

private:
    DemoParticleSandRigid_Sphere()
    {
        setKeyboardFunction(DemoParticleSandRigid_Sphere::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSandRigid_Sphere* m_instance;

public:
    static DemoParticleSandRigid_Sphere* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSandRigid_Sphere;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};

//
//class DemoParticleSandRigid_Cube :public GLApp
//{
//
//private:
//       DemoParticleSandRigid_Cube() {
//              setKeyboardFunction(DemoParticleSandRigid_Cube::demoKeyboardFunction);
//              createWindow(1024, 768);
//       }
//       static DemoParticleSandRigid_Cube* m_instance;
//
//public:
//
//       static DemoParticleSandRigid_Cube* getInstance()
//       {
//              if (m_instance == 0)
//                     m_instance = new DemoParticleSandRigid_Cube;
//              return m_instance;
//       }
//
//
//       void createScene();
//
//       void run()
//       {
//              Log::setOutput("console_log.txt");
//              Log::setLevel(Log::Info);
//              Log::sendMessage(Log::Info, "Simulation begin");
//
//              mainLoop();
//
//              Log::sendMessage(Log::Info, "Simulation end!");
//       }
//
//       static void demoKeyboardFunction(unsigned char key, int x, int y)
//       {
//
//              {
//                     if (!m_instance)
//                            return;
//                     switch (key)
//                     {
//                     case 'v':
//                            m_instance->_changeVisibility();
//
//                            break;
//                     default:
//                            GLApp::keyboardFunction(key, x, y);
//                            break;
//                     }
//              }
//       }
//
//private:
//       void _changeVisibility()
//       {
//              if (m_rigidVisible)
//              {
//                     for (int i = 0; i < m_rigids.size(); ++i)
//                     {
//                            m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
//                     }
//
//              }
//              else
//              {
//                     for (int i = 0; i < m_rigids.size(); ++i)
//                     {
//                            m_rigids[i]->addVisualModule(m_rigidRenders[i]);
//                     }
//              }
//              m_rigidVisible = !m_rigidVisible;
//       }
//
//public:
//
//
//       std::vector<RigidBody2_ptr> m_rigids;
//       std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;
//
//       bool m_rigidVisible = true;
//};

class DemoParticleSandSlop : public GLApp
{

private:
    DemoParticleSandSlop()
    {
        setKeyboardFunction(DemoParticleSandSlop::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSandSlop* m_instance;

public:
    static DemoParticleSandSlop* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSandSlop;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};

class DemoParticleSandLand : public GLApp
{

private:
    DemoParticleSandLand()
    {
        setKeyboardFunction(DemoParticleSandLand::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSandLand* m_instance;

public:
    static DemoParticleSandLand* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSandLand;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};

class DemoParticleSandLand2 : public GLApp
{

private:
    DemoParticleSandLand2()
    {
        setKeyboardFunction(DemoParticleSandLand2::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSandLand2* m_instance;

public:
    static DemoParticleSandLand2* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSandLand2;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};

class DemoParticleSandSlide : public GLApp
{

private:
    DemoParticleSandSlide()
    {
        setKeyboardFunction(DemoParticleSandSlide::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSandSlide* m_instance;

public:
    static DemoParticleSandSlide* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSandSlide;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};

class DemoParticleSandSlide2 : public GLApp
{

private:
    DemoParticleSandSlide2()
    {
        setKeyboardFunction(DemoParticleSandSlide2::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSandSlide2* m_instance;

public:
    static DemoParticleSandSlide2* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSandSlide2;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    std::shared_ptr<ParticleGenerationCallback> m_particleGenerator;

    bool m_rigidVisible = true;
};

class DemoParticleSandMultiRigid : public GLApp
{

private:
    DemoParticleSandMultiRigid()
    {
        setKeyboardFunction(DemoParticleSandMultiRigid::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSandMultiRigid* m_instance;

public:
    static DemoParticleSandMultiRigid* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSandMultiRigid;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        if (!m_instance)
            return;
        switch (key)
        {
            case 'a':
                m_instance->m_car->goLeft(0.016);
                break;
            case 'd':
                m_instance->m_car->goRight(0.016);
                break;
            case 'w':
                m_instance->m_car->forward(0.016);
                break;
            case 's':
                m_instance->m_car->backward(0.016);
                break;
            case 'v':
                m_instance->_changeVisibility();

                break;
            default:
                GLApp::keyboardFunction(key, x, y);
                break;
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::shared_ptr<PBDCar>                       m_car;
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};

class DemoParticleSandPile : public GLApp
{

private:
    DemoParticleSandPile()
    {
        setKeyboardFunction(DemoParticleSandPile::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleSandPile* m_instance;

public:
    static DemoParticleSandPile* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleSandPile;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool                                        m_rigidVisible = true;
    std::shared_ptr<ParticleGenerationCallback> m_particleGenerator;
};

class DemoParticleAvalanche : public GLApp
{

private:
    DemoParticleAvalanche()
    {
        setKeyboardFunction(DemoParticleAvalanche::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleAvalanche* m_instance;

public:
    static DemoParticleAvalanche* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleAvalanche;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};

class DemoParticleRiver : public GLApp
{

private:
    DemoParticleRiver()
    {
        setKeyboardFunction(DemoParticleRiver::demoKeyboardFunction);
        createWindow(1024, 768);
    }
    static DemoParticleRiver* m_instance;

public:
    static DemoParticleRiver* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoParticleRiver;
        return m_instance;
    }

    void createScene();

    void run()
    {
        Log::setOutput("console_log.txt");
        Log::setLevel(Log::Info);
        Log::sendMessage(Log::Info, "Simulation begin");

        mainLoop();

        Log::sendMessage(Log::Info, "Simulation end!");
    }

    static void demoKeyboardFunction(unsigned char key, int x, int y)
    {

        {
            if (!m_instance)
                return;
            switch (key)
            {
                case 'v':
                    m_instance->_changeVisibility();

                    break;
                default:
                    GLApp::keyboardFunction(key, x, y);
                    break;
            }
        }
    }

private:
    void _changeVisibility()
    {
        if (m_rigidVisible)
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->deleteVisualModule(m_rigidRenders[i]);
            }
        }
        else
        {
            for (int i = 0; i < m_rigids.size(); ++i)
            {
                m_rigids[i]->addVisualModule(m_rigidRenders[i]);
            }
        }
        m_rigidVisible = !m_rigidVisible;
    }

public:
    std::vector<RigidBody2_ptr>                   m_rigids;
    std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    bool m_rigidVisible = true;
};
