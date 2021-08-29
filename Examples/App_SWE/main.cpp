/**
 * @author     : Xu Ben (592228912@qq.com)
 * @date       : 2020-07-27
 * @description: Simulate with shallow water equation
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-17
 * @description: poslish code
 * @version    : 1.1
 */
#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Dynamics/HeightField/HeightFieldNode.h"
#include "Rendering/HeightFieldRender.h"
#include "IO/Image_IO/image.h"
#include "IO/Image_IO/png_io.h"
#include "IO/Image_IO/image_io.h"
#include "GUI/GlutGUI/GLApp.h"

using namespace std;
using namespace PhysIKA;

void RecieveLogMessage(const Log::Message& m)
{
    switch (m.type)
    {
        case Log::Info:
            cout << ">>>: " << m.text << endl;
            break;
        case Log::Warning:
            cout << "???: " << m.text << endl;
            break;
        case Log::Error:
            cout << "!!!: " << m.text << endl;
            break;
        case Log::User:
            cout << ">>>: " << m.text << endl;
            break;
        default:
            break;
    }
}

/**
 * setup different scenes depending on input mode parameter
 *
 * @param[in] mode   create basic scene if mode == 0
 *                   create city scene with river if mode == 1
 */
void createScene(int mode = 1)
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(1.5, 1, 1.5));
    scene.setLowerBound(Vector3f(-0.5, 0, -0.5));

    std::shared_ptr<HeightFieldNode<DataType3f>> root = scene.createNewScene<HeightFieldNode<DataType3f>>();
    root->setMass(100);
    if (mode == 1)
        root->loadParticles(Vector3f(0, 0, 0), Vector3f(2, 1.5, 2), 1024, 0.7, 1);
    else
    {
        std::string filename1 = "../../../Examples/App_SWE/terrain4-4.png";  //The pixel count is 1024*1024
        std::string filename2 = "../../../Examples/App_SWE/river4-4.png";
        root->loadParticlesFromImage(filename1, filename2, 0.1, 1);
    }
    auto ptRender = std::make_shared<HeightFieldRenderModule>();
    ptRender->setColor(Vector3f(1, 0, 0));
    root->addVisualModule(ptRender);
}

/**
 * setup city scene with river, execute one step and output the height field difference
 */
void executeOnce()
{
    std::shared_ptr<HeightFieldNode<DataType3f>> root(new HeightFieldNode<DataType3f>);
    std::string                                  filename1 = "../../../Examples/App_SWE/terrain4-4.png";  //The pixel count is 1024*1024
    std::string                                  filename2 = "../../../Examples/App_SWE/river4-4.png";
    root->loadParticlesFromImage(filename1, filename2, 0.1, 0.998);

    float             dt   = 1.0 / 60;
    std::vector<Real> vec1 = root->outputDepth();
    root->run(1, dt);
    std::vector<Real> vec2 = root->outputDepth();
    std::cout << "the depth difference:" << std::endl;
    for (int i = 0; i < vec1.size(); i++)
    {
        if (vec1[i] != vec2[i])
        {
            std::cout << i << std::endl;
        }
    }
}

int main()
{
#if 0
       executeOnce();
#else
    createScene(1);

    Log::setOutput("console_log.txt");
    Log::setLevel(Log::Info);
    Log::setUserReceiver(&RecieveLogMessage);
    Log::sendMessage(Log::Info, "Simulation begin");

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();

    Log::sendMessage(Log::Info, "Simulation end!");
#endif

    return 0;
}