#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "GUI/QtGUI/QtApp.h"
#include "GUI/QtGUI/PVTKSurfaceMeshRender.h"
#include "GUI/QtGUI/PVTKPointSetRender.h"

#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Framework/ControllerAnimation.h"

#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/ParticleEmitter.h"
#include "Dynamics/ParticleSystem/ParticleEmitterRound.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleEmitterSquare.h"

#include "Dynamics/ParticleSystem/StaticMeshBoundary.h"

#include "Dynamics/ParticleSystem/ParticleWriter.h"

#include "Dynamics/ParticleSystem/StaticMeshBoundary.h"
#include "Framework/Topology/TriangleSet.h"

#include "Dynamics/ParticleSystem/SemiAnalyticalSFINode.h"
#include "Dynamics/ParticleSystem/TriangularSurfaceMeshNode.h"

// add by HNU
#include "GUI/QtGUI/PPropertyWidget.h"
#include <GUI/QtGUI/PMainWindow.h>
#include <GUI/QtGUI/PVTKOpenGLWidget.h>
#include <QtConcurrent\qtconcurrentrun.h>

/********************** add by HNU **************/
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include "femdynamic.h"
#include <io.h>
#include <direct.h>

int  readKFile(FEMDynamic* femAnalysis, string inFile);
int  readINPFile(FEMDynamic* femAnalysis, string inFile);
void readCommand(int argc, char* argv[], string& inFile, string& outFile, FEMDynamic* dy);
void history_record_start(string fin, string fout, char exeFile[]);
void history_record_end(double calculae_time, double contact_time, double fileTime, int calFlag, int iterNum, int gpuNum);
void copyrightStatement();
void AISIMExplicitCopyrightStatement();
int  connection_for_fem(FEMDynamic* femAnalysis);

extern ofstream f_record;

void calculateMX(int argcmx, char* argvmx[])
{
    double          t_start, t_end, contact_time, fileTime;
    FEMDynamic*     fem_analysis;
    cudaError_t     cudaStatus;
    string          inFile, outFile, inpFileType;
    ofstream        fout;
    MultiGpuManager mulGpuMag;

    /*AISIMExplicitCopyrightStatement();*/

    /*copyrightStatement();*/

    //string line = argvmx[2];
    //inFile = line.substr(line.find("INP=") + 1);
    //string line2 = argvmx[3];
    //outFile = line2.substr(line.find("WD=") + 1);

    fem_analysis = new FEMDynamic[1];

    fem_analysis->femAllocateCpu();
    if (argcmx > 1)
    {
        readCommand(argcmx, argvmx, inFile, outFile, fem_analysis);
        std::cout << inFile << std::endl;
        std::cout << outFile << std::endl;
    }
    else
    {
        std::cout << "please input calculation file" << std::endl;
        cin >> inFile;
        std::cout << "please input work path" << std::endl;
        cin >> outFile;
    }

    if (inFile == outFile)
    {
        std::cout << "输入文件有误" << std::endl;
        system("pause");
    }

    fem_analysis->multiGpuManager->judgeGpuNum();

    /**
    Choose which GPU to run on, change this on a multi-GPU system.
    */
    cudaStatus = cudaSetDevice(fem_analysis->multiGpuManager->gpu_id_array[0]);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    fem_analysis->intFilePath_    = inFile;
    fem_analysis->outputFilePath_ = outFile;

    if (0 != _access(fem_analysis->outputFilePath_.c_str(), 0))
    {
        _mkdir(fem_analysis->outputFilePath_.c_str());
    }

    string folderPath = outFile;

    folderPath += "\\history";
    if (0 != _access(folderPath.c_str(), 0))
    {
        _mkdir(folderPath.c_str());
    }
    f_record.open(folderPath + "\\history_record.txt", ios::app | ios::out);

    if (!f_record.is_open())
    {
        printf("\n记录文件打开错误\n");
    }

    //    history_record_start(inFile, outFile, argv[0]);

    /*fem_analysis->multiGpuManager->verP2PForMultiGpu();*/
    cudaStatus = cudaSetDevice(fem_analysis->multiGpuManager->gpu_id_array[0]);

    /**
    判断输入文件类型
    */
    inpFileType = inFile.substr(inFile.rfind(".") + 1);

    readKFile(fem_analysis, inFile);

    connection_for_fem(fem_analysis);

    /**
    记录开始时间
    */
    t_start = clock();

    /**
    开始计算
    */
    contact_time = 0;

    if (fem_analysis->multiGpuManager->gpu_id_array.size() >= 1)
    {
        fem_analysis->calDevice_ = SingleGpu;
        fem_analysis->DynamicAnalysisParallel(fout, contact_time, fileTime);
    }
    else
    {
        fem_analysis->calDevice_ = Cpu;
        fem_analysis->DynamicAnalysis(fout, contact_time, fileTime);
    }

    mulGpuMag = *fem_analysis->multiGpuManager;

    /**
    记录结束时间
    */
    t_end = clock();
    printf("Total calculate time: %lf\n\n", (t_end - t_start) / CLOCKS_PER_SEC);
    std::cout << "Calculation completed" << std::endl;

    /**
    记录
    */
    history_record_end((t_end - t_start) / CLOCKS_PER_SEC, contact_time / CLOCKS_PER_SEC, fileTime / CLOCKS_PER_SEC, 1, fem_analysis->cyc_num, mulGpuMag.num_gpu);

    f_record.close();

    delete[] fem_analysis;

    mulGpuMag.resetDevice();
}

/********************** add by HNU**************/

using namespace std;
using namespace PhysIKA;

std::vector<float> test_vector;

#ifdef DEBUG

std::vector<float>& creare_scene_init()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(1.5, 1.5, 1.5));
    scene.setLowerBound(Vector3f(-1.5, -0.5, -1.5));

    std::shared_ptr<StaticMeshBoundary<DataType3f>> root = scene.createNewScene<StaticMeshBoundary<DataType3f>>();
    //root->loadMesh("D:/phyaIka1/bin/Media/bowl/b3.obj");
    root->setName("StaticMesh");
    //root->loadMesh();

    std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
    root->addParticleSystem(child1);
    child1->setName("fluid");

    //std::shared_ptr<ParticleEmitterSquare<DataType3f>> child2 = std::make_shared<ParticleEmitterSquare<DataType3f>>();
    //child1->addParticleEmitter(child2);
    //child1->setMass(100);

    auto pRenderer = std::make_shared<PVTKPointSetRender>();
    pRenderer->setName("VTK Point Renderer");
    child1->addVisualModule(pRenderer);
    printf("outside visual\n");
    //     printf("outside 1\n");
    //
    std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();

    //rigidbody->loadShape("D:/phyaIka1/bin/Media/bowl/b3.obj");
    printf("outside 2\n");
    auto sRenderer = std::make_shared<PVTKSurfaceMeshRender>();
    sRenderer->setName("VTK Surface Renderer");
    rigidbody->getSurface()->addVisualModule(sRenderer);
    rigidbody->setActive(false);

    root->addRigidBody(rigidbody);

#ifdef DEBUG
    SceneGraph::Iterator it_end(nullptr);
    for (auto it = scene.begin(); it != it_end; it++)
    {
        auto node_ptr = it.get();
        std::cout << node_ptr->getClassInfo()->getClassName() << ": " << node_ptr.use_count() << std::endl;
    }

    std::cout << "Rigidbody use count: " << rigidbody.use_count() << std::endl;

    //    std::cout << "Rigidbody use count: " << rigidbody.use_count() << std::endl;

#endif DEBUG

    test_vector.resize(10);
    return test_vector;
}

#endif  // DEBUG

void create_scene_semianylitical()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(1.2));
    scene.setLowerBound(Vector3f(-0.2));

    scene.setFrameRate(1000);

    std::shared_ptr<SemiAnalyticalSFINode<DataType3f>> root = scene.createNewScene<SemiAnalyticalSFINode<DataType3f>>();
    root->setName("SemiAnalyticalSFI");
    //root->loadMesh();

    auto writer = std::make_shared<ParticleWriter<DataType3f>>();
    writer->setNamePrefix("particles_");
    root->getParticlePosition()->connect(&writer->m_position);
    root->getParticleMass()->connect(&writer->m_color_mapping);
    //root->addModule(writer);

    std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
    root->addParticleSystem(child1);
    child1->setName("fluid");
    //child1->loadParticles(Vector3f(0.75, 0.05, 0.75), Vector3f(0.85, 0.35, 0.85), 0.005);
    child1->setMass(1);

    std::shared_ptr<ParticleEmitterSquare<DataType3f>> child3 = std::make_shared<ParticleEmitterSquare<DataType3f>>();
    //    child1->addParticleEmitter(child3);

    auto pRenderer = std::make_shared<PVTKPointSetRender>();
    pRenderer->setName("VTK Point Renderer");

    //auto pRenderer = std::make_shared<PointRenderModule>();
    //pRenderer->setColor(Vector3f(0, 0, 1));

    child1->addVisualModule(pRenderer);

    std::shared_ptr<TriangularSurfaceMeshNode<DataType3f>> child2 = std::make_shared<TriangularSurfaceMeshNode<DataType3f>>("boundary");
    child2->getTriangleSet()->loadObjFile("../../Media/standard/standard_cube_01.obj");

    root->addTriangularSurfaceMeshNode(child2);

    QtApp window;
    window.createWindow(1024, 768);
    //printf("outside 4\n");
    auto classMap = Object::getClassMap();

    for (auto const c : *classMap)
        std::cout << "Class Name: " << c.first << std::endl;

    window.mainLoop();
}

int main()
{
    //adjust by HUN
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(1.5, 1.5, 1.5));
    scene.setLowerBound(Vector3f(-1.5, -0.5, -1.5));

    std::shared_ptr<StaticMeshBoundary<DataType3f>> root = scene.createNewScene<StaticMeshBoundary<DataType3f>>();
    //root->loadMesh("D:/phyaIka1/bin/Media/bowl/b3.obj");
    root->setName("StaticMesh");

    std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
    root->addParticleSystem(child1);
    child1->setName("fluid");

    auto pRenderer = std::make_shared<PVTKPointSetRender>();
    pRenderer->setName("VTK Point Renderer");
    child1->addVisualModule(pRenderer);
    printf("outside visual\n");
    //     printf("outside 1\n");
    //
    std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();

    //rigidbody->loadShape("D:/phyaIka1/bin/Media/bowl/b3.obj");
    printf("outside 2\n");
    auto sRenderer = std::make_shared<PVTKSurfaceMeshRender>();
    sRenderer->setName("VTK Surface Renderer");
    rigidbody->getSurface()->addVisualModule(sRenderer);
    rigidbody->setActive(false);

    root->addRigidBody(rigidbody);

    auto& v = test_vector;
    v.resize(5);

    QtApp window;
    window.createWindow(1024, 868);

    auto classMap = Object::getClassMap();

    // add By HNU
    QObject::connect(window.getMainWindow()->getProperty(), &PPropertyWidget::QTextFieldInitSignal, [=]() {
        QTextFieldWidget* core = window.getMainWindow()->getProperty()->getTextFieldWidget();
        //qDebug() << core;
        QObject::connect(core, &QTextFieldWidget::loadFileSignal, [&](const QString str) {
            //std::cout << "receive Output File success " << str.toStdString() << std::endl;

            //root->loadMesh("C:/Users/Desktop/Release_subdomainscale0.5/output/result_15.obj");
            //rigidbody->loadShape("C:/Users/Desktop/Release_subdomainscale0.5/output/result_15.obj");

            //QtConcurrent::run(root, &StaticMeshBoundary<DataType3f>::loadMesh, str.toStdString());
            //QtConcurrent::run([&]() {
            //    qDebug() << QThread::currentThreadId;
            //    root->loadMesh(str.toStdString());
            //    //rigidbody->loadShape(str.toStdString());
            //});

            root->loadMesh(str.toStdString());
            rigidbody->loadShape(str.toStdString());

            rigidbody->getSurface()->addVisualModule(sRenderer);
            rigidbody->setActive(false);

            // refresh view
            window.getMainWindow()->getVTKOpenGL()->prepareRenderingContex();
        });

        QObject::connect(core, &QTextFieldWidget::startCalculate, [&](QString Qstr) {
            //qDebug() << "receive calculate signal!";
            QStringList    strList = Qstr.split(" ");
            vector<string> ret;
            for (auto x : strList)
            {
                ret.push_back(x.toStdString());
                //qDebug() << x;
            }
            vector<char*> cstrings;
            cstrings.reserve(ret.size());
            for (size_t i = 0; i < ret.size(); ++i)
                cstrings.push_back(const_cast<char*>(ret[i].c_str()));

            if (!cstrings.empty())
            {
                //QtConcurrent::run(calculateMX, ret.size(), &cstrings[0]);
                calculateMX(ret.size(), &cstrings[0]);
            }
        });
    });

    //     for (auto const c : *classMap)
    //     std::cout << "Class Name: " << c.first << std::endl;

    window.mainLoop();

    //    std::cout << "Rigidbody use count: " << SceneGraph::getInstance().getRootNode()->getChildren().front().use_count() << std::endl;

    //create_scene_semianylitical();
    return 0;
}