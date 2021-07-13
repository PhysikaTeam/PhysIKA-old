#pragma once

//#include "Framework/Framework/ModuleCompute.h"
#include "Framework/Framework/ModuleCustom.h"
#include "Dynamics/RigidBody/RigidBody2.h"
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

namespace PhysIKA {
class RigidEnergyComputeModule : public CustomModule
{
public:
    virtual void applyCustomBehavior()
    {

        //
        if (m_curStep == 0 && outfile != "")
        {
            m_outStream.open(outfile);
        }

        if (m_curStep < maxStep)
        {
            totalEng = 0.0;

            for (auto prigid : allRigids)
            {
                double cureng = 0.0;

                auto& inertia = prigid->getI();

                Vector3f linv = prigid->getLinearVelocity();
                cureng += 0.5 * (inertia.getMass()) * linv.dot(linv);

                Vector3f angv = prigid->getAngularVelocity();
                angv          = prigid->getGlobalQ().getConjugate().rotate(angv);
                cureng += 0.5 * angv.dot(inertia.getInertiaDiagonal() * angv);

                totalEng += cureng;
            }

            m_outStream << totalEng << std::endl;

            std::cout << "  Total Eng:  " << totalEng << std::endl;

            ++m_curStep;
        }
        else if (m_curStep == maxStep)
        {
            m_outStream.close();
            ++m_curStep;
        }
    };

public:
    std::string outfile;

    std::vector<RigidBody2_ptr> allRigids;

    int maxStep = 1000;

private:
    std::ofstream m_outStream;

    int    m_curStep = 0;
    double totalEng  = 0.0;
};

class PendulumExtensionComputeModule : public CustomModule
{
public:
    virtual void applyCustomBehavior()
    {

        //
        if (m_curStep == 0 && outfile != "")
        {
            m_outStream.open(outfile);
        }

        if (m_curStep < maxStep)
        {
            Vector3f curp       = pLastRigid->getGlobalR();
            double   extension  = offsetPos[1] - curp[1];
            double   extPercent = extension / totalLength;

            std::cout << "    Pendulum Extension:  " << extension << "    Percent:  " << extPercent << std::endl;
            m_outStream << extPercent << std::endl;

            ++m_curStep;
        }
        else if (m_curStep == maxStep)
        {
            m_outStream.close();
            ++m_curStep;
        }
    };

public:
    std::string outfile;

    //std::vector<RigidBody2_ptr> allRigids;
    RigidBody2_ptr pLastRigid;

    Vector3f offsetPos;

    double totalLength = 1.0;

    int maxStep = 600;

private:
    std::ofstream m_outStream;

    int m_curStep = 0;
};

//void computeMeshAABB(std::shared_ptr<PointSet<DataType3f>> points, Vector3f & center, Vector3f & halfSize)
//{
//    int nPoints = points->getPointSize();
//    if (nPoints <= 0)
// return;

//    auto& pointArr = points->getPoints();
//    HostArray<Vector3f> hpoints;
//    hpoints.resize(nPoints);
//    PhysIKA::Function1Pt::copy(hpoints, pointArr);

//    Vector3f pmin = hpoints[0];
//    Vector3f pmax = hpoints[0];
//    for (int i = 1; i < nPoints; ++i)
//    {
// Vector3f curp = hpoints[i];
// pmin[0] = min(pmin[0], curp[0]);
// pmin[1] = min(pmin[1], curp[1]);
// pmin[2] = min(pmin[2], curp[2]);
// pmax[0] = max(pmax[0], curp[0]);
// pmax[1] = max(pmax[1], curp[1]);
// pmax[2] = max(pmax[2], curp[2]);
//    }

//    center = (pmin + pmax)*0.5;
//    halfSize = (pmax - pmin)*0.5;
//}

}  // namespace PhysIKA