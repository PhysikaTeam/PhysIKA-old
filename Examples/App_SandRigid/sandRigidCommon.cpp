#include "sandRigidCommon.h"

#include "Dynamics/RigidBody/RigidBody2.h"
#include "Rendering/RigidMeshRender.h"

bool computeBoundingBox(PhysIKA::Vector3f& center, PhysIKA::Vector3f& boxsize, const std::vector<PhysIKA::Vector3f>& vertices)
{
    if (vertices.size() <= 0)
        return false;

    boxsize                = PhysIKA::Vector3f();
    PhysIKA::Vector3f bmin = vertices[0];
    PhysIKA::Vector3f bmax = vertices[0];
    for (int i = 0; i < vertices.size(); ++i)
    {
        const PhysIKA::Vector3f& ver = vertices[i];
        bmin[0]                      = min(bmin[0], ver[0]);
        bmin[1]                      = min(bmin[1], ver[1]);
        bmin[2]                      = min(bmin[2], ver[2]);

        bmax[0] = max(bmax[0], ver[0]);
        bmax[1] = max(bmax[1], ver[1]);
        bmax[2] = max(bmax[2], ver[2]);
    }

    center  = (bmin + bmax) * 0.5;
    boxsize = bmax - bmin;
    return true;
}

void PkAddBoundaryRigid(std::shared_ptr<Node> root, Vector3f origin, float sizex, float sizez, float boundarysize, float boundaryheight)
{
    float bsizeXlr = boundarysize;
    float bsizeZlr = (2.0 * boundarysize + sizez) / 2.0;

    float bsizeZud = boundarysize;
    float bsizeXud = (2.0 * boundarysize + sizex) / 2.0;

    Vector3f color(133.0 / 255.0, 95.0 / 255.0, 66.0 / 255.0);
    color *= 2;

    {
        auto prigidl = std::make_shared<RigidBody2<DataType3f>>();
        root->addChild(prigidl);

        auto renderModule = std::make_shared<RigidMeshRender>(prigidl->getTransformationFrame());
        renderModule->setColor(color);
        prigidl->addVisualModule(renderModule);

        Vector3f scale(bsizeXlr, boundaryheight, bsizeZlr);
        prigidl->loadShape("../../Media/standard/standard_cube.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigidl->getTopologyModule());
        triset->scale(scale);

        prigidl->setGlobalR(origin + Vector3f(-0.5 * sizex - boundarysize, boundaryheight, 0));
    }

    {
        auto prigidr = std::make_shared<RigidBody2<DataType3f>>();
        root->addChild(prigidr);

        auto renderModule = std::make_shared<RigidMeshRender>(prigidr->getTransformationFrame());
        renderModule->setColor(color);
        prigidr->addVisualModule(renderModule);

        Vector3f scale(bsizeXlr, boundaryheight, bsizeZlr);
        prigidr->loadShape("../../Media/standard/standard_cube.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigidr->getTopologyModule());
        triset->scale(scale);

        prigidr->setGlobalR(origin + Vector3f(0.5 * sizex + boundarysize, boundaryheight, 0));
    }

    {
        auto prigidu = std::make_shared<RigidBody2<DataType3f>>();
        root->addChild(prigidu);

        auto renderModule = std::make_shared<RigidMeshRender>(prigidu->getTransformationFrame());
        renderModule->setColor(color);
        prigidu->addVisualModule(renderModule);

        Vector3f scale(bsizeXud, boundaryheight, bsizeZud);
        prigidu->loadShape("../../Media/standard/standard_cube.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigidu->getTopologyModule());
        triset->scale(scale);

        prigidu->setGlobalR(origin + Vector3f(0, boundaryheight, -0.5 * sizez - boundarysize));
    }

    {
        auto prigidd = std::make_shared<RigidBody2<DataType3f>>();
        root->addChild(prigidd);

        auto renderModule = std::make_shared<RigidMeshRender>(prigidd->getTransformationFrame());
        renderModule->setColor(color);
        prigidd->addVisualModule(renderModule);

        Vector3f scale(bsizeXud, boundaryheight, bsizeZud);
        prigidd->loadShape("../../Media/standard/standard_cube.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigidd->getTopologyModule());
        triset->scale(scale);

        prigidd->setGlobalR(origin + Vector3f(0, boundaryheight, 0.5 * sizez + boundarysize));
    }
}