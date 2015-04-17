/*
 * @file fem_solid.h 
 * @Brief FEM driver for solids, not necessarily homogeneous.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_SOLID_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_SOLID_H_

#include <vector>
#include <string>
#include "Physika_Dynamics/Utilities/time_stepping_method.h"
#include "Physika_Dynamics/FEM/fem_base.h"

namespace Physika{

template <typename Scalar, int Dim> class ConstitutiveModel;
template <typename Scalar, int Dim> class FEMSolidForceModel;
template<typename Scalar,int Dim> class CollidableObject;

/*
 * FEM driver for solids, not necessarily homogeneous:
 * 1. All elements share one constitutive model if only one is provided. The solid
 *    is homogeneous in this case.
 * 2. Elements in the same region of simulation mesh share one constitutive model. In this
 *    case the number of constitutive models equals the number of regions of the simulation mesh.
 * 3. Element-wise consitutive model is used.
 *
 */

template <typename Scalar, int Dim>
class FEMSolid: public FEMBase<Scalar,Dim>
{
public:
    FEMSolid();
    FEMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    ~FEMSolid();

    //virtual methods
    void initConfiguration(const std::string &file_name);
    void printConfigFileFormat();
    void initSimulationData();
    void advanceStep(Scalar dt);
    bool withRestartSupport() const;
    void write(const std::string &file_name);
    void read(const std::string &file_name);
    void addPlugin(DriverPluginBase<Scalar> *plugin);

    //set&&get constitutive model (data are copied)
    //set***Material() needs to be called to update material if volumetric mesh is updated
    unsigned int materialNum(unsigned int object_idx) const;
    void setHomogeneousMaterial(unsigned int object_idx, const ConstitutiveModel<Scalar,Dim> &material);
    //the number of materials must match the number of regions on simulation mesh
    void setRegionWiseMaterial(unsigned int object_idx, const std::vector<ConstitutiveModel<Scalar,Dim>*> &materials);
    //the number of materials must match the number of simulation elements
    void setElementWiseMaterial(unsigned int object_idx, const std::vector<ConstitutiveModel<Scalar,Dim>*> &materials);  
    const ConstitutiveModel<Scalar,Dim>& elementMaterial(unsigned int object_idx, unsigned int ele_idx) const;  
    ConstitutiveModel<Scalar,Dim>& elementMaterial(unsigned int object_idx, unsigned int ele_idx);

    void setTimeSteppingMethod(TimeSteppingMethod method);

    //manage the kinematic collidable objects in scene
    unsigned int kinematicObjectNum() const;
    void addKinematicObject(const CollidableObject<Scalar,Dim> &object);
    void removeKinematicObject(unsigned int object_idx);
    const CollidableObject<Scalar,Dim>& kinematicObject(unsigned int object_idx) const;
    CollidableObject<Scalar,Dim>& kinematicObject(unsigned int object_idx);

protected:
    virtual void appendDataWithObject();
    virtual void removeDataWithObject(unsigned int object_idx);
    void clearAllMaterials(); //clear materials of all objects
    void clearMaterial(unsigned int object_idx); //clear material of a specific object
    void addMaterial(unsigned int object_idx, const ConstitutiveModel<Scalar,Dim> &material);
    void advanceStepForwardEuler(Scalar dt);
    void advanceStepBackwardEuler(Scalar dt);
    void createFEMSolidForceModel(unsigned int object_idx);
    void clearAllFEMSolidForceModels();
    void clearFEMSolidForceModel(unsigned int object_idx);
    void clearKinematicObjects();
    void resolveContactWithKinematicObjects(unsigned int object_idx);
protected:
    std::vector<std::vector<ConstitutiveModel<Scalar,Dim> *> > constitutive_model_;
    std::vector<FEMSolidForceModel<Scalar,Dim>*> force_model_;
    std::vector<CollidableObject<Scalar,Dim>*> collidable_objects_; //the kinematic collidable objects in scene
    TimeSteppingMethod integration_method_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_SOLID_H_
