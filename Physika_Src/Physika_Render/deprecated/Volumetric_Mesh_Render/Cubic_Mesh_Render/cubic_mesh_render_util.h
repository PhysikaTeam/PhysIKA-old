/*
 * @file cubic_mesh_render_util.h 
 * @Basic class CubicMeshRenderUtil
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#pragma once

#include <memory>

#include "Physika_Render/Volumetric_Mesh_Render/Volumetric_Mesh_Render_Util_Base/volumetric_mesh_render_util_base.h"

namespace Physika{

template <typename Scalar>
class CubicMesh;

class CubeRenderUtil;

template <typename Scalar>
class CubicMeshRenderUtil: public VolumetricMeshRenderUtilBase<Scalar, 3>
{
public:
    explicit CubicMeshRenderUtil(CubicMesh<Scalar> * mesh, bool auto_compute_normal = true);
    ~CubicMeshRenderUtil() = default;

    //disable copy
    CubicMeshRenderUtil(const CubicMeshRenderUtil &) = delete;
    CubicMeshRenderUtil & operator = (const CubicMeshRenderUtil &) = delete;

    const VolumetricMesh<Scalar, 3> * mesh() const override;
    void setMesh(VolumetricMesh<Scalar, 3> * mesh, bool auto_compute_normal = true) override;

    unsigned int cubeNum() const;
    unsigned int eleNum() const override;
    std::shared_ptr<CubeRenderUtil> getInnerCubeRenderUtil();

    void bindCubicMeshLineVAO();
    void unbindCubicMeshLineVAO();

    void bindCubicMeshVAO();
    void unbindCubicMeshVAO();

private:
    void initCubeRenderUtil(bool auto_compute_normal);

private:
    CubicMesh<Scalar> * mesh_ = nullptr;
    
    std::shared_ptr<CubeRenderUtil> cube_render_util_;
};

}//end of namespace Physika