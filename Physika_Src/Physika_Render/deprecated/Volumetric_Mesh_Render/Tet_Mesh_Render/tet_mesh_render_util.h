/*
 * @file tet_mesh_render_util.h 
 * @Basic class TetMeshRenderUtil
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
class TetMesh;

class TetrahedronRenderUtil;

template <typename Scalar>
class TetMeshRenderUtil: public VolumetricMeshRenderUtilBase<Scalar, 3>
{
public:
    explicit TetMeshRenderUtil(TetMesh<Scalar> * mesh, bool auto_compute_normal = true);
    ~TetMeshRenderUtil() = default;

    //disable copy
    TetMeshRenderUtil(const TetMeshRenderUtil &) = delete;
    TetMeshRenderUtil & operator = (const TetMeshRenderUtil &) = delete;

    const VolumetricMesh<Scalar, 3> * mesh() const override;
    void setMesh(VolumetricMesh<Scalar, 3> * mesh, bool auto_compute_normal = true) override;

    unsigned int tetrahedronNum() const;
    unsigned int eleNum() const override;
    std::shared_ptr<TetrahedronRenderUtil> getInnerTetrahedronRenderUtil();

    void bindTetMeshVAO();
    void unbindTetMeshVAO();

private:
    void initTetrahedronRenderUtil(bool auto_compute_normal);

private:
    TetMesh<Scalar> * mesh_ = nullptr;
    
    std::shared_ptr<TetrahedronRenderUtil> tet_render_util_;
};

}//end of namespace Physika