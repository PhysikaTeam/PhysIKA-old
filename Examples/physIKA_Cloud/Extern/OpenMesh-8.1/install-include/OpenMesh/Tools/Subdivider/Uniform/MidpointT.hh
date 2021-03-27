#pragma once

#include <OpenMesh/Core/Mesh/BaseKernel.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Core/Utils/PropertyManager.hh>

#include <algorithm>

namespace OpenMesh {
namespace Subdivider {
namespace Uniform {

/**
 * Midpoint subdivision algorithm.
 *
 * With every step, the set of vertices is replaced by the midpoints of all
 * current edges. Then, two sets of faces are created to set up the new
 * connectivity: From all midpoints of edges surrounding an original face, a new
 * face is created. Also, for all midpoints of edges surrounding an original
 * vertex, a new face is created.
 *
 * @note This algorithm ignores the _update_points option.
 * @note This algorithm is best suited for closed meshes since boundaries tend
 *       to fragment into isolated faces after a few iterations.
 */
template<typename MeshType, typename RealType = double>
class MidpointT : public SubdividerT<MeshType, RealType>
{
public:
    typedef RealType real_t;
    typedef MeshType mesh_t;
    typedef SubdividerT<MeshType, RealType> parent_t;

    // Inherited constructors
    MidpointT() : parent_t() {}
    MidpointT(mesh_t& _m) : parent_t(_m) {}

    const char* name() const { return "midpoint"; }

protected: // SubdividerT interface
    bool prepare(mesh_t& _m)
    {
        return true;
    }

    //! Performs one step of Midpoint subdivision.
    //! @note The _update_points option is ignored
    bool subdivide(mesh_t& _m, size_t _n, const bool _update_points = true)
    {
        _m.request_halfedge_status();
        _m.request_edge_status();
        _m.request_vertex_status();
        _m.request_face_status();
        PropertyManager<EPropHandleT<typename mesh_t::VertexHandle>> edge_midpoint(_m, "edge_midpoint");
        PropertyManager<VPropHandleT<bool>> is_original_vertex(_m, "is_original_vertex");

        for (size_t iteration = 0; iteration < _n; ++iteration) {
            is_original_vertex.set_range(_m.vertices_begin(), _m.vertices_end(), true);
            // Create vertices on edge midpoints
            for (auto eh : _m.edges()) {
                VertexHandle new_vh = _m.new_vertex(_m.calc_edge_midpoint(eh));
                edge_midpoint[eh] = new_vh;
                is_original_vertex[new_vh] = false;
            }
            // Create new faces from original faces
            for (auto fh : _m.faces()) {
                std::vector<typename mesh_t::VertexHandle> new_corners;
                for (auto eh : _m.fe_range(fh))
                    new_corners.push_back(edge_midpoint[eh]);
                _m.add_face(new_corners);
            }
            // Create new faces from original vertices
            for (auto vh : _m.vertices()) {
                if (is_original_vertex[vh]) {
                    if (!_m.is_boundary(vh)) {
                        std::vector<typename mesh_t::VertexHandle> new_corners;
                        for (auto eh : _m.ve_range(vh))
                            new_corners.push_back(edge_midpoint[eh]);
                        std::reverse(new_corners.begin(), new_corners.end());
                        _m.add_face(new_corners);
                    }
                }
            }
            for (auto vh : _m.vertices())
                if (is_original_vertex[vh])
                    _m.delete_vertex(vh);
            _m.garbage_collection();
        }
        _m.release_face_status();
        _m.release_vertex_status();
        _m.release_edge_status();
        _m.release_halfedge_status();
        return true;
    }

    bool cleanup(mesh_t& _m)
    {
        return true;
    }
};

} // namespace Uniform
} // namespace Subdivider
} // namespace OpenMesh
