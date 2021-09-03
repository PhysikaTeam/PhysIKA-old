#pragma once

#include <vector>
#include <string>
#include <map>
#include <set>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <Eigen/Dense>

class SimpleObj
{
private:
    void   reset_v_min_max();
    size_t select_correct_idx_in_face(const size_t v_idx, const size_t f_idx);

protected:
    void set_v_min_max(const double x, const double y, const double z);

public:
    size_t          v_num, vn_num, vt_num;
    Eigen::MatrixXd v_mat;  // (3, v_num)
    Eigen::MatrixXd vn_mat;
    Eigen::MatrixXd vt_mat;
    Eigen::Vector3d v_max, v_min;

    size_t          f_num;
    Eigen::MatrixXi f_mat;  // (3, f_num)
    Eigen::MatrixXi ft_mat;
    Eigen::MatrixXi fn_mat;

    SimpleObj()
    {
        reset_v_min_max();
        v_num = 0, vn_num = 0, vt_num = 0, f_num = 0;
        has_built_one_ring_faces_set     = false;
        has_built_one_ring_vertices_dict = false;
        has_computed_angles              = false;
    }

    void set_obj(const std::string& obj_file);
    void write(const std::string& obj_file) const;

    std::vector<double> face_property(const size_t idx, const std::string type) const;

    void normalize(const Eigen::Vector3d& center, const Eigen::Vector3d& rate);
    // type == "max-direction" | "all-direction"
    void normalize(Eigen::Vector3d& center, Eigen::Vector3d& rate, const std::string type, const bool inplace = true);
    void normalize(const std::string type);
    void change_axis(const std::string x, const std::string y, const std::string z);

    void translate(const Eigen::Vector3d& displacement);
    void rotate(const std::string axis, const double angle);

    double normalize_attr(const double val, double min, double max) const;
    double normalize_attr(const double val, double min, double max, const double avg, const double rate, const std::string type) const;
    void   clean(const double threshold, const bool verbose = true);  // clean those circumference >> area

    // One Ring
    std::vector<std::set<size_t>> one_ring_faces_set;
    void                          build_one_ring_faces_set(const bool verbose = true);  // Pre: None
    bool                          has_built_one_ring_faces_set;

    std::vector<std::map<size_t, std::vector<size_t>>> one_ring_vertices_dict;
    void                                               build_one_ring_vertices_dict(const bool verbose = true);  // Pre: one_ring_faces_set
    bool                                               has_built_one_ring_vertices_dict;
    std::set<size_t>                                   find_two_ring_vertices(const size_t idx, const bool verbose = true);

    std::vector<int> is_boundary;
    int              boundary_num;
    void             build_is_boundary(const bool verbose = true);

    // Angle
    std::vector<std::vector<double>> angles;                                     // (f_num, 4) - last one is the max angle
    void                             compute_angles(const bool verbose = true);  // Pre: None
    bool                             has_computed_angles;

    // Area
    std::vector<double>              areas;                                                    // (f_num)
    void                             compute_face_areas(const bool verbose = true);            // Pre: None
    std::vector<std::vector<double>> face_voronoi_areas;                                       // (f_num, 3)
    void                             compute_face_voronoi_areas(const bool verbose = true);    // Pre: areas, angles
    std::vector<double>              vertex_voronoi_areas;                                     // (v_num)
    void                             compute_vertex_voronoi_areas(const bool verbose = true);  // Pre: face_voronoi_areas, one_ring_faces_set

    // Volume
    double compute_directed_volume(const std::string& type, const bool verbose = true);

    // Curvature & Normals
    // Normal & Mean Curvature
    Eigen::MatrixXd     normals;  // (3, v_num)
    std::vector<double> KHs;      // (v_num)
    double              KH_min, KH_max, KH_avg;
    double              sum_cot_angle(const size_t idx, const size_t neighbor);
    void                compute_normals_KHs(const bool verbose = true);                       // Pre: angles, vertex_voronoi_areas, one_ring_vertices_dict
    void                correct_normal_direction(const size_t idx, Eigen::Vector3d& normal);  // TODO
    // Gaussian Curvature
    std::vector<double> KGs;  // (v_num)
    double              KG_min, KG_max, KG_avg;
    void                compute_KGs(const bool verbose = true);  // Pre: one_ring_faces_set, vertex_voronoi_areas
    // Max & Min Curvature
    std::vector<double> K_maxs;  // (v_num)
    std::vector<double> K_mins;  // (v_num)
    double              K_max_max, K_max_min, K_max_avg, K_min_max, K_min_min, K_min_avg;
    void                compute_Kmaxs_Kmins(const bool verbose = true);     // Pre: KHs, KGs
    void                compute_all_curvatures(const bool verbose = true);  // Pre: all above

    // Convex & Concave
    std::vector<double> convex;
    double              convex_min, convex_max;
    void                naive_convex(const bool verbose = true);

    // Smooth
    std::vector<double> smooth_weights;  // (v_num)
    void                compute_smooth_weight(const double threshold);
    void                smooth(const bool anisotropic = false, const double step = 1e-6, const double smooth_weight = 0.5, const double threshold = 30, const bool verbose = true);
    void                continuously_smooth(const int n_iter, const bool anisotropic = false, const double step = 1e-6, const double smooth_weight = 0.5, const double threshold = 30, const bool verbose = true);  // Pre: compute_all_curvatures()

    // Output
    void write_attrs_to_vtk(const std::string& save_file, const bool verbose = true);
    void write_attrs_to_vtk(const std::string& save_file, const std::map<std::string, std::vector<double>>& features, const bool verbose = true);
};
