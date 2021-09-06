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

/**
 * @brief FEM Geometry SimpleObj
 * 
 */
class SimpleObj
{
private:
    /**
     * @brief Reset the vertex minimum and maximum
     * 
     */
    void   reset_v_min_max();

    /**
     * @brief Select the correct index in the face
     * 
     * @param v_idx 
     * @param f_idx 
     * @return size_t 
     */
    size_t select_correct_idx_in_face(const size_t v_idx, const size_t f_idx);

protected:
    /**
     * @brief Set the vertex minimum and maximum
     * 
     * @param x 
     * @param y 
     * @param z 
     */
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

    /**
     * @brief Construct a new Simple Obj object
     * 
     */
    SimpleObj()
    {
        reset_v_min_max();
        v_num = 0, vn_num = 0, vt_num = 0, f_num = 0;
        has_built_one_ring_faces_set     = false;
        has_built_one_ring_vertices_dict = false;
        has_computed_angles              = false;
    }

    /**
     * @brief Set the obj object
     * 
     * @param obj_file 
     */
    void set_obj(const std::string& obj_file);

    /**
     * @brief Write data to a obj file
     * 
     * @param obj_file 
     */
    void write(const std::string& obj_file) const;

    /**
     * @brief Get the face property
     * 
     * @param idx 
     * @param type 
     * @return std::vector<double> 
     */
    std::vector<double> face_property(const size_t idx, const std::string type) const;

    /**
     * @brief Normalize
     * 
     * @param center 
     * @param rate 
     */
    void normalize(const Eigen::Vector3d& center, const Eigen::Vector3d& rate);

    /**
     * @brief Normalize
     * 
     * @param center 
     * @param rate 
     * @param type "max-direction" | "all-direction"
     * @param inplace 
     */
    void normalize(Eigen::Vector3d& center, Eigen::Vector3d& rate, const std::string type, const bool inplace = true);

    /**
     * @brief Normalize
     * 
     * @param type 
     */
    void normalize(const std::string type);

    /**
     * @brief Change the axis
     * 
     * @param x 
     * @param y 
     * @param z 
     */
    void change_axis(const std::string x, const std::string y, const std::string z);

    /**
     * @brief Translate the object
     * 
     * @param displacement 
     */
    void translate(const Eigen::Vector3d& displacement);

    /**
     * @brief Rotate the object
     * 
     * @param axis 
     * @param angle 
     */
    void rotate(const std::string axis, const double angle);

    /**
     * @brief Normalize the attributes
     * 
     * @param val 
     * @param min 
     * @param max 
     * @return double 
     */
    double normalize_attr(const double val, double min, double max) const;

    /**
     * @brief Normalize the attributes
     * 
     * @param val 
     * @param min 
     * @param max 
     * @param avg 
     * @param rate 
     * @param type 
     * @return double 
     */
    double normalize_attr(const double val, double min, double max, const double avg, const double rate, const std::string type) const;

    /**
     * @brief Clean the object
     * 
     * @param threshold 
     * @param verbose 
     */
    void   clean(const double threshold, const bool verbose = true);  //!< clean those circumference >> area

    // One Ring
    std::vector<std::set<size_t>> one_ring_faces_set;
    /**
     * @brief Build the ring faces set
     * 
     * @param verbose 
     */
    void                          build_one_ring_faces_set(const bool verbose = true);  //!< Pre: None
    bool                          has_built_one_ring_faces_set;

    std::vector<std::map<size_t, std::vector<size_t>>> one_ring_vertices_dict;
    /**
     * @brief Build the ring vertices dict
     * 
     * @param verbose 
     */
    void                                               build_one_ring_vertices_dict(const bool verbose = true);  //!< Pre: one_ring_faces_set
    bool                                               has_built_one_ring_vertices_dict;
    /**
     * @brief Find two rings vertices
     * 
     * @param idx 
     * @param verbose 
     * @return std::set<size_t> 
     */
    std::set<size_t>                                   find_two_ring_vertices(const size_t idx, const bool verbose = true);

    std::vector<int> is_boundary;
    int              boundary_num;
    void             build_is_boundary(const bool verbose = true);

    // Angle
    std::vector<std::vector<double>> angles;                                     //!< (f_num, 4) - last one is the max angle
    /**
     * @brief Compute the angles
     * 
     * @param verbose 
     */
    void                             compute_angles(const bool verbose = true);  //!< Pre: None
    bool                             has_computed_angles;

    // Area
    std::vector<double>              areas;                                                    //!< (f_num)
    /**
     * @brief Compute the face areas
     * 
     * @param verbose 
     */
    void                             compute_face_areas(const bool verbose = true);            //!< Pre: None
    std::vector<std::vector<double>> face_voronoi_areas;                                       //!< (f_num, 3)
    /**
     * @brief Compute the face voronoi areas
     * 
     * @param verbose 
     */
    void                             compute_face_voronoi_areas(const bool verbose = true);    //!< Pre: areas, angles
    std::vector<double>              vertex_voronoi_areas;                                     //!< (v_num)
    /**
     * @brief Compute the vertes voronoi areas
     * 
     * @param verbose 
     */
    void                             compute_vertex_voronoi_areas(const bool verbose = true);  //!< Pre: face_voronoi_areas, one_ring_faces_set

    // Volume
    /**
     * @brief Compute the directed volume
     * 
     * @param type 
     * @param verbose 
     * @return double 
     */
    double compute_directed_volume(const std::string& type, const bool verbose = true);

    // Curvature & Normals
    // Normal & Mean Curvature
    Eigen::MatrixXd     normals;  // (3, v_num)
    std::vector<double> KHs;      // (v_num)
    double              KH_min, KH_max, KH_avg;

    /**
     * @brief Get the cot angle
     * 
     * @param idx 
     * @param neighbor 
     * @return double 
     */
    double              sum_cot_angle(const size_t idx, const size_t neighbor);

    /**
     * @brief Compute normals KHs
     * 
     * @param verbose 
     */
    void                compute_normals_KHs(const bool verbose = true);                       //!< Pre: angles, vertex_voronoi_areas, one_ring_vertices_dict

    /**
     * @brief Correct the normal direction
     * 
     * @param idx 
     * @param normal 
     */
    void                correct_normal_direction(const size_t idx, Eigen::Vector3d& normal);  //!< TODO
    // Gaussian Curvature
    std::vector<double> KGs;  //!< (v_num)
    double              KG_min, KG_max, KG_avg;

    /**
     * @brief Compute the KGs
     * 
     * @param verbose 
     */
    void                compute_KGs(const bool verbose = true);  //!< Pre: one_ring_faces_set, vertex_voronoi_areas
    // Max & Min Curvature
    std::vector<double> K_maxs;  //!< (v_num)
    std::vector<double> K_mins;  //!< (v_num)
    double              K_max_max, K_max_min, K_max_avg, K_min_max, K_min_min, K_min_avg;

    /**
     * @brief Compute the Kmaxs and the Kmins
     * 
     * @param verbose 
     */
    void                compute_Kmaxs_Kmins(const bool verbose = true);     //!< Pre: KHs, KGs

    /**
     * @brief Compute all curcatures
     * 
     * @param verbose 
     */
    void                compute_all_curvatures(const bool verbose = true);  //!< Pre: all above

    // Convex & Concave
    std::vector<double> convex;
    double              convex_min, convex_max;

    /**
     * @brief Naive the convex
     * 
     * @param verbose 
     */
    void                naive_convex(const bool verbose = true);

    // Smooth
    std::vector<double> smooth_weights;  //!< (v_num)

    /**
     * @brief Compute the smooth weight
     * 
     * @param threshold 
     */
    void                compute_smooth_weight(const double threshold);

    /**
     * @brief The smooth function
     * 
     * @param anisotropic 
     * @param step 
     * @param smooth_weight 
     * @param threshold 
     * @param verbose 
     */
    void                smooth(const bool anisotropic = false, const double step = 1e-6, const double smooth_weight = 0.5, const double threshold = 30, const bool verbose = true);
    
    /**
     * @brief The continuously smooth function
     * 
     * @param n_iter 
     * @param anisotropic 
     * @param step 
     * @param smooth_weight 
     * @param threshold 
     * @param verbose 
     */
    void                continuously_smooth(const int n_iter, const bool anisotropic = false, const double step = 1e-6, const double smooth_weight = 0.5, const double threshold = 30, const bool verbose = true);  // Pre: compute_all_curvatures()

    // Output
    /**
     * @brief Write the attributes to vtk file
     * 
     * @param save_file 
     * @param verbose 
     */
    void write_attrs_to_vtk(const std::string& save_file, const bool verbose = true);
    
    /**
     * @brief Write the attributes to vtk file
     * 
     * @param save_file 
     * @param features 
     * @param verbose 
     */
    void write_attrs_to_vtk(const std::string& save_file, const std::map<std::string, std::vector<double>>& features, const bool verbose = true);
};
