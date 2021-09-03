#pragma once

#include <Eigen/Core>
#include <vector>

typedef Eigen::Matrix<size_t, 3, 1> Vector3st;

class Mesh;
class Loop;

class Vert
{
public:
    Vert(const Eigen::Vector3d& v);
    Vert();

    virtual const Eigen::Vector3d get_vert_coord() const;
    static int                    init_vert(Mesh* mesh);
    int                           set_grid_line(size_t axis, size_t grid);
    size_t                        get_grid_line(size_t axis) const;
    size_t                        get_vert_id() const;
    int                           set_id(size_t id);
    bool                          is_vert_on_axis_grid(size_t axis, size_t id_grid) const;
    const Vector3st&              get_grid_id() const;

    virtual bool is_same_vert_with(size_t id_v) const;
    virtual int  get_edge_vert_info(size_t& id_v1, size_t& id_v2, size_t& axis, size_t& id_grid_line) const;
    virtual int  get_line_vert_info(
         size_t& a1,
         size_t& a2,
         size_t& g1,
         size_t& g2,
         size_t& id_tri) const;

    virtual bool is_triangle_vert() const;
    virtual bool is_grid_vert() const;
    virtual bool is_edge_vert() const;
    virtual bool is_line_vert() const;

    bool is_on_triangle_edge() const;
    int  set_vert_on_triangle_edge();
    bool is_on_triangle_vert() const;
    int  set_vert_on_triangle_vert();

    virtual std::vector<size_t> get_triangle_id() const;
    int                         set_edge_v(size_t v1_e, size_t v2_e);
    int                         get_edge_v(size_t& v1_e, size_t& v2_e) const;

    virtual int get_edge_vert(size_t& v1, size_t& v2) const;

protected:
    Eigen::Vector3d v_;
    Vector3st       grid_line_;
    size_t          id_;
    bool            is_on_triangle_edge_;
    bool            is_on_triangle_vert_;
    size_t          id_edge_v_[2];

    // protected:
public:
    static Mesh* mesh_;
};

class VertEdge : public Vert  // vert on triangle edge
{
public:
    VertEdge(size_t id_v1, size_t id_v2, size_t axis, size_t id_grid_line);

    virtual const Eigen::Vector3d get_vert_coord() const;
    virtual bool                  is_same_vert_with(size_t id_v) const;
    virtual int                   get_edge_vert_info(size_t& id_v1, size_t& id_v2, size_t& axis, size_t& id_grid_line) const;

    virtual std::vector<size_t> get_triangle_id() const;
    virtual int                 get_edge_vert(size_t& v1, size_t& v2) const;
    virtual bool                is_triangle_vert() const;
    virtual bool                is_edge_vert() const;

private:
    size_t id_v_[2];
    size_t axis_;
    size_t id_grid_;
};

class VertLine : public Vert  // vert on cutted edge line
{
public:
    VertLine(size_t id_tri, size_t axis_1, size_t id_grid_1, size_t axis_2, size_t id_grid_2);

    virtual const Eigen::Vector3d get_vert_coord() const;
    virtual bool                  is_same_vert_with(size_t id_v) const;
    virtual int                   get_line_vert_info(
                          size_t& a1,
                          size_t& a2,
                          size_t& g1,
                          size_t& g2,
                          size_t& id_tri) const;
    virtual bool is_triangle_vert() const;
    virtual bool is_line_vert() const;

private:
    size_t axis_[2];
    size_t id_grid_[2];
    size_t id_tri_;
};

class VertGrid : public Vert
{
public:
    VertGrid(size_t axis, size_t id_grid_1, size_t axis_cut, size_t id_grid_2, size_t axis_projection, size_t id_grid_3);

public:
    virtual bool                  is_same_vert_with(size_t id_v) const;
    virtual const Eigen::Vector3d get_vert_coord() const;
    virtual bool                  is_triangle_vert() const;
    virtual bool                  is_grid_vert() const;
};
