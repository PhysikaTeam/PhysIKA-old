#ifndef CVFEM_H
#define CVFEM_H

#include "mymesh.h"

typedef OpenMesh::VPropHandleT<float> VFloatPropHandle;
typedef OpenMesh::VPropHandleT<MyMesh::Point> VVectorPropHandle;

// 求vh切平面上点P的插值、梯度、散度
float point_interpolation(MyMesh const &mesh, VFloatPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh);
// 向量插值，是把有关向量旋转到vh切平面后再插值
MyMesh::Point point_interpolation(MyMesh const &mesh, VVectorPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh);
MyMesh::Point point_gradient(MyMesh const &mesh, VFloatPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh);
float point_divergence(MyMesh const &mesh, VVectorPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh);

// 求vh_center切平面上顶点vh_target的梯度、散度、拉普拉斯
MyMesh::Point vertex_gradient(MyMesh const &mesh, VFloatPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center);
float vertex_divergence(MyMesh const &mesh, VVectorPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center);
float vertex_laplace(MyMesh const &mesh, VFloatPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center);

// Surface tension utility functions
MyMesh::Point computeNormali(std::vector<MyMesh::Point> const &points, MyMesh::Point const &ipoint, bool boundaryFlag); // 以角度为权重
float areaGradP(std::vector<MyMesh::Point> const &points, MyMesh::Point const &ipoint, MyMesh::Point const &normali, bool boundaryFlag);

#endif
