#ifndef CVFEM_H
#define CVFEM_H

#include "mymesh.h"

typedef OpenMesh::VPropHandleT<float> VFloatPropHandle;
typedef OpenMesh::VPropHandleT<MyMesh::Point> VVectorPropHandle;

// ��vh��ƽ���ϵ�P�Ĳ�ֵ���ݶȡ�ɢ��
float point_interpolation(MyMesh const &mesh, VFloatPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh);
// ������ֵ���ǰ��й�������ת��vh��ƽ����ٲ�ֵ
MyMesh::Point point_interpolation(MyMesh const &mesh, VVectorPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh);
MyMesh::Point point_gradient(MyMesh const &mesh, VFloatPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh);
float point_divergence(MyMesh const &mesh, VVectorPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh);

// ��vh_center��ƽ���϶���vh_target���ݶȡ�ɢ�ȡ�������˹
MyMesh::Point vertex_gradient(MyMesh const &mesh, VFloatPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center);
float vertex_divergence(MyMesh const &mesh, VVectorPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center);
float vertex_laplace(MyMesh const &mesh, VFloatPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center);

// Surface tension utility functions
MyMesh::Point computeNormali(std::vector<MyMesh::Point> const &points, MyMesh::Point const &ipoint, bool boundaryFlag); // �ԽǶ�ΪȨ��
float areaGradP(std::vector<MyMesh::Point> const &points, MyMesh::Point const &ipoint, MyMesh::Point const &normali, bool boundaryFlag);

#endif
