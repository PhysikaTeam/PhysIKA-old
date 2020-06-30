#ifndef SOLVE_MS_JJ_H
#define SOLVE_MS_JJ_H

#include <vector>
#include <Eigen/Core>
#include "mass_spring_obj.h"

template<typename T,size_t dim>
int solve(std::vector<Eigen::Triplet<double> > &tripletsForHessian,Eigen::VectorXd &Jacobian, mass_spring_obj<T, dim> &ms);


#endif // SOLVE_MS_JJ_H
