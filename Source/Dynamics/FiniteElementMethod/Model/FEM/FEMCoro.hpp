void corotated_elas3d(float* val_ptr, const float* F_ptr, const float* R_ptr, const float* lam_ptr, const float* miu_ptr);
void corotated_elas3d_jac(float* jac_ptr, const float* F_ptr, const float* R_ptr, const float* lam_ptr, const float* miu_ptr);
void corotated_elas3d_hes(float* hes_ptr, const float* F_ptr, const float* R_ptr, const float* lam_ptr, const float* miu_ptr);
void corotated_elas3d(double* val_ptr, const double* F_ptr, const double* R_ptr, const double* lam_ptr, const double* miu_ptr);
void corotated_elas3d_jac(double* jac_ptr, const double* F_ptr, const double* R_ptr, const double* lam_ptr, const double* miu_ptr);
void corotated_elas3d_hes(double* hes_ptr, const double* F_ptr, const double* R_ptr, const double* lam_ptr, const double* miu_ptr);
