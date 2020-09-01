
/*
 * optimize reprojection and photometric error for two images from the same camera
 * zhenbo song (songzb@njust.edu.cn)
 * created on 2019-9-14
 */

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <sophus/se3.hpp>
#include <string.h>


namespace py = pybind11;

class ReprojectionError : public ceres::SizedCostFunction<2, 6> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionError(
            const Eigen::Vector3d observed_P,
            const Eigen::Vector2d observed_p,
            const Eigen::Matrix3d cam_k) :
            observed_p_(observed_p), observed_P_(observed_P), cam_k_(cam_k) {}

    virtual ~ReprojectionError() {}

    virtual bool Evaluate(
            double const* const* parameters, double *residuals, double **jacobians) const {

        Eigen::Map<const Eigen::Matrix<double,6,1>> T_se3(*parameters);

        Sophus::SE3d T_SE3 = Sophus::SE3d::exp(T_se3);

        Eigen::Vector3d Pc = T_SE3 * observed_P_;

        double fx = cam_k_(0, 0);
        double fy = cam_k_(1, 1);

        Eigen::Vector2d residual =  observed_p_ - (cam_k_ * Pc).hnormalized();

        residuals[0] = residual[0];
        residuals[1] = residual[1];

        if(jacobians != NULL) {

            Eigen::Matrix<double, 2, 6> J;

            double x = Pc[0];
            double y = Pc[1];
            double z = Pc[2];

            double x2 = x*x;
            double y2 = y*y;
            double z2 = z*z;

            J(0,0) = -fx/z;
            J(0,1) =  0;
            J(0,2) =  fx*x/z2;
            J(0,3) =  fx*x*y/z2;
            J(0,4) = -fx-fx*x2/z2;
            J(0,5) =  fx*y/z;
            J(1,0) =  0;
            J(1,1) = -fy/z;
            J(1,2) =  fy*y/z2;
            J(1,3) =  fy+fy*y2/z2;
            J(1,4) = -fy*x*y/z2;
            J(1,5) = -fy*x/z;

            int k=0;
            for(int i=0; i<2; ++i) {
                for(int j=0; j<6; ++j) {
                    jacobians[0][k++] = J(i,j);
                }
            }
        }

        return true;
    }

private:
    const Eigen::Vector2d observed_p_;
    const Eigen::Vector3d observed_P_;
    const Eigen::Matrix3d cam_k_;
};


class PhotoMetricError : public ceres::SizedCostFunction<1, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PhotoMetricError(
            const double * tar_image,
            const Eigen::Vector3d ref_pt3,
            const double ref_intensity,
            const Eigen::Matrix3d cam_k,
            const int image_h,
            const int image_w):
            tar_image_(tar_image), ref_intensity_(ref_intensity),
            ref_pt3_(ref_pt3), cam_k_(cam_k), image_w_(image_w), image_h_(image_h) {
//            std::cout << "point 3D: " << ref_pt3_ << std::endl;
//            std::cout << "ref_intensity: " << ref_intensity_ << std::endl;
//            std::cout << "image pixel 1: " << tar_image_[1] <<std::endl;
            }

    virtual ~PhotoMetricError() {}

    inline double linear_intensity(double x, double y)
    const
    {
        int idx = int(floor(x));
        int idy = int(floor(y));
        if(idx < 0 || idx > image_w_-2 || idy < 0 || idy > image_h_-2)
        {
            return 0;
        }
        double xx = x - floor(x);
        double yy = y - floor(y);
        double w_0_0 = (1 - xx) * (1 - yy);
        double w_0_1 = xx * (1 - yy);
        double w_1_0 = (1 - xx) * yy;
        double w_1_1 = xx * yy;
        double i_0_0 = tar_image_[idy * image_w_ + idx];
        double i_0_1 = tar_image_[idy * image_w_ + idx + 1];
        double i_1_0 = tar_image_[(idy + 1) * image_w_ + idx];
        double i_1_1 = tar_image_[(idy + 1) * image_w_ + idx + 1];
        return (w_0_0*i_0_0 + w_0_1*i_0_1 + w_1_0*i_1_0 + w_1_1*i_1_1);
    }

    inline Eigen::Matrix<double, 1, 2>  image_gradient(double x, double y)
    const
    {
        Eigen::Matrix<double, 1, 2> J_img;
        if(x-1 < 0 || x+1 > image_w_-2 || y-1 < 0 || y+1 > image_h_-2)
        {
            J_img(0, 0) = 0;
            J_img(0, 1) = 0;
        }
        else
        {
            J_img(0, 0) = 0.5*(linear_intensity(x+1, y)- linear_intensity(x-1, y));
            J_img(0, 1) = 0.5*(linear_intensity(x, y+1)- linear_intensity(x, y-1));
        }
        return J_img;
    }

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Matrix<double,6,1>> T_se3(*parameters);

        Sophus::SE3d T_SE3 = Sophus::SE3d::exp(T_se3);

        Eigen::Vector3d Pc = T_SE3 * ref_pt3_;

        Eigen::Vector2d proj_pt2 =  (cam_k_ * Pc).hnormalized();

        residuals[0] = ref_intensity_ - linear_intensity(proj_pt2(0), proj_pt2(1));

        double fx = cam_k_(0, 0);
        double fy = cam_k_(1, 1);

        if(jacobians != NULL)
        {

            Eigen::Matrix<double, 2, 6> J_proj;

            double x = Pc[0];
            double y = Pc[1];
            double z = Pc[2] + 1e-7;

            double x2 = x*x;
            double y2 = y*y;
            double z2 = z*z;

            J_proj(0,0) = -fx/z;
            J_proj(0,1) =  0;
            J_proj(0,2) =  fx*x/z2;
            J_proj(0,3) =  fx*x*y/z2;
            J_proj(0,4) = -fx-fx*x2/z2;
            J_proj(0,5) =  fx*y/z;
            J_proj(1,0) =  0;
            J_proj(1,1) = -fy/z;
            J_proj(1,2) =  fy*y/z2;
            J_proj(1,3) =  fy+fy*y2/z2;
            J_proj(1,4) = -fy*x*y/z2;
            J_proj(1,5) = -fy*x/z;

            Eigen::Matrix<double, 1, 2> J_img = image_gradient(proj_pt2(0), proj_pt2(1));
//            J_img(0, 0) = 0.5*(linear_intensity(proj_pt2(0)+1, proj_pt2(1))- linear_intensity(proj_pt2(0)-1, proj_pt2(1)));
//            J_img(0, 1) = 0.5*(linear_intensity(proj_pt2(0), proj_pt2(1)+1)
//                               - linear_intensity(proj_pt2(0), proj_pt2(1)-1));

            Eigen::Matrix<double, 1, 6> J = J_img * J_proj;

            int k=0;
            for(int i=0; i<1; ++i)
            {

                for(int j=0; j<6; ++j)
                {
                    jacobians[0][k++] = J(i,j);
                }
            }
        }
        return true;
    }

private:
    const double * tar_image_;
    const double ref_intensity_;
    const Eigen::Vector3d ref_pt3_;
    const Eigen::Matrix3d cam_k_;
    const int image_w_;
    const int image_h_;
};

struct PhotoMetricResidual {
    PhotoMetricResidual(
            const double * tar_image,
            const double * ref_pt3,
            const double ref_intensity,
            const double * cam_k,
            const int image_h,
            const int image_w):
            tar_image_(tar_image), ref_intensity_(ref_intensity),
            ref_pt3_(ref_pt3), cam_k_(cam_k), image_w_(image_w), image_h_(image_h) {
//                    std::cout << "point3d: "<< ref_pt3_[0] << "," << ref_pt3_[1] << ","<< ref_pt3_[2] << std::endl;
//
//        std::cout << "camera: "<< cam_k_[0] << "," << cam_k_[4] << ","<< cam_k_[6] << ","<< cam_k_[7] << std::endl;
//        std::cout << "image: " << tar_image_[0] << "," << tar_image_[1] << "," << tar_image_[2] << std::endl;
//        std::cout << "intensity: "<< ref_intensity_ << std::endl << std::endl;
}
    bool operator()(const double* const m, double* residual) const {
        double Pc[3];
        ceres::AngleAxisRotatePoint(m, ref_pt3_, Pc);
//        std::cout << "residual pt3: "<< ref_pt3_[0] << "," << ref_pt3_[1] << ","<< ref_pt3_[2] << std::endl;
//        std::cout << "residual image: " << tar_image_[0] << "," << tar_image_[1] << "," << tar_image_[2] << std::endl;
//        std::cout << "residual intensity: "<< ref_intensity_ << std::endl << std::endl;
        Pc[0] += m[3];
        Pc[1] += m[4];
        Pc[2] += m[5];
        double fx = cam_k_[0];
        double fy = cam_k_[4];
        double cx = cam_k_[6];
        double cy = cam_k_[7];

        double x = Pc[0] / Pc[2] * fx + cx;
        double y = Pc[1] / Pc[2] * fy + cy;
        int idx = int(floor(x));
        int idy = int(floor(y));

        if(x-1 < 0 || x+1 > image_w_-2 || y-1 < 0 || y+1 > image_h_-2)
        {
            residual[0] = 0;
        }
        else{
            double xx = x - floor(x);
            double yy = y - floor(y);
            double w_0_0 = (1 - xx) * (1 - yy);
            double w_0_1 = xx * (1 - yy);
            double w_1_0 = (1 - xx) * yy;
            double w_1_1 = xx * yy;
            double i_0_0 = tar_image_[idy * image_w_ + idx];
            double i_0_1 = tar_image_[idy * image_w_ + idx + 1];
            double i_1_0 = tar_image_[(idy + 1) * image_w_ + idx];
            double i_1_1 = tar_image_[(idy + 1) * image_w_ + idx + 1];
            double tar_intensity = (w_0_0*i_0_0 + w_0_1*i_0_1 + w_1_0*i_1_0 + w_1_1*i_1_1);
            residual[0] = ref_intensity_ - tar_intensity;
        }
        return true;
    }

    const double * tar_image_;
    const double ref_intensity_;
    const double * ref_pt3_;
    const double * cam_k_;
    const int image_w_;
    const int image_h_;

    static ceres::CostFunction* Create(const double * tar_image,
                                       const double * ref_pt3,
                                       const double ref_intensity,
                                       const double * cam_k,
                                       const int image_h,
                                       const int image_w)
    {
        return (new ceres::NumericDiffCostFunction<PhotoMetricResidual, ceres::CENTRAL, 1, 6>(
                new PhotoMetricResidual(tar_image,ref_pt3,ref_intensity,cam_k,image_h,image_w)));
    }
};


/*
 * part1. observation:
 * kNumObservations: number of observations
 * pt3: observed reference 3D space point in shape [3, n]
 * pt2: observed target 2D image point in shape [2, n]
 *
 * part2. initial estimation:
 * init_pose: initial pose in shape [1, 6]
 *
 * part3. system setting
 * cam_k: camera intrinsic parameters [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
 */
Eigen::Matrix4d opt_reprojection(
        const int kNumObservations,
        const Eigen::Matrix3Xd pt3,
        const Eigen::Matrix2Xd pt2,
        Eigen::Matrix<double, 1, 6> init_pose,
        Eigen::Matrix3d cam_k)
{
    Sophus::Vector6d se3;
    for(int i=0; i<6;i++)
        se3(i) = init_pose(0, i);

    ceres::Problem problem;

    for(int i=0; i<kNumObservations; i++)
    {
        Eigen::Vector2d p;
        p[0] = pt2(0, i);
        p[1] = pt2(1, i);
        Eigen::Vector3d P;
        P[0] = pt3(0, i);
        P[1] = pt3(1, i);
        P[2] = pt3(2, i);
//        std::cout << p << "," << P << std::endl;
        ceres::CostFunction* cost_function = new ReprojectionError(P, p, cam_k);
        // Set up the only cost function (also known as residual).
        problem.AddResidualBlock(cost_function, NULL, se3.data());
    }

    ceres::Solver::Options options;
    options.dynamic_sparsity = true;
    options.max_num_iterations = 1000;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    options.dogleg_type = ceres::SUBSPACE_DOGLEG;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//     std::cout << summary.BriefReport() << "\n";
//     std::cout << "estimated pose: \n" << Sophus::SE3d::exp(se3).matrix() << std::endl;

    return Sophus::SE3d::exp(se3).matrix();
}


/*
 * part1. observation:
 * kNumObservations: number of observations
 * tar_image: target image for projection in shape [C, H, W]
 * ref_pt3: reference 3D space point in shape [3, n]
 * ref_intensity: reference intensity corresponding to ref_pt3 in shape [C, n]
 *
 * part2. initial estimation:
 * init_pose: initial pose in shape [1, 6]
 *
 * part3. system setting
 * cam_k: camera intrinsic parameters [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
 * max_iter: max iteration number
 */
Eigen::Matrix<double, 1, 6> opt_photometric(
        const int kNumObservations,
        const py::array_t<double> tar_image,
        const Eigen::Matrix3Xd ref_pt3,
        const Eigen::MatrixXd ref_intensity,
        Eigen::Matrix<double, 1, 6> init_pose,
        const Eigen::Matrix3d cam_k,
        const int max_iter)
{
    py::buffer_info image_buff = tar_image.request();

    if (image_buff.ndim != 3)
        throw std::runtime_error("Image shape must be  [C, H, W]");
    int C = image_buff.shape[0], H = image_buff.shape[1], W = image_buff.shape[2];
    double *image_ptr = (double *)image_buff.ptr;

//std::cout << "image stride: " << C << H <<W << "image_size:" << image_buff.size <<std::endl;
    Sophus::Vector6d se3;
    for(int i=0; i<6;i++)
        se3(i) = init_pose(0, i);

//    for(int i=0; i< 10; i++)
//           std::cout << "image: " << image_ptr[i] <<std::endl;
    ceres::Problem problem;

    for(int i=0; i < C; i++)
    {
        // TODO: test directly double index method to save memory
        for(int j=0; j<kNumObservations; j++)
        {
            Eigen::Vector3d P;
            P[0] = ref_pt3(0, j);
            P[1] = ref_pt3(1, j);
            P[2] = ref_pt3(2, j);
            double inten = ref_intensity(i, j);

            // constructor:
            ceres::CostFunction* cost_function = new PhotoMetricError(&image_ptr[i * H * W],P, inten, cam_k, H, W);
            // Set up the only cost function (also known as residual).
            problem.AddResidualBlock(cost_function, NULL, se3.data());
        }
    }

    ceres::Solver::Options options;
    options.dynamic_sparsity = true;
    options.max_num_iterations = max_iter;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    options.dogleg_type = ceres::SUBSPACE_DOGLEG;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//     std::cout << summary.BriefReport() << "\n";
//     std::cout << "estimated pose: \n" << Sophus::SE3d::exp(se3).matrix() << std::endl;
    Eigen::Matrix<double, 1, 6> output_pose;
    for(int i=0;i < 6; i++)
        output_pose(0, i) = se3(i);

    return output_pose;
}


/*
 * part1. observation:
 * kNumObservations: number of observations
 * tar_image: target image for projection in shape [C, H, W]
 * ref_pt3: reference 3D space point in shape [3, n]
 * ref_intensity: reference intensity corresponding to ref_pt3 in shape [C, n]
 *
 * part2. initial estimation:
 * init_pose: initial pose in shape [1, 6]
 *
 * part3. system setting
 * cam_k: camera intrinsic parameters [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
 * max_iter: max iteration number
 */
Eigen::Matrix<double, 1, 6> opt_photometric_num(
        const int kNumObservations,
        const py::array_t<double> tar_image,
        const Eigen::Matrix3Xd ref_pt3,
        const Eigen::MatrixXd ref_intensity,
        Eigen::Matrix<double, 1, 6> init_pose,
        const Eigen::Matrix3d cam_k,
        const int max_iter)
{
    py::buffer_info image_buff = tar_image.request();

    if (image_buff.ndim != 3)
        throw std::runtime_error("Image shape must be  [C, H, W]");
    int C = image_buff.shape[0], H = image_buff.shape[1], W = image_buff.shape[2];
    double *image_ptr = (double *)image_buff.ptr;

    ceres::Problem problem;

    for(int i=0; i < C; i++)
    {
        // TODO: test directly double index method to save memory
        for(int j=0; j<kNumObservations; j++)
        {
            double inten = ref_intensity(i, j);
            double * pt3 = (double *)ref_pt3.data();
            // constructor:
            ceres::CostFunction* cost_function =
                    PhotoMetricResidual::Create(&image_ptr[i * H * W], &pt3[j*3],
                                                inten, (double *)cam_k.data(), H, W);
            // Set up the only cost function (also known as residual).
            problem.AddResidualBlock(cost_function, NULL, (double*)init_pose.data());
        }
    }

    ceres::Solver::Options options;
    options.dynamic_sparsity = true;
    options.max_num_iterations = max_iter;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    options.dogleg_type = ceres::SUBSPACE_DOGLEG;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//     std::cout << summary.BriefReport() << "\n";
//     std::cout << "estimated pose: \n" << Sophus::SE3d::exp(se3).matrix() << std::endl;

    return init_pose;
}

//int main()
//{
//    FILE * f_p2d = fopen("/home/song/test/ceres-python/data/p2d.txt","r");
//    FILE * f_p3d = fopen("/home/song/test/ceres-python/data/p3d.txt","r");
//    if(!f_p2d||!f_p3d)
//    {
//        printf("data file open failed \n");
//        return 0;
//    }
//    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pt2;
//    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pt3;
//    while((!feof(f_p2d))&&(!feof(f_p3d)))
//    {
//        Eigen::Vector2d t_2d;
//        Eigen::Vector3d t_3d;
//        int k = fscanf(f_p2d,"%lf %lf\n",&t_2d(0),&t_2d(1));
//        k = fscanf(f_p3d,"%lf %lf %lf\n",&t_3d(0),&t_3d(1),&t_3d(2));\
//        pt2.push_back(t_2d);
//        pt3.push_back(t_3d);
//    }
//    fclose(f_p2d);
//    fclose(f_p3d);
//    Sophus::Vector6d se3;
//    ceres::Problem problem;
//
//    for(int i=0; i<pt2.size(); i++)
//    {
//        Eigen::Vector2d p;
//        p[0] = pt2[i](0);
//        p[1] = pt2[i](1);
//        Eigen::Vector3d P;
//        P[0] = pt3[i](0);
//        P[1] = pt3[i](1);
//        P[2] = pt3[i](2);
//        ceres::CostFunction* cost_function = new ReprojectionError(p, P);
//        // Set up the only cost function (also known as residual).
//        problem.AddResidualBlock(cost_function, NULL, se3.data());
//    }
//
//    ceres::Solver::Options options;
//    options.dynamic_sparsity = true;
//    options.max_num_iterations = 100;
//    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
//    options.minimizer_type = ceres::TRUST_REGION;
//    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//    options.trust_region_strategy_type = ceres::DOGLEG;
//    options.minimizer_progress_to_stdout = true;
//    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
//
//    ceres::Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.BriefReport() << "\n";
//
//    std::cout << "estimated pose: \n" << Sophus::SE3d::exp(se3).matrix() << std::endl;
//
//}



PYBIND11_MODULE(ceres_python, m) {
    m.doc() = R"pbdoc(
        python binding ceres for VO problems
        -----------------------
        .. currentmodule:: ceres_python
        .. autosummary::
           :toctree: _generate
           optimize
    )pbdoc";

    m.def("opt_reproject", &opt_reprojection, R"pbdoc(
        optimize reprojection error for two images from the same camera
    )pbdoc");

    m.def("opt_photometric", &opt_photometric, R"pbdoc(
        optimize photometric error for two images from the same camera
    )pbdoc");

    m.def("opt_photometric_auto", &opt_photometric_num, R"pbdoc(
        optimize photometric error using autodiff for two images from the same camera
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}