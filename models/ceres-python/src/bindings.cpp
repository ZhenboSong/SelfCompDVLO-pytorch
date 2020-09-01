#include "ceres/ceres.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using std::vector;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
namespace py = pybind11;

struct ExponentialResidual {
    ExponentialResidual(double x, double y)
            : x_(x), y_(y) {}

    template <typename T> bool operator()(const T* const m,
                                          const T* const c,
                                          T* residual) const {
        residual[0] = y_ - exp(m[0] * x_ + c[0]);
        return true;
    }

private:
    const double x_;
    const double y_;
};

int curving(const int kNumObservations, const vector<double> data) {

    double m = 0.0;
    double c = 0.0;

    Problem problem;
    for (int i = 0; i < kNumObservations; ++i) {
        problem.AddResidualBlock(
                new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
                        new ExponentialResidual(data[2 * i], data[2 * i + 1])),
                NULL,
                &m, &c);
    }

    Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
    std::cout << "Final   m: " << m << " c: " << c << "\n";
    return 0;
}


PYBIND11_MODULE(ceres_python, m) {
    m.doc() = R"pbdoc(
        Yet another python bindings for ceres
        -----------------------
        .. currentmodule:: ceres_python
        .. autosummary::
           :toctree: _generate
           optimize
    )pbdoc";

    m.def("optimize", &curving, R"pbdoc(
        Non-linear optimization entry
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
