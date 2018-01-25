#include "ParameterOptimisation.h"
#include <iostream>
#include <typeinfo>

using namespace std;

namespace ParameterOptimisation {

//struct CostFunctor {
    //CostFunctor(
        //const Eigen::Matrix<double, 3, 1> target_mean,
        //const Eigen::Matrix<double, 3, 1> target_stddev,
        //const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> patch)
        //: mTargetMean(target_mean), mTargetStddev(target_stddev), mPatch(patch)
    //{
    //}

    //template <typename T>
    //bool operator()(const T* const x1, const T* const x2, const T* const x3,
                    //T* residuals) const
    //{

        //const Eigen::Matrix<T, 3, 1> parameters(x1[0], x2[0], x3[0]);

        //const static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> patch =
            //mPatch.cast<T>();

        //const static Eigen::Matrix<T, 3, 1> targetMean = mTargetMean.cast<T>();
        //const static Eigen::Matrix<T, 3, 1> targetStddev =
            //mTargetStddev.cast<T>();

        //Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> edited_patch(
            //patch.rows(), patch.cols());
        //for (auto i = 0; i < mPatch.rows(); i++) {
            //edited_patch.row(i) = applyEdit<T>(patch.row(i), parameters);
        //}

        //Eigen::Matrix<T, 3, 1> edited_mean = edited_patch.colwise().mean();

        //Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> zeromean =
            //edited_patch.rowwise() - edited_mean.transpose();

        //zeromean = zeromean.cwiseAbs();
        //Eigen::Matrix<T, 3, 1> edited_stddev = zeromean.colwise().mean();

        //Eigen::Matrix<T, 3, 1> residualsMeanVec = (targetMean - edited_mean);
        //Eigen::Matrix<T, 3, 1> residualsStddevVec =
            //(targetStddev - edited_stddev);

        //const static T meanWeight = T(1.0);
        //residuals[0] = residualsMeanVec(0) * meanWeight;
        //residuals[1] = residualsMeanVec(1) * meanWeight;
        //residuals[2] = residualsMeanVec(2) * meanWeight;

        //residuals[3] = residualsStddevVec(0);
        //residuals[4] = residualsStddevVec(1);
        //residuals[5] = residualsStddevVec(2);

        //T hist_delta = T(1.0) - (parameters(2) - parameters(1));
        //residuals[6] = hist_delta * hist_delta;
        //// residuals[6] = parameters(0) * T(10);

        //[>        if ((x2[0] > x1[0])) {<]
        //// residuals[0] *= residuals[0];
        ////}

        //// if (x1[0] > x3[0]) {
        //// residuals[2] *= residuals[2];
        //[>}<]
        //[>        residuals[3] = residualsStddevVec(0);<]
        //// residuals[4] = residualsStddevVec(1);
        //[>residuals[5] = residualsStddevVec(2);<]

        //return true;
    //}

    //const Eigen::Matrix<double, 3, 1> mTargetMean;
    //const Eigen::Matrix<double, 3, 1> mTargetStddev;
    //const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mPatch;
//};

void printVec(Eigen::Vector3d vec)
{
    cout << vec(0) << " " << vec(1) << " " << vec(2) << endl;
}

Eigen::Vector3d run(Eigen::Vector3d taget_mean, Eigen::Vector3d taget_stddev,
                    Eigen::MatrixXd input_patch,
                    Eigen::Vector3d input_parameters)
{

    // cout << input_patch << endl;

    // input_parameters = Vector3d(3.36, 0.1, 1.0);
    // JWH::convertFromGamma<double>(&input_parameters(0));

    Eigen::Matrix<double, 3, 1> output_parameters(input_parameters);

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> original_edit(
        //input_patch.rows(), input_patch.cols());
    //for (auto i = 0; i < input_patch.rows(); i++) {
        //original_edit.row(i) =
            //applyEdit<double>(input_patch.row(i), output_parameters);
    //}

    //pair<Vector3d, Vector3d> original_edit_stats =
        //JWH::computeMeanStddev(original_edit, input_patch.rows());

    //double* output_parameters_ptr = output_parameters.data();
    //ceres::Problem problem;

    /*    taget_mean = Eigen::Vector3d(0.844656, 0.845109, 0.866639);*/
    /*taget_stddev = Eigen::Vector3d(0.0336158, 0.0336651, 0.03269);*/

    //problem.AddResidualBlock(
        //new ceres::AutoDiffCostFunction<CostFunctor, 7, 1, 1, 1>(
            //new CostFunctor(taget_mean, taget_stddev, input_patch)),
        //NULL, &output_parameters_ptr[0], &output_parameters_ptr[1],
        //&output_parameters_ptr[2]);

    //problem.SetParameterLowerBound(output_parameters_ptr, 0, 0.0);
    //problem.SetParameterUpperBound(output_parameters_ptr, 0, 1.0);
    //problem.SetParameterLowerBound(output_parameters_ptr + 1, 0, 0.0);
    //problem.SetParameterUpperBound(output_parameters_ptr + 1, 0, 1.0);
    //problem.SetParameterLowerBound(output_parameters_ptr + 2, 0, 0.0);
    //problem.SetParameterUpperBound(output_parameters_ptr + 2, 0, 1.0);

    //// Run the solver!
    //ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_QR;
    //options.minimizer_progress_to_stdout = true;
    //ceres::Solver::Summary summary;
    //ceres::Solve(options, &problem, &summary);

    //std::cout << summary.BriefReport() << "\n";

    //cout << "original_edit colour     ";
    //printVec(original_edit_stats.first);
    //cout << "original_edit stddev     ";
    //printVec(original_edit_stats.second);

    //cout << "input_parameters  ";
    //printVec(input_parameters);
    //cout << "output_parameters ";
    //printVec(output_parameters);
    //cout << "target colour     ";
    //printVec(taget_mean);
    //cout << "target stddev     ";
    //printVec(taget_stddev);

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> edited_patch(
        //input_patch.rows(), input_patch.cols());
    //for (auto i = 0; i < input_patch.rows(); i++) {
        //edited_patch.row(i) =
            //applyEdit<double>(input_patch.row(i), output_parameters);
    //}

    //pair<Vector3d, Vector3d> stats =
        //JWH::computeMeanStddev(edited_patch, input_patch.rows());

    //cout << "output colour     ";
    //printVec(stats.first);
    //cout << "stddev            ";
    //printVec(stats.second);

    return output_parameters;
}
}
