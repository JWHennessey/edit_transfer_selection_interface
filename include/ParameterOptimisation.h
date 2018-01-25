#ifndef PARAMETER_OPTIMISATION_H
#define PARAMETER_OPTIMISATION_H

//#include <ceres/ceres.h>
//#include <glog/logging.h>
#include "jwh_util.h"

namespace ParameterOptimisation {

template <typename T>
Eigen::Matrix<T, 3, 1> min(const Eigen::Matrix<T, 3, 1> lhs,
                           const Eigen::Matrix<T, 3, 1> rhs)
{
    Eigen::Matrix<T, 3, 1> output;
    output(0) = std::min(lhs(0), rhs(0));
    output(1) = std::min(lhs(1), rhs(1));
    output(2) = std::min(lhs(2), rhs(2));
    return output;
}

template <typename T>
Eigen::Matrix<T, 3, 1> max(const Eigen::Matrix<T, 3, 1> lhs,
                           const Eigen::Matrix<T, 3, 1> rhs)
{
    Eigen::Matrix<T, 3, 1> output;
    output(0) = std::max(lhs(0), rhs(0));
    output(1) = std::max(lhs(1), rhs(1));
    output(2) = std::max(lhs(2), rhs(2));
    return output;
}

template <typename T>
Eigen::Matrix<T, 3, 1> mix(const Eigen::Matrix<T, 3, 1> lhs,
                           const Eigen::Matrix<T, 3, 1> rhs,
                           const Eigen::Matrix<T, 3, 1> weights)
{
    Eigen::Matrix<T, 3, 1> output;
    output(0) = (lhs(0) * (1.0 - weights(0))) + (rhs(0) * weights(0));
    output(1) = (lhs(1) * (1.0 - weights(1))) + (rhs(0) * weights(1));
    output(2) = (lhs(2) * (1.0 - weights(2))) + (rhs(0) * weights(2));
    return output;
}

template <typename T>
Eigen::Matrix<T, 3, 1> gammaCorrection(const Eigen::Matrix<T, 3, 1> colour,
                                       const T gamma)
{
    Eigen::Matrix<T, 3, 1> output;
    T power = T(1.0) / JWH::convertToGamma<T>(gamma);
    output(0) = std::pow(colour(0), power);
    output(1) = std::pow(colour(1), power);
    output(2) = std::pow(colour(2), power);
    return output;
}

template <typename T>
Eigen::Matrix<T, 3, 1> levelsControlInputRange(
    const Eigen::Matrix<T, 3, 1> colour, const T minInput, const T maxInput)
{

    Eigen::Matrix<T, 3, 1> inputMinVec(minInput, minInput, minInput);
    Eigen::Matrix<T, 3, 1> inputMaxVec(maxInput, maxInput, maxInput);

    Eigen::Matrix<T, 3, 1> val =
        max<T>(colour - inputMinVec, Eigen::Matrix<T, 3, 1>(T(0), T(0), T(0)));

    val = val.cwiseQuotient(inputMaxVec - inputMinVec);
    val = min<T>(val, Eigen::Matrix<T, 3, 1>(T(1), T(1), T(1)));
    return val;
}

template <typename T>
Eigen::Matrix<T, 3, 1> levelsControlInput(const Eigen::Matrix<T, 3, 1> colour,
                                          const T minInput, const T gamma,
                                          const T maxInput)
{
    return gammaCorrection<T>(
        levelsControlInputRange<T>(colour, minInput, maxInput), gamma);
}

template <typename T>
Eigen::Matrix<T, 3, 1> levelsControlOutputRange(
    const Eigen::Matrix<T, 3, 1> colour, const T minOutput, const T maxOutput)
{
    Eigen::Matrix<T, 3, 1> minOutputVec(minOutput, minOutput, minOutput);
    Eigen::Matrix<T, 3, 1> maxOutputVec(maxOutput, maxOutput, maxOutput);
    Eigen::Matrix<T, 3, 1> output = mix<T>(minOutputVec, maxOutputVec, colour);
    return output;
}

template <typename T>
Eigen::Matrix<T, 3, 1> applyEdit(const Eigen::Matrix<T, 3, 1> colour,
                                 const Eigen::Matrix<T, 3, 1> parameters)
{

    Eigen::Matrix<T, 3, 1> output =
        levelsControlInput<T>(colour, T(0.0), parameters(0), T(1.0));
    output = levelsControlOutputRange<T>(output, parameters(1), parameters(2));

    return output;
}

Eigen::Vector3d run(Eigen::Vector3d taget_mean, Eigen::Vector3d taget_stddev,
                    Eigen::MatrixXd input_patch,
                    Eigen::Vector3d input_parameters);
}

#endif
