// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

#ifndef COLMAP_SRC_BASE_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "util/types.h"

namespace colmap {

template <typename T>
static inline T CeresCalcW(T u, T v) {
  return ceres::sqrt(T(1) - u*v - v*v);
}
template <typename T>
static inline T CeresCalcLen(T x, T y, T z) {
  return ceres::sqrt(x*x + y*y + z*z);
}

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentCostFunction {
 public:
  explicit BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : x_(point2D(0)), y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
            CameraModel::kNumParams>(
        new BundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += tvec[0];
    projection[1] += tvec[1];
    projection[2] += tvec[2];

    // Project to image plane.
    //projection[0] /= projection[2];
    //projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1], projection[2],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(x_);
    residuals[1] -= T(y_);

    return true;
  }

 private:
  const double x_;
  const double y_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class BundleAdjustmentConstantPoseCostFunction {
 public:
  BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec,
                                           const Eigen::Vector2d& point2D)
      : qw_(qvec(0)),
        qx_(qvec(1)),
        qy_(qvec(2)),
        qz_(qvec(3)),
        tx_(tvec(0)),
        ty_(tvec(1)),
        tz_(tvec(2)),
        x_(point2D(0)),
        y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
            CameraModel::kNumParams>(
        new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D)));
  }

  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += T(tx_);
    projection[1] += T(ty_);
    projection[2] += T(tz_);

    // Project to image plane.
    //projection[0] /= projection[2];
    //projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1], projection[2],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(x_);
    residuals[1] -= T(y_);

    return true;
  }

 private:
  double qw_;
  double qx_;
  double qy_;
  double qz_;
  double tx_;
  double ty_;
  double tz_;
  double x_;
  double y_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigBundleAdjustmentCostFunction {
 public:
  explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : x_(point2D(0)), y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            RigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
            CameraModel::kNumParams>(
        new RigBundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const rig_qvec, const T* const rig_tvec,
                  const T* const rel_qvec, const T* const rel_tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // Concatenate rotations.
    T qvec[4];
    ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

    // Concatenate translations.
    T tvec[3];
    ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
    tvec[0] += rel_tvec[0];
    tvec[1] += rel_tvec[1];
    tvec[2] += rel_tvec[2];

    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += tvec[0];
    projection[1] += tvec[1];
    projection[2] += tvec[2];

    // Project to image plane.
    //projection[0] /= projection[2];
    //projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1], projection[2],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(x_);
    residuals[1] -= T(y_);

    return true;
  }

 private:
  const double x_;
  const double y_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `HomogeneousVectorParameterization`.
class RelativePoseCostFunction {
 public:
RelativePoseCostFunction(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2)
        : x1_(x1), x2_(x2) {dist_tp_ = IsNormalized(x1) ? 3 : 2;}

  static ceres::CostFunction* Create(const Eigen::Vector3d& x1,
                                     const Eigen::Vector3d& x2) {
    return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(
                    new RelativePoseCostFunction(x1, x2)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
    ceres::QuaternionToRotation(qvec, R.data());

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0],
        T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    if (dist_tp_ == 3) {
      const Eigen::Matrix<T, 3, 1> x1_h(T(x1_(0)), T(x1_(1)), T(x1_(2)));
      const Eigen::Matrix<T, 3, 1> x2_h(T(x2_(0)), T(x2_(1)), T(x1_(2)));

      // Squared sampson error.
      const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
      const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
      const T x2tEx1 = x2_h.transpose() * Ex1;
                    residuals[0] = x2tEx1 * x2tEx1 /
                    (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Ex1(2)*Ex1(2) +
                     Etx2(0) * Etx2(0) + Etx2(1) * Etx2(1) + Etx2(2)*Etx2(2));

    } else {
      // Homogeneous image coordinates.
      const Eigen::Matrix<T, 3, 1> x1_h(T(x1_(0)/x1_(2)), T(x1_(1)/x1_(2)), T(1));
      const Eigen::Matrix<T, 3, 1> x2_h(T(x2_(0)/x2_(2)), T(x2_(1)/x2_(2)), T(1));

      // Squared sampson error.
      const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
      const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
      const T x2tEx1 = x2_h.transpose() * Ex1;
      residuals[0] = x2tEx1 * x2tEx1 / (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) +
                                        Etx2(0) * Etx2(0) + Etx2(1) * Etx2(1));
    }
    return true;
  }

 private:
  int dist_tp_;
  const Eigen::Vector3d x1_;
  const Eigen::Vector3d x2_;
};

// Plus operation of 2D local parameterization of unit translation in 3D.
struct UnitTranslationPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    x_plus_delta[0] = x[0] + delta[0];
    x_plus_delta[1] = x[1] + delta[1];
    x_plus_delta[2] = x[2] + delta[2];

    const T squared_norm = x_plus_delta[0] * x_plus_delta[0] +
                           x_plus_delta[1] * x_plus_delta[1] +
                           x_plus_delta[2] * x_plus_delta[2];

    if (squared_norm > T(0)) {
      const T norm = T(1.0) / ceres::sqrt(squared_norm);
      x_plus_delta[0] *= norm;
      x_plus_delta[1] *= norm;
      x_plus_delta[2] *= norm;
    }

    return true;
  }
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_COST_FUNCTIONS_H_
