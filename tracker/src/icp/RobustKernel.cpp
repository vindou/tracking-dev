// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Source: cpp/open3d/pipelines/registration/RobustKernel.cpp (unchanged).

#include "RobustKernel.h"

#include <algorithm>
#include <cmath>

namespace {
double inline square(double x) { return x * x; }
}  // namespace

namespace open3d {
namespace pipelines {
namespace registration {

double L2Loss::Weight(double /*residual*/) const { return 1.0; }

double L1Loss::Weight(double residual) const {
    return 1.0 / std::abs(residual);
}

double HuberLoss::Weight(double residual) const {
    const double e = std::abs(residual);
    return k_ / std::max(e, k_);
}

double CauchyLoss::Weight(double residual) const {
    return 1.0 / (1 + square(residual / k_));
}

double GMLoss::Weight(double residual) const {
    return k_ / square(k_ + square(residual));
}

double TukeyLoss::Weight(double residual) const {
    const double e = std::abs(residual);
    return square(1.0 - square(std::min(1.0, e / k_)));
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
