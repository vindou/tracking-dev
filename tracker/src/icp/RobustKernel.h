// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Ported from Open3D v0.20.0 (commit b6c5e196384ad71e75b6e6f9c5da22d046221f1d) for the tracker. Changes from the
// original are listed in tracker/DEVIATIONS.md (no TBB: sequential loops; no logging; minimal PointCloud).
// Source: cpp/open3d/pipelines/registration/RobustKernel.h (documentation comments shortened).
#pragma once

namespace open3d {
namespace pipelines {
namespace registration {

/// Base class that models a robust kernel for outlier rejection. The virtual
/// function Weight() must be implemented in derived classes.
class RobustKernel {
public:
    virtual ~RobustKernel() = default;
    /// Obtain the weight for the given residual according to the robust kernel
    /// model.
    virtual double Weight(double residual) const = 0;
};

/// The loss p(r) for a given residual 'r' is computed as follow: p(r) = r^2 / 2
class L2Loss : public RobustKernel {
public:
    double Weight(double residual) const override;
};

/// p(r) = abs(r)
class L1Loss : public RobustKernel {
public:
    double Weight(double residual) const override;
};

/// p(r) = r^2 / 2 if abs(r) <= k, k * (abs(r) - k / 2) otherwise
class HuberLoss : public RobustKernel {
public:
    explicit HuberLoss(double k) : k_(k) {}
    double Weight(double residual) const override;
public:
    double k_;
};

/// p(r) = k^2 / 2 * log(1 + (r / k)^2)
class CauchyLoss : public RobustKernel {
public:
    explicit CauchyLoss(double k) : k_(k) {}
    double Weight(double residual) const override;
public:
    double k_;
};

/// p(r) = (r^2 / 2) / (k + r^2)
class GMLoss : public RobustKernel {
public:
    explicit GMLoss(double k) : k_(k) {}
    double Weight(double residual) const override;
public:
    double k_;
};

/// p(r) = k^2 * (1 - (1 - r / k^2)^3) / 2 if abs(r) <= k, k^2 / 2 otherwise
class TukeyLoss : public RobustKernel {
public:
    explicit TukeyLoss(double k) : k_(k) {}
public:
    double Weight(double residual) const override;
public:
    double k_;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
