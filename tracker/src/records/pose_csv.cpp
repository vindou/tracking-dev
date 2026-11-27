#include "pose_csv.hpp"

#include <Eigen/Geometry>

namespace trk {
void write_pose_csv_row(FILE* f, const PoseRecord& p, const double* t) {
    std::fprintf(f, "%" PRIu64 ",%" PRIu64 ",%016" PRIx64 ",%016" PRIx64 ",%" PRId64 ",%" PRId64 ",%.4f,%u,%u,%u", p.pose_seq,
                 p.pair_seq, p.writer_instance_id, p.capture_instance_id, p.meas_time_mono_ns, p.publish_time_mono_ns,
                 p.pair_dt_ms, p.track_state, p.valid, p.reject_reasons);
    for (double v : p.position) std::fprintf(f, ",%.7f", v);
    for (double v : p.orientation) std::fprintf(f, ",%.8f", v);
    for (double v : p.velocity) std::fprintf(f, ",%.5f", v);
    for (double v : p.angular_velocity) std::fprintf(f, ",%.5f", v);
    for (double v : p.icp_position) std::fprintf(f, ",%.7f", v);
    for (double v : p.icp_orientation) std::fprintf(f, ",%.8f", v);
    std::fprintf(f, ",%.7f,%.5f,%.6f,%u,%u,%.4f", p.fit_rmse, p.inlier_ratio, p.min_eigenvalue, p.point_count,
                 p.icp_iterations, double(p.free_space_violation));
    for (uint32_t s = 0; s < kStageCount; s++) std::fprintf(f, ",%.1f", double(p.stage_us[s]));
    std::fprintf(f, ",%" PRIu64 ",%u,%u", p.pairs_skipped, unsigned(p.hints_tried), unsigned(p.by_hint));
    if (t) {
        Eigen::Matrix3d R;
        for (int i = 0; i < 9; i++) R(i / 3, i % 3) = t[(i / 3) * 4 + i % 3];
        Eigen::Quaterniond q(R);
        if (q.w() < 0) q.coeffs() = -q.coeffs();
        std::fprintf(f, ",%.7f,%.7f,%.7f,%.8f,%.8f,%.8f,%.8f", t[3], t[7], t[11], q.w(), q.x(), q.y(), q.z());
    }
    std::fputc('\n', f);
}
}  // namespace trk
