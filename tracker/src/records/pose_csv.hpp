// CSV form of the pose record (trk_offline, trk_posedump; the viewer writes the same columns).
#pragma once
#include <cinttypes>
#include <cstdio>

#include "records.hpp"

namespace trk {
inline void write_pose_csv_header(FILE* f, bool with_truth) {
    std::fprintf(f, "pose_seq,pair_seq,writer_instance,capture_instance,meas_time_mono_ns,publish_time_mono_ns,pair_dt_ms,"
                    "state,valid,reject,px,py,pz,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz,icp_px,icp_py,icp_pz,icp_qw,icp_qx,icp_qy,"
                    "icp_qz,rmse,inlier_ratio,min_eig,points,iterations,violation,t_read_us,t_build_us,t_filter_us,t_solve_us,"
                    "t_check_us,t_track_us,t_publish_us,t_total_us,pairs_skipped,hints_tried,by_hint%s\n",
                 with_truth ? ",true_px,true_py,true_pz,true_qw,true_qx,true_qy,true_qz" : "");
}
void write_pose_csv_row(FILE* f, const PoseRecord& p, const double* truth16);
}  // namespace trk
