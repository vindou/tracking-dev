/* C interface of the tracker side for the inference process of the flow module (flowio.cpp, library trk_flowio).
   Used by flow/live/trk_flow.cpp (C++) and flow/live.py (Python, ctypes). Functions return 0 on success unless
   stated; err (if not null) receives a message. */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FioInfo {
    double centroid[3];          /* of the model surface points, object frame (m) */
    double radius;
    double wv_min[3], wv_max[3]; /* work volume, rig frame */
    char pair_buffer[64], pose_buffer[64], hint_buffer[64];
    int width, height;
} FioInfo;

typedef struct FioPair {
    uint64_t pair_seq;
    uint64_t capture_instance_id;
    int64_t meas_time_mono_ns;   /* mean capture time of the two frames (as the pose record) */
    int64_t read_done_mono_ns;
    int32_t points;              /* points written */
    int32_t reserved;
} FioPair;

void* fio_open(const char* config_path, char* err, int errlen);
void fio_close(void* h);
void fio_info(void* h, FioInfo* out);
/* Cameras without a pair buffer (files of scenes): the default cameras of the synthetic tests. */
int fio_default_cameras(void* h, char* err, int errlen);
/* Attach to the pair buffer (or attach again after the writer has gone). 1: no writer now. */
int fio_attach_pairs(void* h, char* err, int errlen);
/* The cloud of a pair of full images: every stride-th pixel, rig frame, inside the work volume. P: cap x 3 floats (m),
   C: cap bytes (0: D435i_1, 1: D435i_2). Returns the number of points. */
int fio_cloud(void* h, const uint16_t* a, const uint16_t* b, int stride, float* P, uint8_t* C, int cap_points);
/* Waits for a pair newer than the last one read, takes the newest and builds its cloud.
   0: ok, 1: timeout, 2: the writer has gone (call fio_attach_pairs). */
int fio_read_cloud(void* h, int timeout_ms, int stride, float* P, uint8_t* C, int cap_points, FioPair* out);
/* State of the tracker from the newest pose record: 0 search, 1 track, 2 lost; -1: no live tracker. */
int fio_tracker_state(void* h);
int fio_hints_open(void* h, char* err, int errlen);
/* Publishes count hypotheses (T: count x 16 doubles, row-major T_rig_from_object; score: count floats), best first. */
int fio_hints_publish(void* h, const FioPair* pair, int count, const double* T, const float* score, float prep_ms,
                      float net_ms, float post_ms, int hypotheses);
int64_t fio_mono_ns(void);

#ifdef __cplusplus
}
#endif
