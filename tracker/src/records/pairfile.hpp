// Record/replay file of depth pairs (T2 writes it, T3 records and replays it). Format: README.md "Pair file".
//   header (512 bytes): "TRKPAIRS", u32 version 1, u32 header_size 512, u32 width, u32 height, u32 bytes_per_pixel,
//   u32 source (0 capture, 1 synthetic), u64 pair_count, i64 created_real_ns, f64 frame_period_ms, f64 tolerance_ms,
//   CameraInfo[2] (96 bytes each, as in the pair-buffer header) at offset 128, reserved to 512.
//   per pair: chunk header (272 bytes): u32 magic "PAIR", u32 coded size A, u32 coded size B, u32 flags (bit 0: truth
//   present), PairRecord (128 bytes, capture layout v2), f64 truth T_rig_from_object[16] row-major (synthetic data),
//   then image A and image B, each run-length coded (repeated: u16 zero-run length, u16 literal count, literals).
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "shm_layout.hpp"   // capture: PairRecord, CameraInfo

namespace trk {

constexpr uint32_t kPairFileVersion = 1;

struct PairFileHeader {             // 512 bytes
    char magic[8];                  // 0   "TRKPAIRS"
    uint32_t version;               // 8
    uint32_t header_size;           // 12  512
    uint32_t width, height;         // 16, 20
    uint32_t bytes_per_pixel;       // 24  2
    uint32_t source;                // 28  0 capture recording, 1 synthetic
    uint64_t pair_count;            // 32
    int64_t created_real_ns;        // 40
    double frame_period_ms;         // 48
    double tolerance_ms;            // 56
    uint8_t reserved0[64];          // 64
    cap::CameraInfo cam[2];         // 128 A, 224 B
    uint8_t reserved[512 - 320];    // 320
};
static_assert(sizeof(PairFileHeader) == 512, "layout");

struct PairChunkHeader {            // 272 bytes
    uint32_t magic;                 // 0  0x52494150 "PAIR"
    uint32_t coded_a, coded_b;      // 4, 8 bytes
    uint32_t flags;                 // 12 bit 0: truth present
    cap::PairRecord rec;            // 16
    double truth[16];               // 144 T_rig_from_object, row-major (synthetic)
};
static_assert(sizeof(PairChunkHeader) == 272, "layout");
constexpr uint32_t kPairChunkMagic = 0x52494150;

std::size_t rle_encode(const uint16_t* img, std::size_t n, std::vector<uint8_t>& out);   // returns coded bytes
bool rle_decode(const uint8_t* in, std::size_t bytes, uint16_t* img, std::size_t n);

class PairFileWriter {
public:
    bool open(const std::string& path, const cap::CameraInfo cams[2], uint32_t source, double period_ms,
              double tolerance_ms, std::string& err);
    bool append(const cap::PairRecord& rec, const uint16_t* a, const uint16_t* b, const double* truth16 = nullptr);
    bool close();   // writes the pair count
    ~PairFileWriter() { close(); }
    uint64_t pairs() const { return hdr_.pair_count; }
    uint64_t bytes() const { return bytes_; }
private:
    FILE* f_ = nullptr;
    PairFileHeader hdr_{};
    std::vector<uint8_t> ca_, cb_;
    uint64_t bytes_ = 0;
};

class PairFileReader {
public:
    bool open(const std::string& path, std::string& err);
    const PairFileHeader& header() const { return hdr_; }
    // Next pair; false at the end or on error (err set on error).
    bool next(PairChunkHeader& ch, uint16_t* a, uint16_t* b, std::string& err);
    bool rewind();
    ~PairFileReader() { if (f_) std::fclose(f_); }
private:
    FILE* f_ = nullptr;
    PairFileHeader hdr_{};
    std::vector<uint8_t> buf_;
};

}  // namespace trk
