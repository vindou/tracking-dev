#include "pairfile.hpp"

#include <cstring>

#include "../common/clock.hpp"

namespace trk {

std::size_t rle_encode(const uint16_t* img, std::size_t n, std::vector<uint8_t>& out) {
    out.clear();
    std::size_t i = 0;
    auto put16 = [&](uint16_t v) { out.push_back(uint8_t(v & 0xFF)); out.push_back(uint8_t(v >> 8)); };
    while (i < n) {
        std::size_t z = 0;
        while (i + z < n && img[i + z] == 0 && z < 65535) z++;
        i += z;
        std::size_t l = 0;
        while (i + l < n && img[i + l] != 0 && l < 65535) l++;
        put16(uint16_t(z));
        put16(uint16_t(l));
        for (std::size_t k = 0; k < l; k++) put16(img[i + k]);
        i += l;
    }
    return out.size();
}

bool rle_decode(const uint8_t* in, std::size_t bytes, uint16_t* img, std::size_t n) {
    std::size_t p = 0, i = 0;
    auto get16 = [&](uint16_t& v) {
        if (p + 2 > bytes) return false;
        v = uint16_t(in[p] | (in[p + 1] << 8));
        p += 2;
        return true;
    };
    while (p < bytes) {
        uint16_t z, l;
        if (!get16(z) || !get16(l) || i + z + l > n) return false;
        std::memset(img + i, 0, std::size_t(z) * 2);
        i += z;
        for (uint16_t k = 0; k < l; k++) {
            uint16_t v;
            if (!get16(v)) return false;
            img[i++] = v;
        }
    }
    return i == n;
}

bool PairFileWriter::open(const std::string& path, const cap::CameraInfo cams[2], uint32_t source, double period_ms,
                          double tolerance_ms, std::string& err) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) { err = "cannot write " + path; return false; }
    std::memset(&hdr_, 0, sizeof hdr_);
    std::memcpy(hdr_.magic, "TRKPAIRS", 8);
    hdr_.version = kPairFileVersion;
    hdr_.header_size = sizeof(PairFileHeader);
    hdr_.width = cams[0].width ? cams[0].width : cap::kWidth;
    hdr_.height = cams[0].height ? cams[0].height : cap::kHeight;
    hdr_.bytes_per_pixel = 2;
    hdr_.source = source;
    hdr_.created_real_ns = real_ns();
    hdr_.frame_period_ms = period_ms;
    hdr_.tolerance_ms = tolerance_ms;
    hdr_.cam[0] = cams[0];
    hdr_.cam[1] = cams[1];
    bytes_ = sizeof hdr_;
    return std::fwrite(&hdr_, sizeof hdr_, 1, f_) == 1;
}

bool PairFileWriter::append(const cap::PairRecord& rec, const uint16_t* a, const uint16_t* b, const double* truth16) {
    if (!f_) return false;
    const std::size_t n = std::size_t(hdr_.width) * hdr_.height;
    PairChunkHeader ch;
    std::memset(&ch, 0, sizeof ch);
    ch.magic = kPairChunkMagic;
    ch.coded_a = uint32_t(rle_encode(a, n, ca_));
    ch.coded_b = uint32_t(rle_encode(b, n, cb_));
    ch.rec = rec;
    if (truth16) {
        ch.flags |= 1;
        std::memcpy(ch.truth, truth16, sizeof ch.truth);
    }
    const bool ok = std::fwrite(&ch, sizeof ch, 1, f_) == 1 && std::fwrite(ca_.data(), 1, ca_.size(), f_) == ca_.size() &&
                    std::fwrite(cb_.data(), 1, cb_.size(), f_) == cb_.size();
    if (ok) {
        hdr_.pair_count++;
        bytes_ += sizeof ch + ca_.size() + cb_.size();
    }
    return ok;
}

bool PairFileWriter::close() {
    if (!f_) return true;
    bool ok = std::fseek(f_, 0, SEEK_SET) == 0 && std::fwrite(&hdr_, sizeof hdr_, 1, f_) == 1;
    ok = std::fclose(f_) == 0 && ok;
    f_ = nullptr;
    return ok;
}

bool PairFileReader::open(const std::string& path, std::string& err) {
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) { err = "cannot read " + path; return false; }
    if (std::fread(&hdr_, sizeof hdr_, 1, f_) != 1 || std::memcmp(hdr_.magic, "TRKPAIRS", 8) != 0) {
        err = path + ": not a pair file";
        return false;
    }
    if (hdr_.version != kPairFileVersion || hdr_.header_size != sizeof hdr_ || hdr_.bytes_per_pixel != 2) {
        err = path + ": pair file version " + std::to_string(hdr_.version) + ", this program reads version 1";
        return false;
    }
    return true;
}

bool PairFileReader::rewind() { return f_ && std::fseek(f_, long(sizeof hdr_), SEEK_SET) == 0; }

bool PairFileReader::next(PairChunkHeader& ch, uint16_t* a, uint16_t* b, std::string& err) {
    if (!f_) return false;
    if (std::fread(&ch, sizeof ch, 1, f_) != 1) return false;   // end
    if (ch.magic != kPairChunkMagic) { err = "bad chunk"; return false; }
    const std::size_t n = std::size_t(hdr_.width) * hdr_.height;
    for (int k = 0; k < 2; k++) {
        const uint32_t sz = k ? ch.coded_b : ch.coded_a;
        buf_.resize(sz);
        if (std::fread(buf_.data(), 1, sz, f_) != sz) { err = "truncated pair"; return false; }
        if (!rle_decode(buf_.data(), sz, k ? b : a, n)) { err = "bad image coding"; return false; }
    }
    return true;
}

}  // namespace trk
