// trk_bridge: telemetry bridge. Reads the pose buffer and the cloud buffer and sends them to one TCP client
// (TCP_NODELAY) as fixed-size messages (README.md "Network protocol").
//
//   trk_bridge [--config config/tracker.json] [--listen ADDRESS] [--port N] [--duration S] [--final-json FILE]
//
// Threads: reader (waits on the pose-buffer notification, checks the cloud buffer, appends messages to the send queue;
// never waits for the network), network (accepts one client at a time, non-blocking sends from the queue), main
// (SIGINT / SIGTERM). Send queue limited to queue_bytes: when full, the oldest cloud messages are discarded first,
// then the oldest pose messages; both are counted. Without a client the queue is not filled. The bridge follows tracker
// restarts (attach again by name) and accepts a new client at any time.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/clock.hpp"
#include "common/no_thp.hpp"
#include "core/config.hpp"
#include "records/records.hpp"
#include "shmring/shm_ring.hpp"
#include "core/send_queue.hpp"

using namespace trk;

namespace {
std::atomic<bool> g_stop{false};

void to_net_cloud(const CloudRecord& c, NetCloud& n) {
    std::memset(&n, 0, sizeof n);
    n.version = 1;
    n.pose_seq = c.pose_seq;
    n.pair_seq = c.pair_seq;
    double s[3] = {0, 0, 0};
    const uint32_t cnt = std::min(c.point_count, kCloudMaxPoints);
    for (uint32_t i = 0; i < cnt; i++) { s[0] += c.points[i].x; s[1] += c.points[i].y; s[2] += c.points[i].z; }
    for (int k = 0; k < 3; k++) n.center[k] = cnt ? s[k] / cnt : 0.0;
    uint32_t out = 0;
    for (uint32_t i = 0; i < cnt; i++) {
        const double d[3] = {c.points[i].x - n.center[0], c.points[i].y - n.center[1], c.points[i].z - n.center[2]};
        if (std::fabs(d[0]) > 3.2767 || std::fabs(d[1]) > 3.2767 || std::fabs(d[2]) > 3.2767) continue;   // outside the 16-bit range
        NetCloudPoint& p = n.points[out++];
        p.x = int16_t(std::lround(d[0] * 1e4));
        p.y = int16_t(std::lround(d[1] * 1e4));
        p.z = int16_t(std::lround(d[2] * 1e4));
        p.inlier = c.points[i].inlier;
        p.camera = c.points[i].camera;
    }
    n.point_count = out;
}
}  // namespace

int main(int argc, char** argv) {
    trk::disable_transparent_huge_pages();   // before any allocation (common/no_thp.hpp)
    sigset_t sigs;
    sigemptyset(&sigs);
    sigaddset(&sigs, SIGINT);
    sigaddset(&sigs, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigs, nullptr);
    signal(SIGPIPE, SIG_IGN);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string cfgp = "config/tracker.json", final_json, listen_addr;
    int port = -1;
    double duration = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--config") cfgp = argv[i + 1];
        else if (a == "--listen") listen_addr = argv[i + 1];
        else if (a == "--port") port = std::atoi(argv[i + 1]);
        else if (a == "--duration") duration = std::atof(argv[i + 1]);
        else if (a == "--final-json") final_json = argv[i + 1];
        else { std::fprintf(stderr, "usage: trk_bridge [--config F] [--listen ADDR] [--port N] [--duration S] [--final-json F]\n"); return 2; }
    }
    TrackerConfig cfg;
    std::string err;
    if (!load_tracker_config(cfgp, cfg, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 2; }
    if (listen_addr.empty()) listen_addr = cfg.bridge_address;
    if (port < 0) port = cfg.bridge_port;

    const int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (inet_pton(AF_INET, listen_addr.c_str(), &addr.sin_addr) != 1 || bind(ls, (sockaddr*)&addr, sizeof addr) != 0 || listen(ls, 4) != 0) {
        std::fprintf(stderr, "error: cannot listen on %s:%d: %s\n", listen_addr.c_str(), port, std::strerror(errno));
        return 3;
    }
    fcntl(ls, F_SETFL, O_NONBLOCK);
    std::printf("bridge: listening on %s:%d, pose buffer %s, cloud buffer %s, queue %ld bytes\n", listen_addr.c_str(), port,
                cfg.pose_buffer.c_str(), cfg.cloud_buffer.c_str(), cfg.bridge_queue_bytes);

    SendQueue q(cfg.bridge_queue_bytes);
    std::atomic<uint64_t> poses_read{0}, clouds_read{0}, poses_sent{0}, clouds_sent{0}, clients{0}, reattaches{0};
    std::atomic<bool> attached{false};
    std::atomic<uint64_t> tracker_instance{0};

    std::thread reader([&] {
        pthread_setname_np(pthread_self(), "bridge-reader");
        std::unique_ptr<RingReader> pr, cr;
        std::unique_ptr<PoseRecord> pose(new PoseRecord());
        std::unique_ptr<CloudRecord> cloud(new CloudRecord());
        std::unique_ptr<NetCloud> ncloud(new NetCloud());
        int64_t next_status = mono_ns();
        while (!g_stop.load()) {
            std::string e;
            if (!pr) {   // attach (retry every 100 ms while there is no tracker)
                pr = RingReader::attach(cfg.pose_buffer, kPoseRecordType, kPoseRecordVersion, sizeof(PoseRecord), e);
                if (!pr) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
                cr = RingReader::attach(cfg.cloud_buffer, kCloudRecordType, kCloudRecordVersion, sizeof(CloudRecord), e);
                attached.store(true);
                tracker_instance.store(pr->instance_id());
                reattaches.fetch_add(1);
                std::printf("[attach] pose buffer writer instance %016" PRIx64 "\n", pr->instance_id());
            }
            const auto res = pr->next(pose.get(), 100000000LL);
            if (res == RingReader::Result::Ok) {
                poses_read.fetch_add(1);
                q.push(kNetPose, pose.get());
                if (cr && cr->latest(cloud.get()) == RingReader::Result::Ok) {
                    clouds_read.fetch_add(1);
                    to_net_cloud(*cloud, *ncloud);
                    q.push(kNetCloud, ncloud.get());
                }
            } else if (res != RingReader::Result::Timeout) {
                std::printf("[detach] pose buffer: %s\n", RingReader::result_name(res));
                attached.store(false);
                // follow the tracker: attach again to the next instance (same name)
                while (!g_stop.load()) {
                    if (pr->reattach(e)) {
                        cr = RingReader::attach(cfg.cloud_buffer, kCloudRecordType, kCloudRecordVersion, sizeof(CloudRecord), e);
                        attached.store(true);
                        tracker_instance.store(pr->instance_id());
                        reattaches.fetch_add(1);
                        std::printf("[attach] pose buffer writer instance %016" PRIx64 "\n", pr->instance_id());
                        break;
                    }
                    if (res == RingReader::Result::WriterNotAlive && pr->writer_alive()) { attached.store(true); break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            const int64_t now = mono_ns();
            if (now >= next_status) {
                next_status = now + int64_t(cfg.bridge_status_s * 1e9);
                NetStatus st{};
                st.version = 1;
                st.tracker_attached = attached.load() ? 1 : 0;
                st.poses_read = poses_read.load();
                st.clouds_read = clouds_read.load();
                st.poses_dropped = q.dropped_pose();
                st.clouds_dropped = q.dropped_cloud();
                st.tracker_instance_id = tracker_instance.load();
                st.bridge_time_mono_ns = now;
                q.push(kNetStatus, &st);
            }
        }
    });

    std::thread net([&] {
        pthread_setname_np(pthread_self(), "bridge-net");
        int cs = -1;
        std::vector<uint8_t> cur(SendQueue::kMaxMsg, 0);   // message being sent (allocated once)
        uint32_t cur_size = 0;
        uint16_t cur_type = 0;
        std::size_t off = 0;
        bool have = false;
        while (!g_stop.load()) {
            // accept: one client at a time; a second client is closed at once
            sockaddr_in ca{};
            socklen_t cl = sizeof ca;
            const int ns = accept(ls, (sockaddr*)&ca, &cl);
            if (ns >= 0) {
                if (cs >= 0) {
                    close(ns);
                } else {
                    cs = ns;
                    fcntl(cs, F_SETFL, O_NONBLOCK);
                    setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                    char ip[64];
                    inet_ntop(AF_INET, &ca.sin_addr, ip, sizeof ip);
                    clients.fetch_add(1);
                    std::printf("[client] connected from %s:%d\n", ip, ntohs(ca.sin_port));
                    have = false;
                    q.set_client(true);
                }
            }
            if (cs < 0) {
                pollfd pf{ls, POLLIN, 0};
                poll(&pf, 1, 100);
                continue;
            }
            if (!have) {
                cur_size = q.pop(cur.data(), cur_type);
                have = cur_size > 0;
                off = 0;
            }
            if (!have) {
                pollfd pf[2] = {{ls, POLLIN, 0}, {cs, POLLIN, 0}};
                poll(pf, 2, 5);
                if (pf[1].revents & (POLLIN | POLLHUP | POLLERR)) {   // client closed (it never sends)
                    char b[64];
                    const ssize_t r = recv(cs, b, sizeof b, MSG_DONTWAIT);
                    if (r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                        std::printf("[client] disconnected\n");
                        q.set_client(false);
                        close(cs);
                        cs = -1;
                    }
                }
                continue;
            }
            const ssize_t n = send(cs, cur.data() + off, cur_size - off, MSG_DONTWAIT | MSG_NOSIGNAL);
            if (n > 0) {
                off += std::size_t(n);
                if (off == cur_size) {
                    have = false;
                    if (cur_type == kNetPose) poses_sent.fetch_add(1);
                    else if (cur_type == kNetCloud) clouds_sent.fetch_add(1);
                }
            } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                pollfd pf{cs, POLLOUT, 0};
                poll(&pf, 1, 20);
            } else {
                std::printf("[client] disconnected (%s)\n", n == 0 ? "closed" : std::strerror(errno));
                q.set_client(false);
                close(cs);
                cs = -1;
                have = false;
            }
        }
        if (cs >= 0) close(cs);
    });

    const int64_t t0 = mono_ns();
    int64_t next_print = t0 + 10000000000LL;
    for (;;) {
        timespec ts{0, 100000000};
        const int s = sigtimedwait(&sigs, nullptr, &ts);
        if (s == SIGINT || s == SIGTERM) break;
        if (duration > 0 && double(mono_ns() - t0) / 1e9 >= duration) break;
        if (mono_ns() >= next_print) {
            next_print += 10000000000LL;
            std::printf("[status +%.0fs] tracker=%s poses read/sent/dropped=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                        " clouds read/sent/dropped=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " clients=%" PRIu64 " attaches=%" PRIu64 "\n",
                        double(mono_ns() - t0) / 1e9, attached.load() ? "attached" : "none", poses_read.load(),
                        poses_sent.load(), q.dropped_pose(), clouds_read.load(), clouds_sent.load(), q.dropped_cloud(),
                        clients.load(), reattaches.load());
        }
    }
    g_stop.store(true);
    reader.join();
    net.join();
    close(ls);
    std::printf("final: poses read/sent/dropped=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " clouds read/sent/dropped=%" PRIu64
                "/%" PRIu64 "/%" PRIu64 " clients=%" PRIu64 " attaches=%" PRIu64 "\n",
                poses_read.load(), poses_sent.load(), q.dropped_pose(), clouds_read.load(), clouds_sent.load(),
                q.dropped_cloud(), clients.load(), reattaches.load());
    if (!final_json.empty())
        if (FILE* f = std::fopen(final_json.c_str(), "w")) {
            std::fprintf(f, "{\"poses_read\": %" PRIu64 ", \"poses_sent\": %" PRIu64 ", \"poses_dropped\": %" PRIu64
                         ", \"clouds_read\": %" PRIu64 ", \"clouds_sent\": %" PRIu64 ", \"clouds_dropped\": %" PRIu64
                         ", \"clients\": %" PRIu64 ", \"attaches\": %" PRIu64 "}\n",
                         poses_read.load(), poses_sent.load(), q.dropped_pose(), clouds_read.load(), clouds_sent.load(),
                         q.dropped_cloud(), clients.load(), reattaches.load());
            std::fclose(f);
        }
    return 0;
}
