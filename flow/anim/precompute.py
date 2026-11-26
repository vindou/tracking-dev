"""Data for the animations, from the real pipeline: one scene of flow/data/eval2 run through the trained flow
(flow/runs/rnode/model_fm.pt) and the verifier (flow/runs/verifier/model.pt), with the solver trajectories kept.

    .venv/bin/python -m flow.anim.precompute [--scene 1451]          # from the repository root; writes assets/scene.npz

The two stages are repeated here line by line from flow.infer_rnode.propose (same seeds, same random numbers) so that
the stage-1 trajectories can be stored; the result is checked against propose() itself. Nothing under flow/ is
modified. The animations read only assets/scene.npz (and need neither torch nor flow/data)."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .. import geom, infer, infer_rnode, rnode, scene, verifier as V
from ..make_data import load_eval

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def errs(q, p, T, rig):
    """Rotation (deg) and centroid (mm) error of poses (q, p = centroid in the rig frame) against the truth T."""
    c = T[:3, :3] @ rig.centroid + T[:3, 3]
    R = geom.qmat(torch.from_numpy(np.asarray(q, np.float64))).numpy()
    return np.array([geom.rot_angle_deg(r, T[:3, :3]) for r in R]), np.linalg.norm(p - c, axis=1) * 1000


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=1451)
    ap.add_argument("--eval", default="flow/data/eval2")
    ap.add_argument("--model", default="flow/runs/rnode/model_fm.pt")
    ap.add_argument("--verifier", default="flow/runs/verifier/model.pt")
    ap.add_argument("--min-score", type=float, default=0.85)
    ap.add_argument("--out", default=str(HERE / "assets/scene.npz"))
    a = ap.parse_args()
    dev = torch.device("cpu")
    net = infer_rnode.load_model(ROOT / a.model, dev)
    ver = V.load(ROOT / a.verifier, dev)
    table = torch.from_numpy(np.load(ROOT / "flow/data/rnode_table.npy")).to(dev)
    rig = scene.Rig()
    for i, cat, T, depth, labels in load_eval(ROOT / a.eval):
        if i == a.scene:
            break
    P, C, L = rig.cloud(depth, labels=labels)
    c_true = T[:3, :3] @ rig.centroid + T[:3, 3]
    q_true = geom.quat_from_R(T[:3, :3])
    print(f"scene {i} ({cat}): {len(P)} points, {int(L.sum())} on the object")

    # ---- the two stages, as infer_rnode.propose(seed = scene index, RK4 x 4)
    def run(solver, steps):
        rng = np.random.default_rng(i)
        g = torch.Generator(device=dev).manual_seed(i)
        S = infer.seeds_fps(P, 32, rng)

        def objfrac(X, c, ok, q, p):
            seg = rnode.field(net, table, X, c, q, p, 0.0)[3]
            return (((torch.sigmoid(seg) > 0.5) & ok).float().sum(1) / ok.float().sum(1).clamp_min(1)).cpu().numpy()

        X, c, ok = infer_rnode._sets(P, C, S, rng, dev, g)
        qb = geom.qrand(len(S), dev, g)
        q1, p1, tr1 = rnode.generate(net, table, X, c, qb, torch.zeros(len(S), 3), steps, trajectory=True, solver=solver)
        of1 = objfrac(X, c, ok, q1, p1)
        e = p1.numpy().astype(np.float64) + S
        clusters = []
        for j in np.argsort(-of1):
            for cl in clusters:
                if np.linalg.norm(e[j] - e[cl[0]]) < 0.03:
                    cl.append(j)
                    break
            else:
                clusters.append([j])
        clusters.sort(key=lambda cl: -sum(of1[j] for j in cl))
        cen = np.stack([np.average(e[cl], axis=0, weights=of1[cl] + 1e-3) for cl in clusters[:4]])
        Q0, cover = infer.orientations(32)
        X, c, ok = infer_rnode._sets(P, C, cen, rng, dev, g)
        rep = np.repeat(np.arange(len(cen)), 32)
        qb = torch.from_numpy(np.tile(Q0, (len(cen), 1)).astype(np.float32))
        qe, pe, tr2 = rnode.generate(net, table, X[rep], c[rep], qb, torch.zeros(len(rep), 3), steps, trajectory=True, solver=solver)
        qi, pi = V.icp(table, X[rep], qe, pe)
        score = torch.sigmoid(ver(table, X[rep], c[rep], qi, pi)).numpy()
        of_flow = objfrac(X[rep], c[rep], ok[rep], qe, pe)
        of_icp = objfrac(X[rep], c[rep], ok[rep], qi, pi)
        cl_id = np.full(len(S), -1)
        for k, cl in enumerate(clusters):
            cl_id[cl] = k
        return dict(S=S, tr1_q=np.stack([x.numpy() for x, _ in tr1]), tr1_p=np.stack([y.numpy() for _, y in tr1]) + S[None],
                    of1=of1, cl_id=cl_id, cen=cen, Q0=Q0, cover=cover, rep=rep,
                    tr2_q=np.stack([x.numpy() for x, _ in tr2]), tr2_p=np.stack([y.numpy() for _, y in tr2]) + cen[rep][None],
                    q_flow=qe.numpy().astype(np.float64), p_flow=pe.numpy().astype(np.float64) + cen[rep],
                    q_icp=qi.numpy().astype(np.float64), p_icp=pi.numpy().astype(np.float64) + cen[rep],
                    score=score, of_flow=of_flow, of_icp=of_icp, X=X, c=c, ok=ok)

    r = run("rk4", 4)
    # Euler x 8 from the same stage-2 starts (same crops, same 32 orientations at the same 4 positions)
    qb = torch.from_numpy(np.tile(r["Q0"], (len(r["cen"]), 1)).astype(np.float32))
    _, _, tre = rnode.generate(net, table, r["X"][r["rep"]], r["c"][r["rep"]], qb, torch.zeros(len(r["rep"]), 3), 8, trajectory=True, solver="euler")
    eu_q = np.stack([x.numpy() for x, _ in tre])
    eu_p = np.stack([y.numpy() for _, y in tre]) + r["cen"][r["rep"]][None]
    dq_e = np.degrees(2 * np.arccos(np.clip(np.abs((eu_q[-1] * r["tr2_q"][-1]).sum(-1)), 0, 1)))
    dp_e = np.linalg.norm(eu_p[-1] - r["tr2_p"][-1], axis=1) * 1000
    print(f"Euler x 8 against RK4 x 4, same starts: end poses differ by {np.median(dq_e):.1f} deg / {np.median(dp_e):.1f} mm (median), "
          f"{(dq_e < 10).mean() * 100:.0f} % within 10 deg")
    print("stage 1: seeds per group (largest first):", sorted(np.bincount(r["cl_id"]).tolist(), reverse=True)[:8],
          "; kept groups have", [int((r["cl_id"] == k).sum()) for k in range(len(r["cen"]))], "seeds")

    # ---- check against the real function, without and with the verifier
    h0 = infer_rnode.propose(net, table, rig, depth, dev, seed=i)
    h1 = infer_rnode.propose(net, table, rig, depth, dev, seed=i, verifier=ver, min_score=a.min_score)
    assert np.allclose(h0["q"], r["q_flow"], atol=1e-5) and np.allclose(h0["p"], r["p_flow"], atol=1e-6), "stage 2 differs from propose()"
    assert np.allclose(h1["q"], r["q_icp"], atol=1e-5) and np.allclose(h1["score"], r["score"], atol=1e-5), "verifier path differs"
    sel0, sel1 = infer.hint_order(h0, 16), infer.hint_order(h1, 16)
    grp0 = np.full(128, -1)
    for k, gmem in enumerate(h0["groups"]):
        grp0[gmem] = k
    grp1 = np.full(128, -1)
    for k, gmem in enumerate(h1["groups"]):
        grp1[gmem] = k
    er0, ep0 = errs(r["q_flow"], r["p_flow"], T, rig)
    er1, ep1 = errs(r["q_icp"], r["p_icp"], T, rig)
    print("stage 1: clusters kept at", np.round(np.linalg.norm(r["cen"] - c_true, axis=1) * 1000), "mm from the truth")
    print(f"stage 2: {len(h0['groups'])} groups; hints without verifier {len(sel0)}, with verifier >= {a.min_score}: {len(sel1)}")
    print("  without verifier (order sent):", [(j, round(er0[j]), round(ep0[j]), round(float(r['score'][j]), 2)) for j in sel0])
    print("  verifier order, all groups:   ", [(j, round(er1[j]), round(ep1[j]), round(float(r['score'][j]), 2))
                                              for j in sorted((g[0] for g in h1['groups']), key=lambda j: -r['score'][j])][:20])

    # ---- what the tracker did with these hypotheses (result files of the evaluation in flow/README.md)
    def tracker(folder, mode):
        row = [x for x in csv.DictReader(open(ROOT / folder)) if int(x["scene"]) == i and x["mode"] == mode][0]
        return np.array([float(row[k]) for k in ("valid", "by_hint", "px", "py", "pz", "qw", "qx", "qy", "qz", "err_mm", "err_deg", "hints_tried", "cycles")])
    trk = {k: tracker(f, m) for k, f, m in (("classical", "flow/data/eval2/baseline.csv", "baseline"),
                                            ("flow", "flow/data/eval_e2_base/hints.csv", "hints"),
                                            ("ver", "flow/data/eval_e2_ver85/hints.csv", "hints"))}
    for k, v in trk.items():
        print(f"tracker, {k}: valid {int(v[0])} by_hint {int(v[1])} err {v[9]:.1f} mm {v[10]:.1f} deg, hints tried {int(v[11])}, cycles {int(v[12])}")

    # ---- scene "what the network sees": one stage-2 trajectory that ends at the truth, made dense, with the table
    # lookups (closest surface point, normal, residual) of its 512 points at every pose, and the object / clutter output
    good = np.flatnonzero((er0 < 10) & (ep0 < 10))
    ang0 = np.degrees(2 * np.arccos(np.clip(np.abs(r["tr2_q"][0] @ q_true), 0, 1)))
    assert len(good), "no hypothesis of this scene ends at the truth"
    j = good[np.argmin(np.abs(ang0[good] - 100))]            # a start about 100 degrees from the truth
    ci = r["rep"][j]
    X, cc, ok = r["X"][ci:ci + 1], r["c"][ci:ci + 1], r["ok"][ci, :].numpy()
    # labels of the 512 points: nearest point of the labelled cloud (the same points, selected by select_points)
    from scipy.spatial import cKDTree
    lab = L[cKDTree(P).query(X[0].numpy() + r["cen"][ci])[1]]
    dense_q, dense_p = [], []
    tq, tp = torch.from_numpy(r["tr2_q"][:, j]), torch.from_numpy(r["tr2_p"][:, j] - r["cen"][ci]).float()
    for k in range(len(tq) - 1):
        eta = geom.qlog(geom.qmul(geom.qconj(tq[k:k + 1]), tq[k + 1:k + 2]))
        for s in np.linspace(0, 1, 16, endpoint=k == len(tq) - 2):
            dense_q.append(geom.qmul(tq[k:k + 1], rnode.qexp(s * eta))[0])
            dense_p.append((1 - s) * tp[k] + s * tp[k + 1])
    dq, dp = torch.stack(dense_q).float(), torch.stack(dense_p).float()
    xm = (X - dp[:, None]) @ geom.qmat(dq)                                   # [K, 512, 3] body frame
    e = rnode.lookup(table, xm)
    feat = rnode.features(table, xm, cc.expand(len(dq), -1))
    seg = torch.sigmoid(rnode.field(net, table, X.expand(len(dq), -1, -1), cc.expand(len(dq), -1), dq, dp, 0.0)[3])
    print(f"feature scene: hypothesis {j}, start {ang0[j]:.0f} deg from the truth, end {er0[j]:.1f} deg {ep0[j]:.1f} mm; "
          f"{int(lab[ok].sum())} of {int(ok.sum())} points on the object; seg accuracy at the end {((seg[-1].numpy() > 0.5) == lab)[ok].mean():.2f}")

    # points for display only: up to 900 of the object and 3600 of the clutter (the networks used the full cloud)
    rs = np.random.default_rng(0)
    io, ic = np.flatnonzero(L == 1), np.flatnonzero(L == 0)
    keep = np.concatenate([rs.permutation(io)[:900], rs.permutation(ic)[:3600]])
    np.savez_compressed(
        a.out, scene=i, cat=cat, min_score=a.min_score,
        V=rig.V, F=rig.F, centroid=rig.centroid, radius=rig.radius, T_true=T, q_true=q_true, c_true=c_true,
        T_rig_cam=np.stack(rig.T_rig_cam), wv_min=rig.wv_min, wv_max=rig.wv_max,
        P=P[keep], C=C[keep], L=L[keep], n_cloud=len(P),
        seeds=r["S"], tr1_p=r["tr1_p"], tr1_q=r["tr1_q"], of1=r["of1"], cl_id=r["cl_id"], cen=r["cen"],
        Q0=r["Q0"], cover=r["cover"], rep=r["rep"], tr2_q=r["tr2_q"], tr2_p=r["tr2_p"],
        eu_q=eu_q, eu_p=eu_p, n_obj=int(L.sum()),
        q_flow=r["q_flow"], p_flow=r["p_flow"], q_icp=r["q_icp"], p_icp=r["p_icp"], score=r["score"],
        of_flow=r["of_flow"], of_icp=r["of_icp"], er_flow=er0, ep_flow=ep0, er_icp=er1, ep_icp=ep1,
        grp0=grp0, grp1=grp1, sel0=np.array(sel0), sel1=np.array(sel1),
        trk_classical=trk["classical"], trk_flow=trk["flow"], trk_ver=trk["ver"],
        f_j=j, f_seed=r["cen"][ci], f_X=X[0].numpy(), f_cam=cc[0].numpy(), f_ok=ok, f_lab=lab,
        f_q=dq.numpy(), f_p=dp.numpy(), f_xm=xm.numpy().astype(np.float32), f_cp=e[..., :3].numpy().astype(np.float32),
        f_n=e[..., 3:].numpy().astype(np.float32), f_r=feat[..., 10].numpy().astype(np.float32), f_seg=seg.numpy().astype(np.float32))
    print("wrote", a.out, f"{Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
