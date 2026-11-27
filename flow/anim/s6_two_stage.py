"""Scene 6: the two-stage inference of flow/infer_rnode.py on one synthetic scene: 32 seeds -> positions (grouped at
30 mm, the 4 groups with the most object points kept) -> 32 orientations per position -> poses -> groups (15 mm, 15
degrees) -> up to 16 hypotheses. All positions and poses are the stored outputs of the trained flow."""
from style import *


class TwoStageInference(Scene):
    def construct(self):
        d, m = data(), mesh()
        self.add(title("Two-stage inference", "positions first, then orientations at the best positions", n=6))
        self.add(note(SYNTH, 15))
        cap = Caption(self)
        P, col = d["P"], label_colors(d["L"])
        S, t1 = d["seeds"], d["tr1_p"]
        cen, rep, tq, tp = d["cen"], d["rep"], d["tr2_q"], d["tr2_p"]
        ci = int(np.argmin(np.linalg.norm(cen - d["c_true"], axis=1)))
        wide = dict(center=np.array([0.33, -0.03, 0.52]), scale=8.6, at=np.array([-2.3, -0.75, 0.0]))
        zoom = dict(center=cen[ci], scale=21.0, at=np.array([-4.2, -0.2, 0.0]))
        v = View(wide["center"], az=28, el=32, scale=wide["scale"], at=wide["at"])
        v.clip = (-7.2, 3.6, -3.2, 2.8)
        z = ValueTracker(0.0)

        def set_view(mob=None):
            a = z.get_value()
            v.center = (1 - a) * wide["center"] + a * zoom["center"]
            v.scale = float(np.exp((1 - a) * np.log(wide["scale"]) + a * np.log(zoom["scale"])))
            v.at = (1 - a) * wide["at"] + a * zoom["at"]
            v.clip = (-7.2, 3.6 - 4.6 * a, -3.2, 2.8)
        driver = Mobject().add_updater(set_view)
        self.add(driver)
        cl = live_cloud(lambda: cloud(v, P, col, 2.6, 0.85))
        truth = always_redraw(lambda: m.mob(v, d["q_true"], d["c_true"], C_TRUE, fill=0.0, edge_width=2.2))
        cams = VGroup(*[camera_glyph(v, Tc) for Tc in d["T_rig_cam"]])
        leg = legend([(C_OBJ, "object points"), (C_CLUT, "clutter points"), (C_TRUE, "true pose", "box"),
                      (C_SEED, "seeds and position candidates"), (C_EST, "pose hypotheses (dots: tips of the x and y axes)")], 15)
        leg.to_corner(UR, buff=0.3).shift(DOWN * 0.75)
        self.play(FadeIn(cl), FadeIn(truth), FadeIn(cams), FadeIn(leg))
        cap.show(f"One depth pair: {int(d['n_cloud']):,} points, {int(d['n_obj']):,} of them on the object (a sample is drawn). Where to start?", 3)

        # ---- stage 1
        s1 = ValueTracker(0.0)
        K1 = t1.shape[0] - 1

        def seed_pos(j):
            u = s1.get_value() * K1
            k = min(int(u), K1 - 1)
            return (1 - (u - k)) * t1[k, j] + (u - k) * t1[k + 1, j]
        alive = np.ones(len(S))
        seeds = always_redraw(lambda: VGroup(*[VGroup(Line(v(S[j]), v(seed_pos(j)), stroke_width=1.6, color=C_SEED, stroke_opacity=0.6 * alive[j]),
                                                       Dot(v(seed_pos(j)), radius=0.07, color=C_SEED, fill_opacity=alive[j])) for j in range(len(S))]))
        self.play(FadeIn(seeds), FadeOut(cams))
        cap.show("Stage 1, position. 32 seeds spread over the cloud by farthest-point selection.", 2.5)
        j0 = int(np.argmin(np.linalg.norm(S - d["c_true"], axis=1) + 1e3 * (d["cl_id"] != d["cl_id"][np.argmin(np.linalg.norm(t1[-1] - d["c_true"], axis=1))])))
        crop = ring(v, S[j0], 0.16, C_SEED, 2, 0.9)
        crop_l = T("0.16 m", 15, C_SEED).next_to(crop, UP, buff=0.06)
        self.play(Create(crop), FadeIn(crop_l))
        cap.show("Each seed gets 512 points from within 0.16 m and a random orientation. The flow is run from there.", 3)
        self.play(FadeOut(crop), FadeOut(crop_l))
        self.play(s1.animate.set_value(1.0), run_time=4, rate_func=smooth)
        cap.show("Each seed moves to where the network puts the model centroid. Seeds that see the object agree.", 2.5)
        # groups (30 mm) and the 4 kept
        n_cl = int(d["cl_id"].max()) + 1
        kept = len(cen)
        size = np.bincount(d["cl_id"], minlength=n_cl)
        rings = VGroup(*[ring(v, cen[k], 0.03, C_SEED, 3) for k in range(kept)])
        nums = VGroup(*[T(str(k + 1), 15, C_SEED, weight=BOLD).next_to(rings[k], UP if k != 3 else DOWN, buff=0.05) for k in range(kept)])
        rows = [T("kept groups (ranked by summed object share)", 15, C_SEED, weight=BOLD)]
        for k in range(kept):
            of = d["of1"][d["cl_id"] == k]
            rows.append(T(f"{k + 1}:  {size[k]} seeds,  {of.mean() * 100:.0f} % of their points labelled object", 15))
        tabl = VGroup(*rows).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        labs = VGroup(nums, tabl.next_to(leg, DOWN, buff=0.3).align_to(leg, RIGHT))
        self.play(Create(rings), FadeIn(labs))
        cap.show(f"End points within 30 mm are grouped ({n_cl} groups). The 4 with the most points labelled object by the network are kept.", 2)
        alive[:] = (d["cl_id"] < kept) * 1.0
        self.wait(2.5)
        cdots = VGroup(*[Dot(v(c), radius=0.08, color=C_SEED) for c in cen])
        self.play(FadeOut(seeds), FadeOut(labs), FadeIn(cdots))
        far = np.linalg.norm(cen - d["c_true"], axis=1) * 1000
        cap.show(f"One of the four is {far[ci]:.0f} mm from the true centroid. The other three sit on clutter, {np.delete(far, ci).min():.0f} mm or more away.", 3.5)

        # ---- stage 2
        s2 = ValueTracker(0.0)
        vis = np.ones(len(rep))
        hyp = always_redraw(lambda: VGroup(*[spike(v, *path_pose(tq[:, j], tp[:, j], s2.get_value()), length=0.085 - 0.02 * z.get_value(), opacity=0.9)
                                             for j in range(len(rep)) if vis[j]]))
        self.play(FadeIn(hyp), FadeOut(rings), FadeOut(cdots))
        cap.show(f"Stage 2, pose. At each position: 32 orientations spread over SO(3). Any orientation is within {float(d['cover']):.0f}° of one of them.", 4)
        self.play(s2.animate.set_value(1.0), run_time=5, rate_func=smooth)
        cap.show("The flow moves all 4 × 32 = 128 hypotheses (RK4 × 4: 16 network evaluations of the batch).", 2.5)
        vis[:] = rep == ci
        self.play(z.animate.set_value(1.0), FadeOut(leg), run_time=2.5)
        s2.set_value(0.0)
        cap.show("Closer, at the position on the object: the 32 orientation starts again ...", 1.5)
        self.play(s2.animate.set_value(1.0), run_time=4.5, rate_func=smooth)

        # ---- grouping
        ng = int(d["grp0"].max()) + 1
        sel = d["sel0"]
        here = [int(j) for j in sel if rep[j] == ci]
        good = (d["er_flow"] < 10) & (d["ep_flow"] < 10)
        flowc = VGroup(
            T("128 hypotheses", 19), T("↓  group: within 15 mm and 15°", 15, C_DIM), T(f"{ng} groups", 19),
            T("↓  best member of the 16 largest groups", 15, C_DIM), T(f"{len(sel)} hypotheses to the tracker", 19, C_EST, weight=BOLD),
            T(f"{int(good[sel].sum())} of them within 10° and 10 mm of the truth", 15, C_DIM),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([2.3, 1.75, 0])
        # the hypotheses sent from this position, each drawn alone, next to the truth
        cells = [np.array([0.6 + 1.8 * (i % 4), -0.45 - 1.6 * (i // 4), 0]) for i in range(8)]
        thumbs = VGroup()
        vt = View(d["c_true"], az=v.az, el=v.el, scale=10.0, at=cells[0])
        thumbs.add(m.mob(vt, d["q_true"], d["c_true"], C_TRUE, fill=0.85, edge_width=1.5), T("truth", 14, C_TRUE).move_to(cells[0] + DOWN * 0.75))
        for i, j in enumerate(here[:7]):
            vt = View(d["p_flow"][j], az=v.az, el=v.el, scale=10.0, at=cells[i + 1])
            thumbs.add(m.mob(vt, d["q_flow"][j], d["p_flow"][j], C_EST, fill=0.3, edge_width=1.5),
                       T(f"{d['er_flow'][j]:.0f}°, {d['ep_flow'][j]:.0f} mm", 14, C_TRUE if good[j] else C_DIM).move_to(cells[i + 1] + DOWN * 0.75))
        thl = T(f"the {len(here[:7])} sent from this position, with their error to the truth", 15, C_DIM).move_to([3.3, 0.5, 0])
        outl = VGroup(thumbs, thl)
        self.play(hyp.animate.set_opacity(0.35), FadeIn(outl), FadeIn(flowc))
        cap.show("Hypotheses that agree are grouped; one per group is handed on. Most are wrong: same place, another orientation, or on clutter.", 8)
