"""Scene 5: solving the ODE. The 32 orientation starts at one position of stage 2, as stored solver states of the
trained flow (assets/scene.npz: tr2_* RK4 x 4, eu_* Euler x 8 from the same starts), and the cost in network
evaluations (rnode.SOLVER_EVALS)."""
from style import *

C_RK, C_EU = C_NET, C_COND


class SolvingODE(Scene):
    def construct(self):
        d, m = data(), mesh()
        self.add(title("Solving the ODE", "a few fixed steps; cost counted in network evaluations", n=5))
        self.add(note(SYNTH, 15))
        cap = Caption(self)
        ci = int(np.argmin(np.linalg.norm(d["cen"] - d["c_true"], axis=1)))
        idx = np.flatnonzero(d["rep"] == ci)
        tq, tp, eq_, ep_ = d["tr2_q"], d["tr2_p"], d["eu_q"], d["eu_p"]
        c0 = d["cen"][ci]
        v = View(center=c0, az=35, el=25, scale=21, at=[-3.3, -0.1, 0])
        near = np.linalg.norm(d["P"] - c0, axis=1) < 0.16
        cl = cloud(v, d["P"][near], label_colors(d["L"][near]), 3.0, 0.8)
        truth = m.mob(v, d["q_true"], d["c_true"], C_TRUE, fill=0.0, edge_width=2, edge_opacity=0.8)
        leg = legend([(C_OBJ, "object points"), (C_CLUT, "clutter points"), (C_TRUE, "true pose", "box"), (C_EST, "hypothesis: x axis (large dot), y axis (small dot)")], 15)
        leg.to_corner(UR, buff=0.3).shift(DOWN * 0.75)
        s = ValueTracker(0.0)
        tri = always_redraw(lambda: VGroup(*[spike(v, *path_pose(tq[:, j], tp[:, j], s.get_value())) for j in idx]))
        self.play(FadeIn(cl), FadeIn(truth), FadeIn(leg))
        self.play(FadeIn(tri))
        cap.show("32 starts at one candidate position: orientations spread evenly over SO(3). Each spike is one pose hypothesis.", 3)
        step_l = always_redraw(lambda: T(f"RK4 step {min(4, int(s.get_value() * 4 + 1e-6) + (0 if s.get_value() >= 1 else 1))} of 4", 20, C_EST).move_to([-3.3, -2.7, 0]))
        self.add(step_l)
        cap.show("The same network is evaluated at every solver stage; every hypothesis sees the cloud in its own body frame.")
        for i in range(4):
            self.play(s.animate.set_value((i + 1) / 4), run_time=1.6, rate_func=smooth)
            self.wait(0.3)
        good = (d["er_flow"][idx] < 10) & (d["ep_flow"][idx] < 10)
        groups = {}
        for j in idx:
            groups.setdefault(int(d["grp0"][j]), []).append(j)
        gs = sorted(groups.values(), key=lambda g: -len(g))
        reps = [g[0] for g in gs[:3]]
        modes = VGroup(*[m.mob(v, d["q_flow"][j], d["p_flow"][j], C_EST, fill=0.0, edge_width=2.2) for j in reps])
        self.remove(step_l)
        self.play(FadeIn(modes), tri.animate.set_opacity(0.5))
        cap.show(f"They do not all agree: {len(gs)} groups of poses here. {int(good.sum())} of 32 end within 10° and 10 mm of the truth. "
                 f"The outlines are the 3 largest groups.", 5)

        # ---- Euler against RK4 on one hypothesis
        gi = idx[good]
        ang = lambda Q: qangle_deg(Q, d["q_true"])
        jb = int(gi[np.argmax(ang(tq[0, gi]))])
        e_rk, e_eu = ang(tq[:, jb]), ang(eq_[:, jb])
        self.play(FadeOut(modes), FadeOut(tri), FadeOut(leg))
        ymax = float(np.ceil(e_rk[0] / 30) * 30)
        ax = Axes(x_range=[0, 16, 4], y_range=[0, ymax, 30], x_length=5.2, y_length=3.0, tips=False,
                  axis_config={"color": C_DIM, "stroke_width": 2, "tick_size": 0.06}).move_to([3.5, 0.2, 0])
        xl = T("network evaluations", 17, C_DIM).next_to(ax, DOWN, buff=0.42)
        yl = T("rotation error to the truth (°)", 17, C_DIM).next_to(ax, UP, buff=0.2)
        xt = VGroup(*[T(str(i), 14, C_DIM).next_to(ax.c2p(i, 0), DOWN, buff=0.12) for i in range(0, 17, 4)])
        yt = VGroup(*[T(f"{a:.0f}", 14, C_DIM).next_to(ax.c2p(0, a), LEFT, buff=0.12) for a in np.arange(0, ymax + 1, 30)])
        lrk = T("RK4 × 4:  4 evaluations per step", 17, C_RK).move_to(ax.c2p(11, ymax * 0.97))
        leu = T("Euler × 8:  1 evaluation per step", 18, C_EU).next_to(lrk, DOWN, buff=0.12, aligned_edge=LEFT)
        u = ValueTracker(0.0)                      # network evaluations spent, 0 .. 16
        st_rk = lambda: min(u.get_value(), 16) / 16
        st_eu = lambda: min(u.get_value(), 8) / 8
        both = always_redraw(lambda: VGroup(
            m.mob(v, *path_pose(tq[:, jb], tp[:, jb], st_rk()), C_RK, fill=0.0, edge_width=2.6),
            m.mob(v, *path_pose(eq_[:, jb], ep_[:, jb], st_eu()), C_EU, fill=0.0, edge_width=2.6)))

        def plot():
            g = VGroup()
            for e, per, c in ((e_rk, 4, C_RK), (e_eu, 1, C_EU)):
                n = int(min(u.get_value(), per * (len(e) - 1)) / per + 1e-6)
                pts = [ax.c2p(i * per, e[i]) for i in range(n + 1)]
                if n > 0:
                    g.add(VMobject(color=c, stroke_width=3).set_points_as_corners(pts))
                g.add(*[Dot(p, radius=0.07, color=c) for p in pts])
            return g
        pl = always_redraw(plot)
        cnt = always_redraw(lambda: VGroup(T(f"RK4: {min(int(u.get_value() + 1e-6), 16)} evaluations", 18, C_RK), T(f"Euler: {min(int(u.get_value() + 1e-6), 8)} evaluations", 18, C_EU))
                            .arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([-3.3, -2.7, 0]))
        self.play(FadeIn(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), FadeIn(lrk), FadeIn(leu), FadeIn(both), FadeIn(pl), FadeIn(cnt))
        cap.show(f"One start, {e_rk[0]:.0f}° from the truth, solved twice. A dot is a solver state; the x axis counts network evaluations.", 3)
        self.play(u.animate.set_value(16), run_time=9, rate_func=linear)
        dq = qangle_deg(eq_[-1, idx], tq[-1, idx])
        cap.show(f"Euler × 8 ends at {e_eu[-1]:.1f}°, RK4 × 4 at {e_rk[-1]:.1f}°, at half the evaluations. Over these 32 starts the two end poses differ by {np.median(dq):.1f}° (median).", 5)
        tab = VGroup(T("through the tracker, 1,600 synthetic scenes, no verifier (README)", 15, C_DIM),
                     T("RK4 × 4      16 evaluations     1359 / 142 / 99", 17, C_RK),
                     T("Euler × 8      8 evaluations     1364 / 142 / 94", 17, C_EU),
                     T("Euler × 4      4 evaluations     1346 / 151 / 103", 17),
                     T("correct / wrong / none; a difference of a few scenes is within the noise", 15, C_DIM)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        tb = SurroundingRectangle(tab, buff=0.16, color=C_RULE, stroke_width=1, fill_color="#ffffff", fill_opacity=0.97, corner_radius=0)
        tg = VGroup(tb, tab).move_to([3.0, -2.0, 0]).set_z_index(40)
        self.play(FadeOut(xl), FadeOut(xt), ax.animate.shift(UP * 0.5), yl.animate.shift(UP * 0.5), yt.animate.shift(UP * 0.5), FadeIn(tg))
        cap.show("Through the tracker Euler × 8 equals RK4 × 4. Fewer than 8 evaluations starts to cost scenes.", 7)
