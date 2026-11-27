"""Scene 3: what the network sees. The 512 points of one crop along one real solver trajectory of the trained flow
(assets/scene.npz, f_*): the points in the body frame of the estimate, the closest-surface vectors from the table
(rnode.lookup) and the 19 per-point features (rnode.features)."""
from style import *


class NetworkInputs(Scene):
    def construct(self):
        d, m = data(), mesh()
        self.add(title("What the network sees", "the cloud in the body frame of the estimate", n=3))
        self.add(note(SYNTH, 15))
        cap = Caption(self)
        ok, lab = d["f_ok"], d["f_lab"]
        X = d["f_X"][ok] + d["f_seed"]                               # rig frame
        xm, cp, lab = d["f_xm"][:, ok], d["f_cp"][:, ok], lab[ok]
        fq, fp = d["f_q"], d["f_p"] + d["f_seed"]
        K = len(fq)
        col = label_colors(lab)
        rs = np.random.default_rng(1)
        io, ic = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
        show = np.concatenate([rs.permutation(io)[:90], rs.permutation(ic)[:70]])
        dist = np.linalg.norm(cp - xm, axis=-1) * 1000               # mm, [K, n]
        d_obj, d_clu = dist[:, lab == 1].mean(1), dist[:, lab == 0].mean(1)

        vL = View(center=d["c_true"], az=35, el=25, scale=16, at=[-4.7, 0.0, 0])
        vB = View(center=[0, 0, 0], az=35, el=25, scale=16, at=[-0.35, 0.0, 0], up=(0, 0, 1))
        k = ValueTracker(0.0)
        ki = lambda: int(round(min(max(k.get_value(), 0), K - 1)))

        hL = T("rig frame", 20, weight=BOLD).move_to([-4.7, 2.5, 0])
        hB = T("body frame of the estimate", 20, weight=BOLD).move_to([-0.35, 2.5, 0])
        gL = axes_gizmo(vL, d["c_true"] + np.array([-0.13, 0.12, 0.0]), 0.035, ("$x$", "$y$", "$z$"))
        gB = axes_gizmo(vB, np.array([-0.13, -0.02, -0.11]), 0.035, ("$x_b$", "$y_b$", "$z_b$"))
        left_static = Group(cloud(vL, X, col, 3.2), m.mob(vL, d["q_true"], d["c_true"], C_TRUE, fill=0.0, edge_width=2, edge_opacity=0.9))
        est = always_redraw(lambda: m.mob(vL, fq[ki()], fp[ki()], C_EST, fill=0.0, edge_width=2.6))
        leg = legend([(C_OBJ, "points on the object"), (C_CLUT, "points on clutter"), (C_TRUE, "true pose", "box"),
                      (C_EST, "estimate / CAD model at the estimate", "box"), (C_RES, "vector to the closest surface point", "line")], 15)
        leg.move_to([4.75, 1.85, 0])
        self.play(FadeIn(hL), FadeIn(left_static), FadeIn(gL), FadeIn(leg[0]), FadeIn(leg[1][:3]))
        cap.show(f"{int(ok.sum())} measured points within 0.16 m of a seed. {int(lab.sum())} are on the object, the rest on clutter.", 2.5)
        self.play(FadeIn(est), FadeIn(leg[1][3]))
        cap.show(f"A pose estimate, here {qangle_deg(fq[0], d['q_true']):.0f}° from the truth: a start of the flow in this scene.", 2.5)

        model_b = m.mob(vB, [1, 0, 0, 0], [0, 0, 0], C_EST, fill=0.22, edge_width=2.2)
        pts_b = live_cloud(lambda: cloud(vB, xm[ki()], col, 3.2))
        eq = M(r"x_b = R^{\top} (x - p)", 22).move_to([-0.35, -2.8, 0])
        self.play(FadeIn(hB), FadeIn(model_b), FadeIn(pts_b), FadeIn(gB), FadeIn(eq))
        cap.show("The network gets the points in the frame of the estimate. There the CAD model never moves; the cloud does.", 3.5)

        def vecs():
            i = ki()
            S, E = vB(xm[i][show]), vB(cp[i][show])
            g = VGroup()
            for a, b, dd in zip(S, E, dist[i][show]):
                g.add(Line(a, b, stroke_width=1.8, color=C_RES, stroke_opacity=0.95 if dd < 60 else 0.5))
            return g
        vv = always_redraw(vecs)
        self.play(FadeIn(vv), FadeIn(leg[1][4]))
        cap.show("Per point, a table of the CAD model (128³ cells) gives the closest surface point and its normal. Residual r = n · (closest − x).", 4)

        # plot: mean distance to the surface, object against clutter
        ymax = float(np.ceil(max(d_obj.max(), d_clu.max()) / 20) * 20)
        ax = Axes(x_range=[0, 4, 1], y_range=[0, ymax, 20], x_length=3.2, y_length=1.9, tips=False,
                  axis_config={"color": C_DIM, "stroke_width": 2, "include_ticks": True, "tick_size": 0.05}).move_to([4.95, -1.0, 0])
        xl = T("solver steps (RK4 × 4)", 15, C_DIM).next_to(ax, DOWN, buff=0.3)
        yl = T("mean distance to the model surface (mm)", 15, C_DIM).next_to(ax, UP, buff=0.15)
        xt = VGroup(*[T(str(i), 13, C_DIM).next_to(ax.c2p(i, 0), DOWN, buff=0.1) for i in range(5)])
        yt = VGroup(*[T(f"{v:.0f}", 13, C_DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in np.arange(0, ymax + 1, 20)])
        xs = np.linspace(0, 4, K)

        def curve(y, c):
            def f():
                i = max(ki(), 1)
                return VGroup(VMobject(color=c, stroke_width=3.5).set_points_as_corners([ax.c2p(a, b) for a, b in zip(xs[:i + 1], y[:i + 1])]),
                              Dot(ax.c2p(xs[i], y[i]), radius=0.06, color=c))
            return always_redraw(f)
        co, cc = curve(d_obj, C_OBJ), curve(d_clu, C_CLUT)
        lo = T("object points", 14, C_OBJ).move_to(ax.c2p(3.0, d_obj[-1]) + UP * 0.25)
        lc = T("clutter points", 14, C_CLUT).move_to(ax.c2p(3.0, d_clu[-1]) + UP * 0.25)
        self.play(FadeIn(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), FadeIn(co), FadeIn(cc))
        cap.show("Now the trained flow moves the estimate (4 RK4 steps, shown continuously).")
        self.play(k.animate.set_value(K - 1), run_time=9, rate_func=linear)
        self.play(FadeIn(lo), FadeIn(lc))
        cap.show(f"Object points settle onto the surface ({d_obj[0]:.0f} → {d_obj[-1]:.0f} mm on average); clutter points cannot ({d_clu[0]:.0f} → {d_clu[-1]:.0f} mm).", 3.5)
        self.play(k.animate.set_value(0), run_time=1.5)
        self.play(k.animate.set_value(K - 1), run_time=6, rate_func=linear)

        feats = VGroup(
            T("19 inputs per point (rnode.features)", 15, weight=BOLD),
            *[Tex(t, color=C_TEXT, font_size=19) for t in (
                r"3\quad position $x_b$", r"1\quad which camera ($-1$ / $+1$)", r"3\quad vector $d$ to the closest surface point",
                r"3\quad surface normal $n$ there", r"1\quad point-to-plane residual $r = d \cdot n$", r"1\quad distance to the surface",
                r"3\quad $r\,n$", r"3\quad $r\,(x_b \times n)$", r"1\quad weight: near the estimate")],
            T("r n and r (x × n) are the terms of an ICP step.\nVector and distance saturate at about 30 mm.", 13, C_DIM, line_spacing=0.8),
        ).arrange(DOWN, buff=0.085, aligned_edge=LEFT)
        fb = SurroundingRectangle(feats, buff=0.15, color=C_RULE, stroke_width=1, fill_color="#ffffff", fill_opacity=0.97, corner_radius=0)
        fg = VGroup(fb, feats).move_to([4.6, -0.35, 0]).set_z_index(40)
        self.play(FadeOut(leg), FadeOut(VGroup(ax, xl, yl, xt, yt, co, cc, lo, lc)), FadeIn(fg))
        cap.show("The input already holds what a point-to-plane ICP step would use. The network learns which points to believe, and how far to move.", 7)
