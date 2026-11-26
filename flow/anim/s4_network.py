"""Scene 4: the network. RNet of flow/rnode.py (the model of the results), then the two other encoders tried in the
same slot: LPNet (flow/rnode.py, Lean Point Networks) and LitePT (flow/litept.py). Widths and counts are read from
those files; parameter counts are those of rnode.make_net(); results quoted are from flow/README.md."""
from style import *


def conn(a, b, color=C_DIM, **kw):
    return arrow(a, b, color, 2.2, 0.13, **kw)


class Network(Scene):
    def construct(self):
        self.add(title("The network", "one vector field f(pose, t; cloud)", n=4))
        self.add(note(SYNTH, 15))
        cap = Caption(self)
        Y = 0.75
        # ---- fixed parts
        rows = VGroup(*[Rectangle(width=0.9, height=0.07, stroke_width=0, fill_color=C_OBJ if i % 3 else C_CLUT, fill_opacity=0.9) for i in range(12)]).arrange(DOWN, buff=0.04)
        inp = VGroup(rows, T("512 points\n× 19 features", 14, line_spacing=0.8)).arrange(DOWN, buff=0.12).move_to([-6.25, Y, 0])
        slot = RoundedRectangle(width=7.3, height=2.7, corner_radius=0.1, color=C_NET, stroke_width=2).move_to([-1.55, Y, 0])
        slot_d = DashedVMobject(slot, num_dashes=70)
        head = block("head MLP", 1.5, 0.8, C_NET, 16, sub="→ 512 → 512 → 6").move_to([3.35, Y, 0])
        out = VGroup(T("velocity (6)", 17, C_RES, weight=BOLD), T("3 angular + 3 linear,\nbody frame of the pose", 13, C_DIM, line_spacing=0.8)).arrange(DOWN, buff=0.08).move_to([5.75, Y, 0])
        segh = block("per-point MLP", 1.5, 0.7, C_NET, 16, sub="→ 128 → 1").move_to([3.35, 2.45, 0])
        sego = VGroup(T("object / clutter", 16, C_OBJ, weight=BOLD), T("one logit per point", 13, C_DIM)).arrange(DOWN, buff=0.06).move_to([5.75, 2.45, 0])
        tb = block("time t", 1.6, 0.55, C_COND, 15, sub="sin, cos × 16 frequencies").move_to([-5.9, -1.75, 0])
        sb = block("seed position", 1.6, 0.55, C_COND, 15, sub="in the body frame (3)").move_to([-5.9, -2.5, 0])
        cm = block("MLP", 1.2, 0.6, C_COND, 15, sub="35 → 64 → 64").move_to([-3.6, -2.1, 0])
        cl = Line(cm.get_right(), [3.35, -2.1, 0], color=C_COND, stroke_width=2.2)
        c_head = conn([3.35, -2.1, 0], head.get_bottom(), C_COND)
        c_slot = conn([-1.55, -2.1, 0], [-1.55, Y - 1.35, 0], C_COND)
        c_lab = T("condition c (64)", 13, C_COND).next_to(cl, DOWN, buff=0.08).shift(RIGHT * 0.5)
        c_t, c_s = conn(tb.get_right(), cm.get_left() + UP * 0.1, C_COND), conn(sb.get_right(), cm.get_left() + DOWN * 0.1, C_COND)
        a_in = conn(inp.get_right(), slot.get_left())
        a_g = conn(slot.get_right(), head.get_left())
        g_lab = T("global", 12, C_DIM).next_to(a_g, UP, buff=0.1).shift(RIGHT * 0.1)
        a_out = conn(head.get_right(), out.get_left(), C_RES)
        a_pp = conn([1.4, Y + 1.35, 0], segh.get_left() + DOWN * 0.1, path_arc=-0.5)
        pp_lab = T("per-point features\n+ global", 12, C_DIM, line_spacing=0.8).move_to([1.15, 2.85, 0])
        a_so = conn(segh.get_right(), sego.get_left(), C_OBJ)

        # ---- RNet inside the slot
        def rnet():
            b1 = block("per-point MLP", 1.6, 0.8, C_NET, 15, sub="19 → 64 → 128 → 192").move_to([-4.15, Y, 0])
            p1 = block("max", 0.75, 0.5, C_TEXT, 14).move_to([-2.75, Y + 0.85, 0])
            b2 = block("per-point MLP", 1.7, 0.8, C_NET, 15, sub="448 → 256 → 512").move_to([-1.1, Y, 0])
            p2 = block("max, mean", 1.15, 0.5, C_TEXT, 14).move_to([1.1, Y, 0])
            a1 = conn(b1.get_right(), b2.get_left())
            a2 = conn(b1.get_top() + RIGHT * 0.3, p1.get_left(), path_arc=-0.6)
            a3 = conn(p1.get_right(), b2.get_top() + LEFT * 0.3, path_arc=-0.6)
            a4 = conn(b2.get_right(), p2.get_left())
            l1 = T("over the points: 192", 12, C_DIM).next_to(p1, UP, buff=0.06)
            l2 = T("each point: own 192 + global 192 + c 64", 12, C_DIM).move_to([-1.9, Y - 0.85, 0])
            l3 = T("1024", 12, C_DIM).next_to(p2, DOWN, buff=0.06)
            return VGroup(b1, p1, b2, p2, a1, a2, a3, a4, l1, l2, l3)
        rn = rnet()
        name = lambda s, sub: VGroup(T(s, 16, C_NET, weight=BOLD), T(sub, 13, C_DIM)).arrange(RIGHT, buff=0.2).next_to(slot, UP, buff=0.08, aligned_edge=LEFT)
        n_r = name("PointNet-style encoder (RNet)", "1.18 M parameters in total")

        self.play(FadeIn(inp))
        cap.show("Input: the 512 points of the crop, 19 numbers each, in the body frame of the current pose.", 2.5)
        self.play(GrowArrow(a_in), FadeIn(rn[0]))
        cap.show("A shared MLP lifts every point on its own.", 1.5)
        self.play(FadeIn(rn[1]), FadeIn(rn[5]), FadeIn(rn[8]))
        cap.show("The maximum over the points gives one global feature: order of the points does not matter.", 2.5)
        self.play(FadeIn(rn[2]), FadeIn(rn[4]), FadeIn(rn[6]), FadeIn(rn[9]))
        cap.show("Each point is processed again, now knowing the global feature.", 2)
        self.play(FadeIn(tb), FadeIn(sb), FadeIn(cm), Create(cl), FadeIn(c_lab), GrowArrow(c_slot), GrowArrow(c_t), GrowArrow(c_s))
        cap.show("Conditioning: the time t of the flow and where the seed lies as seen from the pose. Slow frequencies: the solver takes few steps.", 4)
        self.play(FadeIn(rn[3]), FadeIn(rn[7]), FadeIn(rn[10]), GrowArrow(a_g), FadeIn(g_lab), FadeIn(head), GrowArrow(c_head))
        self.play(GrowArrow(a_out), FadeIn(out))
        cap.show("Pooled once more (maximum and mean), then a head gives 6 numbers: the velocity of the pose. Its last layer starts at zero.", 4)
        self.play(GrowArrow(a_pp), FadeIn(pp_lab), FadeIn(segh), GrowArrow(a_so), FadeIn(sego))
        cap.show("Second output: for every point, object or clutter. It ranks the position candidates of stage 1; the tracker does not use it.", 4.5)
        self.play(Create(slot_d), FadeIn(n_r))
        cap.show("This is the model behind the results: 1.18 M parameters. The dashed part is the point encoder. Two others were tried in its place.", 4.5)

        # ---- Lean Point Networks
        def lpn():
            g = VGroup()
            def seg(x0, labs, n):
                ys = [Y + 0.8, Y + 0.05, Y - 0.7]
                xs = [x0 + 0.75 * i for i in range(n)]
                for y, lab in zip(ys, labs):
                    g.add(T(lab, 11, C_DIM).next_to([x0 - 0.33, y, 0], LEFT, buff=0))
                    g.add(Line([x0 - 0.3, y, 0], [xs[-1] + 0.45, y, 0], color=C_DIM, stroke_width=1.5))
                    for x in xs:
                        g.add(Square(0.32, color=C_NET, stroke_width=2, fill_color=C_PANEL, fill_opacity=1).move_to([x, y, 0]))
                for x in xs:
                    for i in range(2):
                        g.add(DoubleArrow([x + 0.3, ys[i] - 0.08, 0], [x + 0.3, ys[i + 1] + 0.08, 0], buff=0, color=C_RES, stroke_width=1.8, tip_length=0.09, max_tip_length_to_length_ratio=0.3))
                return xs[-1] + 0.45, ys
            x1, ys = seg(-4.1, ("512 pts", "256", "128"), 3)
            m1 = Dot([x1 + 0.25, ys[2], 0], radius=0.07, color=C_TEXT)
            g.add(m1, *[Line([x1, y, 0], m1.get_center(), color=C_DIM, stroke_width=1.5) for y in ys[:2]])
            x2, _ = seg(-1.0, ("128", "64", "32"), 3)
            g.add(Line(m1.get_center(), [-1.25, ys[2], 0], color=C_DIM, stroke_width=0))
            g.add(conn(m1.get_center(), [-1.72, ys[2], 0]))
            m2 = Dot([x2 + 0.2, ys[2], 0], radius=0.07, color=C_TEXT)
            g.add(m2, *[Line([x2, y, 0], m2.get_center(), color=C_DIM, stroke_width=1.5) for y in ys[:2]])
            e3 = block("MLP 1024", 1.0, 0.6, C_NET, 12, sub="max over points").move_to([1.5, Y + 0.5, 0])
            g.add(e3, conn(m2.get_center(), e3.get_bottom(), path_arc=0.4))
            g.add(VGroup(Square(0.2, color=C_NET, stroke_width=2), T("neighbourhood block: 16 nearest points, max, residual", 12, C_DIM)).arrange(RIGHT, buff=0.1).move_to([-2.9, Y - 1.15, 0]))
            g.add(VGroup(Line(LEFT * 0.12, RIGHT * 0.12, color=C_RES, stroke_width=2.5), T("cross-links", 12, C_DIM)).arrange(RIGHT, buff=0.1).move_to([0.75, Y - 1.15, 0]))
            return g
        lp = lpn()
        n_l = name("Lean Point Networks encoder (LPNet)", "2.65 M parameters")
        self.play(FadeOut(rn), FadeOut(n_r))
        self.play(FadeIn(lp), FadeIn(n_l))
        cap.show("Lean Point Networks: the points at three resolutions in parallel. Blocks look at each point's 16 nearest neighbours; cross-links mix the resolutions.", 5.5)
        res = lambda s, c=C_TEXT: VGroup(BackgroundRectangle(t := T(s, 15, c), color=config.background_color, fill_opacity=0.9, buff=0.08), t).move_to([-1.55, -1.35, 0]).set_z_index(20)
        r_l = res("validation 80 % against 68 %   ·   through the tracker 1345 / 177 / 78 against 1359 / 142 / 99: more wrong acceptances", C_RES)
        r_l.scale_to_fit_width(min(r_l.width, 9.6)).move_to([0.2, -1.2, 0])
        self.play(FadeIn(r_l), c_slot.animate.set_opacity(0.25))
        cap.show("It scored better on held-out validation, but through the tracker it produced more wrong acceptances (README, 1,600 synthetic scenes).", 5.5)

        # ---- LitePT
        def lite():
            g = VGroup()
            rs = np.random.default_rng(3)
            xs = [-4.45, -3.15, -1.85, -0.45, 0.95]
            occ = rs.random((16, 16)) < 0.16
            for s, x in enumerate(xs):
                n = 16 >> s
                o = occ.reshape(n, 16 // n, n, 16 // n).any((1, 3))
                side, c0 = 0.95, np.array([x, Y + 0.6, 0])
                cell = side / n
                g.add(Square(side, color=C_DIM, stroke_width=1).move_to(c0))
                cen = []
                for i in range(n):
                    for j in range(n):
                        if o[i, j]:
                            p = c0 + np.array([(j + 0.5) * cell - side / 2, side / 2 - (i + 0.5) * cell, 0])
                            cen.append(p)
                            g.add(Square(cell * 0.92, stroke_width=0, fill_color=C_OBJ if s < 3 else C_NET, fill_opacity=0.9 if s < 3 else 0.4).move_to(p))
                if s >= 3:
                    for a in range(len(cen)):
                        for b in range(a + 1, len(cen)):
                            g.add(Line(cen[a], cen[b], color=C_TEXT, stroke_width=1.4))
                kind = "sparse conv 3³" if s < 3 else "attention"
                g.add(VGroup(T(kind, 12, C_OBJ if s < 3 else C_NET), T(f"{8 << s} mm cells", 11, C_DIM)).arrange(DOWN, buff=0.03).move_to([x, Y - 0.3, 0]))
                if s < 4:
                    g.add(conn([x + 0.5, Y + 0.6, 0], [xs[s + 1] - 0.5, Y + 0.6, 0]))
            g.add(T("stem: sparse conv 5³ on occupied cells; maximum over 2×2×2 cells between the 5 stages", 10, C_DIM).move_to([-1.55, Y - 0.8, 0]))
            g.add(T("attention over all cells of a cloud; position enters as a rotary encoding of grid coordinates (PointROPE)", 10, C_DIM).move_to([-1.55, Y - 1.05, 0]))
            g.add(T("max,\nmean", 12, line_spacing=0.8).move_to([1.68, Y + 0.6, 0]))
            return g
        lt = lite()
        n_t = name("LitePT encoder", "4.3 M (reduced) or 13.4 M (LitePT-S) parameters")
        self.play(FadeOut(lp), FadeOut(n_l), FadeOut(r_l))
        self.play(FadeIn(lt), FadeIn(n_t))
        cap.show("LitePT: the points go into 8 mm cells of a grid in the body frame. Sparse convolutions on the fine grids, attention on the coarse ones.", 5.5)
        r_t = res("45 × slower per training step than the PointNet-style encoder   ·   result through the tracker: pending", C_RES)
        r_t.scale_to_fit_width(min(r_t.width, 9.6)).move_to([0.2, -1.2, 0])
        self.play(FadeIn(r_t))
        cap.show("Same inputs, same conditioning, same two heads. Its result through the tracker is still pending.", 4.5)
        self.play(FadeOut(lt), FadeOut(n_t), FadeOut(r_t), FadeIn(rn), FadeIn(n_r), c_slot.animate.set_opacity(1))
        cap.show("So far no encoder has improved the tracker result over the PointNet-style one. It is the encoder used in the other scenes.", 5)
