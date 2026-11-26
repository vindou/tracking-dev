"""Scene 1: the pipeline. Depth cloud -> flow proposals -> verifier -> ICP + fit check -> pose, with the classical
search for contrast. The poses drawn are stored results for one synthetic scene of flow/data/eval2: the tracker's
accepted pose without hypotheses (classical search), the hypotheses of the trained flow and verifier, and the
tracker's accepted pose with them (tracker/DESIGN.md sections 7, 12.3 and 13 for what the tracker does)."""
from style import *


class PipelineOverview(Scene):
    def construct(self):
        d, m = data(), mesh()
        self.add(title("Pipeline overview", "a learned flow proposes; ICP and a fit check decide", n=1))
        self.add(note(SYNTH, 15))
        cap = Caption(self)
        P, col = d["P"], label_colors(d["L"])
        v = View(center=np.array([0.30, -0.02, 0.42]), az=28, el=32, scale=6.6, at=np.array([-2.3, 0.2, 0.0]))
        v.clip = (-7.2, 2.3, -3.2, 2.8)
        cams = VGroup(*[camera_glyph(v, Tc, color=C_TEXT) for Tc in d["T_rig_cam"]])
        cam_l = T("two depth cameras", 15).next_to(cams, RIGHT, buff=0.2)
        ax = axes_gizmo(v, np.array([0.0, 0.0, 0.05]), 0.09, ("x", "y", "z"))
        ax.add(T("rig frame", 13, C_DIM).next_to(ax, LEFT, buff=0.1))
        self.play(FadeIn(cams), FadeIn(cam_l), FadeIn(ax))
        cA, cB = cloud(v, P[d["C"] == 0], col[d["C"] == 0], 2.6, 0.9), cloud(v, P[d["C"] == 1], col[d["C"] == 1], 2.6, 0.9)
        leg = legend([(C_OBJ, "points on the object"), (C_CLUT, "points on clutter"), (C_TRUE, "true pose", "box"), (C_EST, "pose hypothesis", "box"),
                      (C_REJ, "wrong / rejected", "box"), (C_ACC, "accepted", "box")], 15).to_corner(UR, buff=0.3).shift(DOWN * 0.75)
        self.play(FadeIn(cA), run_time=0.8)
        self.play(FadeIn(cB), FadeIn(leg), run_time=0.8)
        cap.show(f"Input: one point cloud from two depth cameras, {int(d['n_cloud']):,} points. A synthetic scene: ray-cast depth with a sensor noise model.", 3.5)
        truth = m.mob(v, d["q_true"], d["c_true"], C_TRUE, fill=0.0, edge_width=2.5)
        tl = T("the object", 15, C_TRUE).next_to(truth, UP, buff=0.5)
        ta = arrow(tl.get_bottom(), truth.get_top(), C_TRUE, 2)
        cad_v = View(center=[0, 0, 0], az=35, el=25, scale=17, at=[4.6, -0.6, 0], up=(0, 0, 1))
        cad = VGroup(m.mob(cad_v, [1, 0, 0, 0], [0, 0, 0], C_TRUE, fill=0.9), T("CAD model of the object (known)", 15, C_DIM).move_to([4.6, -1.9, 0]))
        self.play(FadeIn(truth), FadeIn(tl), GrowArrow(ta), FadeIn(cad))
        cap.show(f"Wanted: the 6-DoF pose of a known rigid object. Only {int(d['n_obj']):,} of the points are on it; the rest is clutter.", 3.5)
        self.play(FadeOut(tl), FadeOut(ta), FadeOut(cad), FadeOut(cams), FadeOut(cam_l), FadeOut(ax))

        # ---- classical search: a grid of ICP starts
        rows, cols = 4, 72
        gx, gy, cw, ch = 2.45, 0.15, 0.056, 0.17
        cells = VGroup(*[Rectangle(width=cw * 0.8, height=ch * 0.8, stroke_width=0, fill_color=C_DIM, fill_opacity=0.35).move_to([gx + c * cw, gy - r * ch * 1.25, 0])
                         for r in range(rows) for c in range(cols)])
        gh = VGroup(T("classical search", 20, weight=BOLD), T("up to 4 start positions × 72 orientations: 288 ICP runs", 12, C_DIM)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        gh.next_to(cells, UP, buff=0.2, aligned_edge=LEFT)
        gl = VGroup(T("72 orientations spread over SO(3)  →", 13, C_DIM).next_to(cells, DOWN, buff=0.1, aligned_edge=LEFT),
                    T("start\npositions", 13, C_DIM, line_spacing=0.8).next_to(cells, LEFT, buff=0.1))
        self.play(FadeIn(gh), FadeIn(cells), FadeIn(gl), leg.animate.set_opacity(0))
        self.remove(leg)
        cap.show("Without the network: start ICP from a grid. Each cluster of the cloud that could hold the object × 72 orientations, a few per 11 ms cycle.")
        ncyc = int(d["trk_classical"][12])
        cyc = ValueTracker(0)
        cl_ = always_redraw(lambda: T(f"cycle {int(cyc.get_value())}", 16, C_TEXT).next_to(cells, DOWN, buff=0.45, aligned_edge=RIGHT))
        self.add(cl_)
        self.play(LaggedStart(*[c.animate.set_fill(C_EST, 0.9) for c in cells], lag_ratio=0.01), cyc.animate.set_value(ncyc), run_time=6, rate_func=linear)
        valid = d["trk_classical"][0] > 0.5
        if valid:
            qc, pc = tracker_pose(d["trk_classical"])
            wrong = d["trk_classical"][9] > 10 or d["trk_classical"][10] > 10
            cm = m.mob(v, qc, pc, C_REJ if wrong else C_ACC, fill=0.0, edge_width=4)
            cll = tag(f"classical search accepts this after {ncyc} cycles:\n{d['trk_classical'][9]:.0f} mm and {d['trk_classical'][10]:.0f}° from the truth", 15, C_REJ if wrong else C_ACC, line_spacing=0.8).next_to(cm, DOWN, buff=0.2)
            self.play(FadeIn(cm), FadeIn(cll))
            cap.show("The best candidate that passes the fit check is accepted. In this scene that is a pose on clutter.", 4)
            self.play(FadeOut(cm), FadeOut(cll))
        else:
            cap.show(f"In this scene no candidate passes the fit check in {ncyc} cycles: no pose.", 4)
        self.play(FadeOut(cells), FadeOut(gh), FadeOut(gl), FadeOut(cl_))

        # ---- the pipeline with the flow
        names = [("depth cloud", "two cameras", C_OBJ), ("flow network", "proposes up to 16 poses", C_NET), ("verifier", "refines, scores, drops", C_NET),
                 ("tracker: ICP + fit check", "classical, unchanged", C_TEXT), ("accepted pose", "or none", C_ACC)]
        bl = VGroup(*[block(a, 3.3, 0.72, c, 18, sub=b) for a, b, c in names]).arrange(DOWN, buff=0.33).move_to([4.75, -0.35, 0])
        ars = VGroup(*[arrow(bl[i].get_bottom(), bl[i + 1].get_top(), C_DIM, 2.5) for i in range(4)])
        br1 = Brace(VGroup(bl[1], bl[2]), LEFT, buff=0.1, color=C_NET)
        br2 = Brace(bl[3], LEFT, buff=0.1, color=C_TEXT)
        brl = VGroup(T("learned:\nproposes", 14, C_NET, line_spacing=0.8).next_to(br1, LEFT, buff=0.08), T("classical:\ndecides", 14, C_TEXT, line_spacing=0.8).next_to(br2, LEFT, buff=0.08))
        for b in bl:
            b.save_state()
        self.play(FadeIn(bl[0]))
        cap.show("With the network: the same cloud first goes through a learned flow on SO(3) × R³.")
        sel0, sel1 = [int(j) for j in d["sel0"]], [int(j) for j in d["sel1"]]
        hyp = VGroup(*[m.mob(v, d["q_flow"][j], d["p_flow"][j], C_EST, fill=0.0, edge_width=1.4, edge_opacity=0.85) for j in sel0])
        self.play(GrowArrow(ars[0]), FadeIn(bl[1]), LaggedStart(*[FadeIn(h) for h in hyp], lag_ratio=0.06), run_time=2)
        cap.show(f"It proposes pose hypotheses: here {len(sel0)}, most of them on clutter or turned the wrong way.", 3.5)
        ref = VGroup(*[m.mob(v, d["q_icp"][j], d["p_icp"][j], C_EST, fill=0.0, edge_width=2.2) for j in sel1])
        self.play(GrowArrow(ars[1]), FadeIn(bl[2]), hyp.animate.set_stroke(C_REJ))
        self.play(FadeOut(hyp), FadeIn(ref))
        cap.show(f"A small learned verifier refines and scores them, and drops low scorers. Here {len(sel1)} is left.", 3.5)
        qv, pv = tracker_pose(d["trk_ver"])
        acc = m.mob(v, qv, pv, C_ACC, fill=0.0, edge_width=4.5)
        tk = tick().next_to(acc, UP, buff=0.12)
        al = tag(f"accepted: {d['trk_ver'][9]:.1f} mm, {d['trk_ver'][10]:.1f}° from the truth", 15, C_ACC).next_to(acc, DOWN, buff=0.2)
        self.play(GrowArrow(ars[2]), FadeIn(bl[3]))
        cap.show("The tracker runs its own ICP from each remaining hypothesis and applies the same fit check as before.", 3)
        self.play(GrowArrow(ars[3]), FadeIn(bl[4]), FadeOut(ref), FadeIn(acc), FadeIn(tk), FadeIn(al))
        self.play(FadeIn(br1), FadeIn(br2), FadeIn(brl))
        cap.show("The network never has the last word: it only chooses where ICP starts. No hypothesis passes → the classical search still runs.", 5)
        cap.show("Everything in these videos is evaluated on synthetic scenes. Nothing is claimed about real data.", 4.5)
