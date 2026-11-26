"""Scene 7: the verifier (flow/verifier.py) on the same synthetic scene. Poses, scores and the tracker's accepted
poses are stored results: the hypotheses of the trained flow, verifier.icp, the Verifier network, and the rows of
trk_hint_eval for this scene in the evaluation of flow/README.md (without and with --min-score 0.85)."""
from style import *


class VerifierScene(Scene):
    def construct(self):
        d, m = data(), mesh()
        self.add(title("The verifier", "refine, score, drop: before the tracker sees a hypothesis", n=7))
        self.add(note(SYNTH, 15))
        cap = Caption(self)
        P, col = d["P"], label_colors(d["L"])
        sel0 = [int(j) for j in d["sel0"]]
        v = View(center=np.array([0.31, 0.02, 0.50]), az=28, el=32, scale=10.0, at=np.array([-1.7, -0.5, 0.0]))
        v.clip = (-7.2, 0.8, -3.2, 2.8)
        cl = cloud(v, P, col, 2.6, 0.7)
        truth = m.mob(v, d["q_true"], d["c_true"], C_TRUE, fill=0.55, edge_width=2)
        leg = legend([(C_TRUE, "true pose", "box"), (C_EST, "hypothesis", "box"), (C_REJ, "rejected / wrong", "box"), (C_ACC, "accepted by the tracker", "box")], 15)
        leg.to_corner(UR, buff=0.3).shift(DOWN * 0.75)
        self.play(FadeIn(cl), FadeIn(truth), FadeIn(leg))
        h0 = VGroup(*[m.mob(v, d["q_flow"][j], d["p_flow"][j], C_EST, fill=0.0, edge_width=1.3, edge_opacity=0.8) for j in sel0])
        self.play(LaggedStart(*[FadeIn(x) for x in h0], lag_ratio=0.08))
        right = tag("the right one is among them", 14, C_TRUE).next_to(truth, UP, buff=0.25)
        self.play(FadeIn(right))
        good0 = (d["er_flow"] < 10) & (d["ep_flow"] < 10)
        cap.show(f"Without the verifier: {len(sel0)} hypotheses go to the tracker. {int(good0[sel0].sum())} of them is right, at the true pose.", 3)
        # what the tracker accepted
        qa, pa = tracker_pose(d["trk_flow"])
        acc = m.mob(v, qa, pa, C_REJ, fill=0.0, edge_width=4)
        al = tag(f"accepted by the tracker:\n{d['trk_flow'][9]:.0f} mm, {d['trk_flow'][10]:.0f}° from the truth", 16, C_REJ, line_spacing=0.8).next_to(acc, DOWN, buff=0.15)
        self.play(h0.animate.set_stroke(opacity=0.35), FadeIn(acc), FadeIn(al))
        cap.show("The tracker runs ICP from each and takes the best-fitting one that passes its fit check. Here that is a wrong pose on clutter.", 4.5)

        # ---- verifier: bars
        self.play(FadeOut(acc), FadeOut(al), FadeOut(h0), FadeOut(leg), FadeOut(right))
        reps = {}
        for j in np.argsort(-d["score"]):
            reps.setdefault(int(d["grp1"][j]), int(j))
        order = sorted(reps.values(), key=lambda j: -d["score"][j])
        NB = 12
        show = order[:NB]
        good1 = (d["er_icp"] < 10) & (d["ep_icp"] < 10)
        thr = float(d["min_score"])
        bx, bw, by, bh = 2.0, 2.6, 1.5, 0.2
        head = VGroup(T("verifier score of each group's best hypothesis", 16, weight=BOLD),
                      T(f"highest {NB} of {len(order)} groups", 14, C_DIM)).arrange(DOWN, buff=0.06, aligned_edge=LEFT).move_to([bx + bw / 2 + 0.5, 2.3, 0])
        axis = VGroup(Line([bx, by + 0.25, 0], [bx + bw, by + 0.25, 0], color=C_DIM, stroke_width=1.5))
        for a in (0, 0.5, 1.0):
            axis.add(T(f"{a:g}", 13, C_DIM).move_to([bx + bw * a, by + 0.43, 0]))
        
        bars, tags = VGroup(), VGroup()
        for i, j in enumerate(show):
            y = by - i * (bh + 0.07)
            w = max(0.03, bw * float(d["score"][j]))
            bars.add(Rectangle(width=w, height=bh, stroke_width=0, fill_color=C_EST, fill_opacity=0.95).move_to([bx + w / 2, y, 0]))
            tags.add(T(f"{d['score'][j]:.2f}    {d['ep_icp'][j]:.0f} mm, {d['er_icp'][j]:.0f}°", 13, C_TRUE if good1[j] else C_DIM).next_to([bx + bw + 0.1, y, 0], RIGHT, buff=0))
        tagh = T("score   error to the truth", 12, C_DIM).next_to(tags[0], UP, buff=0.1, aligned_edge=LEFT)
        h1 = VGroup(*[m.mob(v, d["q_flow"][j], d["p_flow"][j], C_EST, fill=0.0, edge_width=1.3, edge_opacity=0.8) for j in show])
        h1r = VGroup(*[m.mob(v, d["q_icp"][j], d["p_icp"][j], C_EST, fill=0.0, edge_width=1.3, edge_opacity=0.8) for j in show])
        self.play(FadeIn(h1))
        cap.show("With the verifier. First every hypothesis is refined: 6 point-to-plane steps on its 512 points, against the closest-point table.", 1.5)
        self.play(Transform(h1, h1r), run_time=2.5)
        self.wait(1)
        br = VGroup(T("cloud branch: 19 residual features of the 512 points", 14), T("model branch: 256 model points, distance to nearest cloud point", 14),
                    T("→ one score: is this refined pose the object?  (0.53 M parameters)", 14, C_EST)).arrange(DOWN, buff=0.08, aligned_edge=LEFT).scale(0.93).move_to([bx + bw / 2 + 0.5, -2.6, 0])
        self.play(FadeIn(head), FadeIn(axis), FadeIn(br), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.1), run_time=2.5)
        self.play(FadeIn(tags), FadeIn(tagh))
        cap.show("Then a small network scores each refined pose: how well the cloud fits the model, and how much of the model the cloud covers.", 4.5)
        ybot = by - (NB - 1) * (bh + 0.07) - 0.2
        tl = DashedLine([bx + bw * thr, by + 0.2, 0], [bx + bw * thr, ybot, 0], color=C_TEXT, stroke_width=2.5, dash_length=0.07)
        tll = T(f"threshold {thr:g}", 14).next_to(tl, DOWN, buff=0.06)
        self.play(Create(tl), FadeIn(tll))
        low = [i for i, j in enumerate(show) if d["score"][j] < thr]
        hi = [i for i, j in enumerate(show) if d["score"][j] >= thr]
        self.play(*[bars[i].animate.set_fill(C_REJ, 0.9) for i in low], *[h1[i].animate.set_stroke(C_REJ) for i in low])
        cap.show(f"Hypotheses below the threshold are not sent. Here {len(order) - len(d['sel1'])} of {len(order)} are dropped and {len(d['sel1'])} is left.", 2.5)
        self.play(*[bars[i].animate.set_fill(opacity=0.25) for i in low], *[FadeOut(h1[i]) for i in low], *[tags[i].animate.set_opacity(0.35) for i in low])
        qv, pv = tracker_pose(d["trk_ver"])
        accv = m.mob(v, qv, pv, C_ACC, fill=0.0, edge_width=4.5)
        tk = tick().next_to(accv, UP, buff=0.12)
        avl = tag(f"accepted by the tracker:\n{d['trk_ver'][9]:.1f} mm, {d['trk_ver'][10]:.1f}° from the truth", 16, C_ACC, line_spacing=0.8).next_to(accv, DOWN, buff=0.15)
        self.play(FadeIn(accv), FadeIn(tk), FadeIn(avl), *[FadeOut(h1[i]) for i in hi])
        cap.show("The tracker still runs its own ICP and fit check on what is left, and now accepts the true pose.", 4.5)
        cap.show("1,600 synthetic scenes: sorting by score alone did not help (wrong 142 → 163). Dropping below 0.85 did (wrong 142 → 42, none 99 → 151).", 7)
