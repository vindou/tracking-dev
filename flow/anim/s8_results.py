"""Scene 8: results on the 1,600 synthetic scenes of flow/data/eval2 (figures of flow/README.md, section "Verifier,
encoders and training variants")."""
from style import *

N = 1600
ROWS = [  # label, sublabel, correct, wrong, none
    ("classical search", "grid of starts + ICP + fit check", 958, 134, 508),
    ("flow + ICP", "runs/rnode/model_fm.pt, RK4 x 4", 1359, 142, 99),
    ("flow + verifier + ICP", "hypotheses scored below 0.85 not sent", 1407, 42, 151),
]
C_NONE = "#C9CCD1"


class Results(Scene):
    def construct(self):
        self.add(title("Results", "1,600 synthetic scenes, one depth pair each, tracker in search state", n=8))
        self.add(note(SYNTH))
        cap = Caption(self)
        x0, w, h = -2.6, 8.6, 0.62
        ys = [0.85, -0.05, -0.95]
        ax = Line([x0, -1.55, 0], [x0 + w, -1.55, 0], color=C_DIM, stroke_width=2)
        ticks = VGroup()
        for v in range(0, N + 1, 400):
            x = x0 + w * v / N
            ticks.add(Line([x, -1.55, 0], [x, -1.65, 0], color=C_DIM, stroke_width=2), T(f"{v:,}", 16, C_DIM).move_to([x, -1.85, 0]))
        axl = T("scenes (of 1,600)", 18, C_DIM).move_to([x0 + w / 2, -2.2, 0])
        leg = legend([(C_TRUE, "correct: accepted, within 10 mm and 10° of the truth", "box"),
                      (C_REJ, "wrong: accepted, farther from the truth", "box"),
                      (C_NONE, "none: nothing accepted", "box")], 17)
        for r in leg[1]:
            r[0][1].set_fill(r[0][1].get_color(), 1)
        leg.move_to([x0 + w / 2 - 1.2, 1.95, 0])
        self.play(Create(ax), FadeIn(ticks), FadeIn(axl), FadeIn(leg))
        bars = []
        for (lab, sub, c, wr, n), y in zip(ROWS, ys):
            l = VGroup(T(lab, 22, weight=BOLD), T(sub, 15, C_DIM)).arrange(DOWN, buff=0.06, aligned_edge=RIGHT)
            l.next_to([x0 - 0.15, y, 0], LEFT, buff=0)
            segs, nums, x = VGroup(), VGroup(), x0
            for v, col in ((c, C_TRUE), (wr, C_REJ), (n, C_NONE)):
                ww = w * v / N
                seg = Rectangle(width=ww, height=h, stroke_width=0, fill_color=col, fill_opacity=0.9).move_to([x + ww / 2, y, 0])
                segs.add(seg)
                tcol = "#ffffff" if col != C_NONE else C_TEXT
                t = T(str(v), 20, tcol, weight=BOLD)
                t.move_to(seg) if ww > 0.55 else t.set_color(col if col != C_NONE else C_TEXT).next_to(seg, UP, buff=0.05)
                nums.add(t)
                x += ww
            bars.append((l, segs, nums))
        cap.show("Correct / wrong / none of 1,600 scenes. The same ICP and the same fit check decide in all three rows.")
        for i, (l, segs, nums) in enumerate(bars):
            self.play(FadeIn(l), run_time=0.4)
            for s, n in zip(segs, nums):
                self.play(GrowFromEdge(s, LEFT), run_time=0.6, rate_func=linear)
                self.add(n)
            if i == 0:
                cap.show("Classical search finds the object in 958 scenes and accepts a wrong pose in 134 (no background map used).", 2.5)
            if i == 1:
                cap.show("Flow hypotheses as ICP starts: 1,359 found. Wrong acceptances do not go down: 142.", 2.5)
            if i == 2:
                cap.show("Dropping hypotheses the verifier scores below 0.85: 1,407 found, wrong acceptances 142 → 42.", 2.5)
        # the wrong segment of rows 2 and 3
        hl = VGroup(*[SurroundingRectangle(VGroup(bars[i][1][1]), color=C_TEXT, buff=0.05, stroke_width=3) for i in (1, 2)])
        paired = T("same scenes, verifier 0.85 against flow + ICP:  correct +82 −34,  wrong −107 +7", 19, C_TEXT).move_to([x0 + w / 2 - 1.2, -2.75, 0])
        self.play(Create(hl), FadeIn(paired))
        self.wait(2.5)
        cav = VGroup(
            T("Caveats (flow/README.md)", 20, weight=BOLD),
            T("•  Synthetic data only. The sensor noise model was not fitted to the real cameras.", 18),
            T("•  The threshold 0.85 was chosen on this same set; not confirmed on fresh scenes.", 18),
            T("•  More scenes end with no pose: 99 → 151. The verifier trades wrong for none.", 18),
            T("•  Two seeds of the same recipe differ by 3 scenes in total while 65 scenes flip: the noise floor.", 18),
            T("•  Flow rows: accepted from a hypothesis in the first cycle, without falling back to the classical search.", 18),
        ).arrange(DOWN, buff=0.11, aligned_edge=LEFT)
        box = SurroundingRectangle(cav, buff=0.2, color=C_RULE, stroke_width=1, fill_color="#ffffff", fill_opacity=0.97, corner_radius=0)
        cg = VGroup(box, cav).move_to([0, -2.6, 0]).set_z_index(60)
        self.play(FadeOut(cap.mob), FadeOut(paired), FadeOut(cap.rule), FadeOut(ticks), FadeOut(axl), FadeOut(ax), FadeIn(cg))
        self.wait(11)
