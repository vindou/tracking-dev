"""Scene 2: the state (a pose), the geodesic path of flow matching on SO(3) x R^3 and the velocity target
(flow/rnode.py: fm_pair, sample_base; flow/train_rnode.py: the flow-matching phase).

SO(3) is drawn as the ball of rotation vectors (direction = axis, length = angle, radius pi with opposite surface
points identified), centred on the true rotation: the point of a rotation q is log(q_true^-1 q). In this chart the
geodesic q_true exp(t eta) of fm_pair is the straight segment from the centre to eta."""
from style import *


class StateAndFlow(Scene):
    def construct(self):
        d, m = data(), mesh()
        self.add(title("State and flow", "a pose on SO(3) × R³", n=2))
        cap = Caption(self)
        q1 = qexp(np.array([0.5, -0.9, 0.3]))                      # the data pose of this illustration
        p1 = np.array([0.0, 0.0, 0.0])
        eta = np.array([0.55, 0.75, -0.25]) * 1.35                  # |eta| = 1.3 rad = 74 degrees (sigma of the base: 0.6 rad per axis; a far sample)
        qb = qmul(q1, qexp(eta))
        pb = p1 + np.array([0.13, -0.035, 0.02])                    # drawn 3 x farther than a typical base sample (40 mm sigma), to be legible
        v = View(center=[0.055, -0.01, 0.0], az=25, el=22, scale=24, at=[-2.9, 0.1, 0])
        s = ValueTracker(0.0)
        pose = lambda u: (qslerp(q1, qb, u), (1 - u) * p1 + u * pb)

        true_m = m.mob(v, q1, p1, C_TRUE, fill=0.9)
        true_t = triad(v, q1, p1, C_TEXT, length=0.085, width=4)
        self.play(FadeIn(true_m))
        lab_p = T("p: position of the model centroid", 18).move_to([-4.3, 2.0, 0])
        lab_R = T("R: rotation of the object\n(a unit quaternion q in the code)", 18, line_spacing=0.8).move_to([-4.3, -2.2, 0])
        ar_p = arrow(lab_p.get_bottom(), v(p1), C_TEXT, 2)
        self.play(FadeIn(true_t), FadeIn(lab_p), FadeIn(lab_R), GrowArrow(ar_p))
        cap.show("A pose is a frame: where the model centroid is and how the object is turned. 3 + 3 degrees of freedom.", 3)
        leg = legend([(C_TRUE, "true (data) pose,  t = 0", "box"), (C_EST, "start pose / estimate,  t = 1", "box"),
                      (C_RES, "velocity target", "line")], 13).move_to([0.5, 3.32, 0])
        self.play(FadeOut(lab_p), FadeOut(lab_R), FadeOut(ar_p), FadeOut(true_t), true_m.animate.set_opacity(0.35), FadeIn(leg))

        # ---- SO(3) as the ball of rotation vectors around the true rotation
        BR = 1.7                                                    # screen radius of the ball (pi rad)
        bv = View(center=[0, 0, 0], az=25, el=22, scale=BR / np.pi, at=[4.5, 0.6, 0], up=(0, 0, 1))
        circ = lambda a, b, r=np.pi, **kw: VMobject(**kw).set_points_smoothly([bv(r * (np.cos(u) * a + np.sin(u) * b)) for u in np.linspace(0, 2 * np.pi, 49)])
        ex, ey, ez = np.eye(3)
        ball = VGroup(Circle(radius=BR, color=C_DIM, stroke_width=2).move_to(bv([0, 0, 0])),
                      circ(ex, ey, color=C_DIM, stroke_width=1, stroke_opacity=0.7), circ(ex, ez, color=C_DIM, stroke_width=1, stroke_opacity=0.7),
                      circ(ey, ez, color=C_DIM, stroke_width=1, stroke_opacity=0.7))
        ball_t = VGroup(T("SO(3): all rotations, as a ball", 16, weight=BOLD),
                        T("direction = rotation axis,  radius = angle from the true rotation\ncentre: the true rotation,  surface: 180°", 11, C_DIM, line_spacing=0.8)).arrange(DOWN, buff=0.06).move_to([4.5, 2.72, 0])
        c_dot = Dot(bv([0, 0, 0]), radius=0.07, color=C_TRUE)
        self.play(FadeIn(ball), FadeIn(ball_t), FadeIn(c_dot))
        cap.show("Rotations form a curved space, SO(3). One picture of it: a ball. Direction is the rotation axis, radius the angle; opposite surface points are the same rotation.", 5)
        rs = np.random.default_rng(4)
        samp = VGroup(*[Dot(bv(x), radius=0.025, color=C_EST, fill_opacity=0.55) for x in rs.normal(size=(140, 3)) * 0.6])
        sig = Circle(radius=0.6 * BR / np.pi, color=C_EST, stroke_width=1.5, stroke_opacity=0.8).move_to(bv([0, 0, 0]))
        sig_l = T("σ = 0.6 rad (34°)", 12, C_EST).next_to(sig, DOWN, buff=0.75)
        b_dot = Dot(bv(eta), radius=0.07, color=C_EST)
        base_m = m.mob(v, qb, pb, C_EST, fill=0.0, edge_width=2.5)
        self.play(FadeIn(samp), FadeIn(sig), FadeIn(sig_l))
        self.play(FadeIn(b_dot), FadeIn(base_m), Flash(b_dot, color=C_EST, line_length=0.12))
        cap.show("Training draws a start pose around the truth: rotation σ = 0.6 rad (dots in the ball), position σ = 40 mm (offset drawn 3× larger).", 4.5)

        # ---- the geodesic: straight line for p, constant-axis rotation for R
        line = DashedLine(v(p1), v(pb), color=C_TEXT, stroke_width=2, dash_length=0.08)
        tip = lambda u: v(((1 - u) * p1 + u * pb) + qmat(qslerp(q1, qb, u))[:, 0] * 0.085)
        arc = VMobject(color=C_TEXT, stroke_width=2).set_points_smoothly([tip(u) for u in np.linspace(0, 1, 24)])
        mover = always_redraw(lambda: VGroup(m.mob(v, *pose(s.get_value()), C_EST, fill=0.0, edge_width=3),
                                             triad(v, *pose(s.get_value()), C_EST, length=0.085, width=4)))
        geo = Line(bv([0, 0, 0]), bv(eta), color=C_TEXT, stroke_width=2.5)
        g_dot = always_redraw(lambda: Dot(bv(s.get_value() * eta), radius=0.085, color=C_EST, stroke_color=C_TEXT, stroke_width=1.5).set_z_index(5))
        tl = NumberLine([0, 1, 0.25], length=3.6, color=C_DIM, include_numbers=False).move_to([-2.9, -2.55, 0])
        tl_l = VGroup(M("t=0", 15, C_TRUE).next_to(tl.n2p(0), LEFT, buff=0.2), M("t=1", 15, C_EST).next_to(tl.n2p(1), RIGHT, buff=0.2),
                      T("time t along the path", 14, C_DIM, slant=ITALIC).next_to(tl, DOWN, buff=0.12))
        knob = always_redraw(lambda: Dot(tl.n2p(s.get_value()), radius=0.09, color=C_EST))
        eq = VGroup(M(r"q(t) = q\,\exp(t\,\eta), \qquad \eta = \log\!\big(q^{-1} q_{\mathrm{start}}\big)", 17), M(r"p(t) = (1-t)\,p + t\,p_{\mathrm{start}}", 17)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        eq.move_to([4.5, -1.6, 0])
        s.set_value(1.0)
        self.remove(base_m, b_dot)
        self.add(mover, g_dot)
        self.play(Create(line), Create(arc), Create(geo), FadeIn(tl), FadeIn(tl_l), FadeIn(knob), FadeIn(eq), samp.animate.set_opacity(0.2), FadeOut(sig_l))
        cap.show("The path between them is a geodesic: in SO(3) the shortest arc, a straight segment in this ball; for the centroid a straight line.")
        self.play(s.animate.set_value(0.0), run_time=4, rate_func=smooth)
        cap.show("Moving along it turns the object about one fixed axis at a constant rate while the centroid slides.")
        self.play(s.animate.set_value(0.6), run_time=3, rate_func=smooth)

        # ---- the velocity target
        hat = eta / np.linalg.norm(eta)
        ykw = dict(buff=0, color=C_RES, stroke_width=5, tip_length=0.2, max_tip_length_to_length_ratio=0.5)

        def vel():
            u = s.get_value()
            q, p = pose(u)
            ax = qmat(q) @ hat
            a = Arrow(v(p), v(p + (pb - p1) * 0.55), **ykw)
            b = Arrow(v(p), v(p + ax * 0.06), **ykw)
            e1 = np.cross(ax, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
            ring_ = VMobject(color=C_RES, stroke_width=3).set_points_smoothly([v(p + ax * 0.045 + 0.014 * (np.cos(a_) * e1 + np.sin(a_) * e2)) for a_ in np.linspace(0, 4.6, 16)])
            c = Arrow(bv(u * eta), bv(u * eta + hat * 0.95), **ykw)          # the same angular velocity, in the ball
            return VGroup(a, b, ring_, c)
        va = always_redraw(vel)
        l_lin = T("linear velocity (3)", 15, C_RES).move_to([-0.9, 2.3, 0])
        l_ang = T("angular velocity (3):\naxis × rate", 15, C_RES, line_spacing=0.8).move_to([-0.55, -1.75, 0])
        tgt = VGroup(M(r"\mathrm{target} = \big(\,\eta,\;\; R(t)^{\!\top} (p_{\mathrm{start}} - p)\,/\,0.1\,\mathrm{m}\,\big)", 17, C_RES),
                     M(r"\mathrm{loss} = \big\lVert f(\mathrm{pose}, t;\, \mathrm{cloud}) - \mathrm{target} \big\rVert^{2}", 17)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(eq, DOWN, buff=0.16, aligned_edge=LEFT)
        self.play(FadeIn(va), FadeIn(l_lin), FadeIn(l_ang), FadeIn(tgt))
        cap.show("The network regresses the velocity of this path in the body frame of the pose: 3 angular and 3 linear numbers, constant along it.", 4)
        self.play(s.animate.set_value(0.2), run_time=2, rate_func=smooth)
        self.play(s.animate.set_value(0.9), run_time=2, rate_func=smooth)
        cap.show("No solver in training: sample a time t, place the pose on the path, compare the network output with the target.", 3.5)
        self.play(FadeOut(va), FadeOut(l_lin), FadeOut(l_ang), s.animate.set_value(1.0))
        run = T("inference: integrate f backward, t = 1 → 0", 14, C_EST).next_to(tgt, DOWN, buff=0.14, aligned_edge=LEFT)
        self.play(FadeIn(run), tgt.animate.set_opacity(0.35), true_m.animate.set_opacity(0.2), FadeOut(samp), FadeOut(sig))
        cap.show("At inference the truth is unknown. The learned field is integrated backward in t: from a start toward the pose it implies.")
        self.play(s.animate.set_value(0.0), run_time=4.5, rate_func=linear)
        self.play(true_m.animate.set_opacity(0.5))
        cap.show("In the code time runs from the data (t = 0) to the start distribution (t = 1), so the solver steps go from 1 down to 0.", 4)
