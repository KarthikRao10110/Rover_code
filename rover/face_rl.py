"""GPU-rendered robot face (raylib / OpenGL ES). Vector/EMO-style procedural eyes.

Each eye is drawn column by column as a rounded rectangle whose top and bottom edges are
clipped by "eyelids". That one shape covers every expression: angled lids (determined),
curved lower lid (happy crescent), half-closed (sleepy), wide (surprised). All parameters
ease toward their target with critically-damped smoothing, so expressions flow into each
other. The mouth lip-syncs from Piper phoneme timings (visemes).

Same interface as face.Face:  FaceRL(state, on_interrupt, on_quit, fullscreen).run()
Keys: Space / tap = interrupt, Esc or Q = quit, F = toggle fullscreen.
"""

import math
import random
import time

import pyray as rl

FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

BG = (6, 9, 14)
NORMAL = (70, 205, 255)
DEBATE = (255, 150, 60)
THINK = (165, 135, 255)
LISTEN = (90, 230, 200)
SAD = (110, 140, 220)
WHITE = (238, 242, 247)
DIM = (125, 135, 150)

# expression -> eye parameters
#   w,h: size multipliers   top: top-lid closure 0..1   tilt: lid slope (+ = angry, - = sad)
#   happy: lower-lid crescent 0..1   lx,ly: gaze   asym: left/right size difference (curious)
#   smile: idle mouth curvature (-1 frown .. 1 big smile)
EXPR = {
    "neutral":    dict(w=1.00, h=1.00, top=0.00, tilt=0.00, happy=0.00, lx=0.0, ly=0.0, asym=0.0, smile=0.35),
    "happy":      dict(w=1.05, h=0.95, top=0.00, tilt=0.00, happy=0.30, lx=0.0, ly=0.0, asym=0.0, smile=0.9),
    "listening":  dict(w=1.04, h=1.10, top=0.00, tilt=-0.05, happy=0.0, lx=0.0, ly=0.05, asym=0.0, smile=0.25),
    "thinking":   dict(w=0.95, h=0.80, top=0.18, tilt=0.05, happy=0.0, lx=0.55, ly=-0.55, asym=0.12, smile=0.0),
    "speaking":   dict(w=1.02, h=1.00, top=0.02, tilt=0.00, happy=0.00, lx=0.0, ly=0.0, asym=0.0, smile=0.4),
    "determined": dict(w=1.05, h=0.92, top=0.30, tilt=0.42, happy=0.0, lx=0.0, ly=0.0, asym=0.0, smile=-0.1),
    "debating":   dict(w=1.05, h=0.95, top=0.22, tilt=0.30, happy=0.00, lx=0.0, ly=0.0, asym=0.0, smile=0.1),
    "surprised":  dict(w=1.12, h=1.28, top=0.00, tilt=-0.08, happy=0.0, lx=0.0, ly=-0.08, asym=0.0, smile=-0.2),
    "sleepy":     dict(w=1.00, h=0.90, top=0.58, tilt=-0.05, happy=0.0, lx=0.0, ly=0.2, asym=0.0, smile=0.0),
    "sad":        dict(w=0.95, h=0.92, top=0.22, tilt=-0.35, happy=0.0, lx=0.0, ly=0.25, asym=0.0, smile=-0.6),
    "curious":    dict(w=1.02, h=1.05, top=0.05, tilt=0.10, happy=0.0, lx=0.25, ly=-0.1, asym=0.22, smile=0.2),
}

# viseme -> (mouth width, mouth height) relative units
MOUTH = {
    "rest": (0.50, 0.05), "MBP": (0.46, 0.02), "FV": (0.56, 0.12), "TH": (0.58, 0.16),
    "L": (0.55, 0.30), "O": (0.34, 0.50), "U": (0.26, 0.30), "AA": (0.66, 0.70),
    "E": (0.68, 0.38), "I": (0.74, 0.20), "C": (0.58, 0.24),
}


def col(rgb, a=255):
    return rl.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(max(0, min(255, a))))


def mixc(c1, c2, t):
    return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))


def ease(cur, target, dt, speed=10.0):
    return cur + (target - cur) * (1 - math.exp(-dt * speed))


def ascii_text(t):
    return (t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
             .replace("…", "...").replace("—", " - ").replace("–", "-"))


class FaceRL:
    def __init__(self, state, on_interrupt, on_quit, fullscreen=True, size=(800, 480)):
        self.state = state
        self.on_interrupt = on_interrupt
        self.on_quit = on_quit
        self.fullscreen = fullscreen
        flags = rl.ConfigFlags.FLAG_MSAA_4X_HINT | rl.ConfigFlags.FLAG_VSYNC_HINT
        if not fullscreen:
            flags |= rl.ConfigFlags.FLAG_WINDOW_RESIZABLE
        rl.set_config_flags(flags)
        rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
        rl.init_window(0 if fullscreen else size[0], 0 if fullscreen else size[1], "Rover")
        if fullscreen:
            rl.toggle_borderless_windowed()
            rl.hide_cursor()
        rl.set_exit_key(rl.KeyboardKey.KEY_NULL)
        rl.set_target_fps(30)   # smooth enough; leaves CPU for the LLM
        self.font_b = rl.load_font_ex(FONT_BOLD, 72, None, 0)
        self.font_r = rl.load_font_ex(FONT_REG, 64, None, 0)
        for f in (self.font_b, self.font_r):
            rl.set_texture_filter(f.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        # animated parameters
        self.p = dict(EXPR["sleepy"])
        self.color = NORMAL
        self.mouth_w, self.mouth_h = MOUTH["rest"]
        self.blink = 0.0
        self.next_blink = time.monotonic() + 2.0
        self.blink_start = -1.0
        self.gaze = [0.0, 0.0]
        self.gaze_target = [0.0, 0.0]
        self.next_glance = time.monotonic() + 2.5
        self.mic = 0.0
        self.t = 0.0
        self.last_status = None
        self.surprise_until = 0.0
        self.happy_until = 0.0

    # ------------------------------------------------------------ loop ----------
    def run(self):
        while self.state.alive:
            if rl.window_should_close():
                self.on_quit()
                break
            if rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE) or rl.is_key_pressed(rl.KeyboardKey.KEY_Q):
                self.on_quit()
            if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE):
                self.on_interrupt()
            if rl.is_key_pressed(rl.KeyboardKey.KEY_F):
                rl.toggle_borderless_windowed()
                self.fullscreen = not self.fullscreen
                (rl.hide_cursor if self.fullscreen else rl.show_cursor)()
            if self.fullscreen and rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
                self.on_interrupt()  # tap the screen
            dt = min(rl.get_frame_time(), 0.1)
            rl.begin_drawing()
            self.draw(rl.get_screen_width(), rl.get_screen_height(), dt)
            rl.end_drawing()
        rl.unload_font(self.font_b)
        rl.unload_font(self.font_r)
        rl.close_window()

    # ------------------------------------------------------------ logic ---------
    def _expression(self, now):
        st = self.state
        if st.status != self.last_status:
            if st.status == "listening" and self.last_status == "speaking":
                self.surprise_until = now + 0.6      # interrupted: brief surprise
            if self.last_status == "starting" and st.status == "idle":
                self.happy_until = now + 2.5         # just woke up
            self.last_status = st.status
        if now < self.surprise_until:
            return "surprised"
        if st.status == "starting":
            return "sleepy"
        if getattr(st, "emotion", ""):
            return st.emotion
        debate = st.mode == "debate"
        if st.status == "thinking":
            return "determined" if debate else "thinking"
        if st.status == "speaking":
            return "debating" if debate else "speaking"
        if st.status == "listening":
            return "determined" if debate else "listening"
        if now < self.happy_until:
            return "happy"
        return "determined" if debate else "neutral"

    def _update(self, dt, now):
        st = self.state
        self.t += dt
        expr = self._expression(now)
        target = EXPR[expr]
        for k, v in target.items():
            self.p[k] = ease(self.p[k], v, dt, 9.0)

        base = DEBATE if st.mode == "debate" else NORMAL
        if st.status == "thinking":
            base = mixc(base, THINK, 0.2 if st.mode == "debate" else 0.55)
        elif st.status == "listening" and st.mode != "debate":
            base = mixc(base, LISTEN, 0.5)
        elif expr == "sad":
            base = SAD
        self.color = tuple(ease(self.color[i], base[i], dt, 6.0) for i in range(3))

        # blinking (double blink sometimes)
        if now >= self.next_blink:
            self.blink_start = now
            self.next_blink = now + (0.25 if random.random() < 0.15 else random.uniform(2.5, 6.0))
        b = (now - self.blink_start) / 0.15
        self.blink = math.sin(math.pi * b) if 0 <= b <= 1 else 0.0

        # gaze: expression gaze + small saccades while idle
        if st.status in ("idle",) and now >= self.next_glance:
            self.gaze_target = [random.uniform(-0.45, 0.45), random.uniform(-0.25, 0.25)]
            if random.random() < 0.35:
                self.gaze_target = [0.0, 0.0]
            self.next_glance = now + random.uniform(1.2, 3.5)
        elif st.status != "idle":
            self.gaze_target = [0.0, 0.0]
        self.gaze = [ease(self.gaze[i], self.gaze_target[i], dt, 14.0) for i in range(2)]

        # mouth: viseme shape while speaking
        if st.status == "speaking":
            vw, vh = MOUTH.get(getattr(st, "viseme", "rest"), MOUTH["rest"])
            vh = max(vh, 0.04 + 0.5 * st.mouth * vh)  # loudness adds a little openness
        else:
            vw, vh = MOUTH["rest"]
        self.mouth_w = ease(self.mouth_w, vw, dt, 22.0)
        self.mouth_h = ease(self.mouth_h, vh, dt, 22.0)
        self.mic = ease(self.mic, min(1.0, st.mic_level * 14), dt, 18.0)
        return expr

    # ------------------------------------------------------------ drawing -------
    @staticmethod
    def _draw_profile(prof, cx, cy, w, h, grow, color):
        sx = (w / 2 + grow) / (w / 2)
        sy = (h / 2 + grow) / (h / 2) if h > 1 else 1.0
        pts = []
        for x, t, b in prof:
            pts.append(rl.Vector2(cx + x * sx, cy + t * sy))
            pts.append(rl.Vector2(cx + x * sx, cy + b * sy))
        rl.draw_triangle_strip(pts, len(pts), color)

    def _eye_profile(self, w, h, top, tilt, happy, side):
        """[(x, top_y, bottom_y)] columns of one eye relative to its centre. side: -1 left, +1 right."""
        w2, h2 = w / 2, h / 2
        r = min(w2, h2) * 0.55
        n = 22
        pts = []
        for i in range(n + 1):
            x = -w2 + 2 * w2 * i / n
            ax = abs(x)
            # rounded-rectangle profile
            if ax > w2 - r:
                dy = math.sqrt(max(0.0, r * r - (ax - (w2 - r)) ** 2))
                t, b = -(h2 - r) - dy, (h2 - r) + dy
            else:
                t, b = -h2, h2
            # top lid: closure + slope (inner corner = toward the face centre)
            inner = -x * side / w2                 # +1 at inner corner, -1 at outer
            lid = -h2 + top * 2 * h2 + tilt * h2 * 0.9 * (inner + 1) / 2
            t = max(t, lid)
            # lower lid crescent (happy): rises most in the middle
            if happy > 0:
                bl = h2 - happy * 2 * h2 * max(0.0, 1 - (x / w2) ** 2) ** 0.8
                b = min(b, bl)
            if b < t:
                t = b = (t + b) / 2
            pts.append((x, t, b))   # drawn top-then-bottom: raylib culls back faces
        return pts

    def draw(self, W, H, dt):
        now = time.monotonic()
        expr = self._update(dt, now)
        st = self.state
        p = self.p
        rl.clear_background(col(BG))
        u = min(W / 100, H * 0.62 / 50)     # layout unit; face uses top ~62%
        cx, cy = W / 2, H * 0.31

        # soft halo behind the face
        rl.draw_circle_gradient(rl.Vector2(cx, cy), 42 * u, col(self.color, 26), col(BG, 0))

        eye_w, eye_h = 17 * u * p["w"], 23 * u * p["h"]
        gap = 14.5 * u
        gx = (self.gaze[0] + p["lx"]) * 4.5 * u
        gy = (self.gaze[1] + p["ly"]) * 4.0 * u
        blink = self.blink
        for side in (-1, 1):
            scale = 1 + p["asym"] * (0.5 if side == 1 else -0.5)
            w, h = eye_w * scale, eye_h * scale * (1 - 0.92 * blink)
            ex = cx + side * (gap + eye_w / 2) + gx
            ey = cy + gy
            top = p["top"] * (1 - blink)
            # one eye shape, drawn scaled up a few times for the glow, then solid on top
            prof = self._eye_profile(w, h, top, p["tilt"], p["happy"], side)
            for grow, a in ((4.6 * u, 10), (3.0 * u, 18), (1.7 * u, 28), (0.7 * u, 45)):
                self._draw_profile(prof, ex, ey, w, h, grow, col(self.color, a))
            self._draw_profile(prof, ex, ey, w, h, 0.0, col(self.color))
            if h > 7 * u and p["happy"] < 0.15 and p["top"] < 0.12:
                hx, hy = ex - w * 0.22, ey - h * 0.26 + p["top"] * h * 0.6
                rl.draw_circle(int(hx), int(hy), 1.9 * u, col(mixc(self.color, WHITE, 0.75), 200))

        self._draw_mouth(cx, cy + 21 * u, u, now)
        self._draw_header(W, H, u)
        self._draw_subtitles(W, H, u)

    def _draw_mouth(self, mx, my, u, now):
        st = self.state
        c = self.color
        if st.status == "thinking":
            for i in range(3):
                a = 0.5 + 0.5 * math.sin(now * 5 - i * 0.9)
                rl.draw_circle(int(mx + (i - 1) * 6 * u), int(my - 1.5 * a * u), 1.4 * u + a * 0.8 * u,
                               col(c, 90 + 165 * a))
            return
        if st.status == "listening":
            bars = 9
            for i in range(bars):
                k = 1 - abs(i - bars // 2) / (bars / 2 + 1)
                h = (1.2 + 9 * self.mic * k * (0.6 + 0.4 * math.sin(now * 13 + i * 1.7))) * u
                x = mx + (i - bars // 2) * 3.0 * u
                rl.draw_rectangle_rounded(rl.Rectangle(x - 0.8 * u, my - h / 2, 1.6 * u, h), 1.0, 6, col(c))
            return
        w = self.mouth_w * 30 * u
        h = max(1.6 * u, self.mouth_h * 13 * u)
        if st.status == "speaking" or self.mouth_h > 0.08:
            r = rl.Rectangle(mx - w / 2, my - h / 2, w, h)
            rl.draw_rectangle_rounded(r, 1.0, 16, col(c))
            if h > 4 * u:  # dark inner mouth
                ir = rl.Rectangle(mx - w * 0.36, my - h * 0.28, w * 0.72, h * 0.56)
                rl.draw_rectangle_rounded(ir, 1.0, 16, col(mixc(BG, c, 0.25)))
            return
        # idle: smile curve
        smile = self.p["smile"]
        pts = []
        for i in range(21):
            t = i / 20
            x = mx + (t - 0.5) * 20 * u
            y = my - 1.5 * u + smile * 4.5 * u * math.sin(math.pi * t)  # +smile bulges down = smile
            pts.append((x, y))
        th = 1.7 * u
        for a, b in zip(pts, pts[1:]):
            rl.draw_line_ex(rl.Vector2(*a), rl.Vector2(*b), th, col(c))
        for e in (pts[0], pts[-1]):
            rl.draw_circle(int(e[0]), int(e[1]), th / 2, col(c))

    def _text(self, font, text, size, color, center_x, y, max_w, max_lines, from_end=True):
        text = ascii_text(text)
        words, lines, cur = text.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if rl.measure_text_ex(font, t, size, 0).x <= max_w:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        lines = lines[-max_lines:] if from_end else lines[:max_lines]
        for line in lines:
            m = rl.measure_text_ex(font, line, size, 0)
            rl.draw_text_ex(font, line, rl.Vector2(center_x - m.x / 2, y), size, 0, color)
            y += size * 1.12
        return y

    def _draw_header(self, W, H, u):
        st = self.state
        if st.mode == "debate":
            label = "DEBATE" + (f"  ·  {st.debate_topic}" if st.debate_topic else "")
            label = ascii_text(label).replace("·", "-")
            size = 3.8 * u
            while rl.measure_text_ex(self.font_b, label, size, 0).x > W * 0.86 and len(label) > 10:
                label = label[:-4] + "..."
            m = rl.measure_text_ex(self.font_b, label, size, 0)
            r = rl.Rectangle(W / 2 - m.x / 2 - 2.4 * u, 1.6 * u, m.x + 4.8 * u, m.y + 1.2 * u)
            rl.draw_rectangle_rounded(r, 1.0, 12, col(DEBATE, 230))
            rl.draw_text_ex(self.font_b, label, rl.Vector2(r.x + 2.4 * u, r.y + 0.6 * u), size, 0, col(BG))
        tags = []
        if st.moving:
            tags.append("MOVING")
        if st.info:
            tags.append(st.info)
        elif getattr(st, "hint", "") and st.status in ("idle", "listening"):
            tags.append(st.hint)
        if tags:
            rl.draw_text_ex(self.font_r, ascii_text("  -  ".join(tags)), rl.Vector2(2 * u, H - 4.6 * u),
                            3.2 * u, 0, col(DIM))

    def _draw_subtitles(self, W, H, u):
        st = self.state
        y = H * 0.665
        if st.user_text:
            y = self._text(self.font_r, f'"{st.user_text}"', 3.7 * u, col(DIM), W / 2, y, W * 0.9, 1) + 0.6 * u
        if st.reply_text:
            self._text(self.font_b, st.reply_text, 4.6 * u, col(WHITE), W / 2, y, W * 0.9, 3)


def render_preview(state, path, size=(800, 480), seconds=1.0):
    """Debug helper: render the face for `seconds` into a PNG (needs a display)."""
    face = FaceRL(state, lambda: None, lambda: None, fullscreen=False, size=size)
    rt = rl.load_render_texture(size[0], size[1])
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        rl.begin_texture_mode(rt)
        face.draw(size[0], size[1], 1 / 60)
        rl.end_texture_mode()
        rl.begin_drawing(); rl.end_drawing()
    img = rl.load_image_from_texture(rt.texture)
    rl.image_flip_vertical(img)
    rl.export_image(img, path)
    rl.unload_image(img)
    rl.unload_render_texture(rt)
    rl.close_window()
