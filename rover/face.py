"""Animated robot face for the HDMI screen (pygame). Scales to any resolution.

States drawn from the shared State object:
  idle      soft glow eyes, slow blinking, gentle smile, eyes wander a little
  listening eyes open wider, sound bars react to the microphone
  thinking  eyes glance up and to the side, three dots pulse
  speaking  mouth opens and closes with the actual audio loudness
  debate    orange colour, determined eyebrows, topic badge
Keys: Space / click / tap = interrupt (stop moving + stop talking), Esc or Q = quit,
      F = toggle fullscreen.
"""

import math
import random
import time

import pygame

from . import config

BG = (8, 11, 16)
NORMAL = (79, 209, 255)
DEBATE = (255, 159, 67)
THINK = (170, 140, 255)
DIM = (120, 130, 145)
WHITE = (235, 240, 245)


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


class Face:
    def __init__(self, state, on_interrupt, on_quit, fullscreen=True, size=(800, 480), on_brain=None):
        self.on_brain = on_brain          # B key: switch the language model
        self.state = state
        self.on_interrupt = on_interrupt
        self.on_quit = on_quit
        pygame.init()
        pygame.mouse.set_visible(True)   # pointer stays visible, also in fullscreen
        flags = pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((0, 0) if fullscreen else size, flags)
        pygame.display.set_caption(config.ROBOT_NAME)
        self.clock = pygame.time.Clock()
        self.fullscreen = fullscreen
        self._fonts = {}
        # animation state
        self.blink_at = time.monotonic() + 2
        self.blink_t = 0.0
        self.look = [0.0, 0.0]
        self.look_target = [0.0, 0.0]
        self.next_glance = time.monotonic() + 3
        self.eye_open = 1.0
        self.mouth = 0.0
        self.color = NORMAL
        self.mic = 0.0

    def font(self, px):
        px = max(10, int(px))
        if px not in self._fonts:
            self._fonts[px] = pygame.font.Font(None, px)
        return self._fonts[px]

    # ------------------------------------------------------------------ loop -----
    def run(self, fps=20):
        while self.state.alive:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.on_quit()
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.on_quit()
                    elif ev.key == pygame.K_SPACE:
                        self.on_interrupt()
                    elif ev.key == pygame.K_b and self.on_brain:
                        self.on_brain()
                    elif ev.key == pygame.K_f:
                        self.fullscreen = not self.fullscreen
                        pygame.display.set_mode((0, 0) if self.fullscreen else (800, 480),
                                                pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE)
                elif ev.type == pygame.FINGERDOWN or (ev.type == pygame.MOUSEBUTTONDOWN and self.fullscreen):
                    self.on_interrupt()  # tap the face to stop (in a window a click just focuses it)
            self.draw(time.monotonic())
            pygame.display.flip()
            self.clock.tick(fps)
        pygame.quit()

    # ------------------------------------------------------------------ drawing ---
    def _update(self, now):
        st = self.state
        target_color = DEBATE if st.mode == "debate" else NORMAL
        if st.status == "thinking":
            target_color = mix(target_color, THINK, 0.6)
        self.color = mix(self.color, target_color, 0.15)

        # blinking
        if now >= self.blink_at:
            self.blink_t = now
            self.blink_at = now + random.uniform(2.5, 6.0)
        b = (now - self.blink_t) / 0.16
        blink = 1 - math.sin(math.pi * b) if 0 <= b <= 1 else 1.0

        # where the eyes look
        if st.status == "thinking":
            self.look_target = [0.35, -0.45]
        elif st.status in ("listening", "speaking"):
            self.look_target = [0.0, 0.0]
        elif now >= self.next_glance:
            self.look_target = [random.uniform(-0.5, 0.5), random.uniform(-0.25, 0.25)]
            self.next_glance = now + random.uniform(1.5, 4.0)
        self.look = [lerp(self.look[0], self.look_target[0], 0.12), lerp(self.look[1], self.look_target[1], 0.12)]

        openness = {"listening": 1.12, "thinking": 0.8, "starting": 0.35}.get(st.status, 1.0)
        self.eye_open = lerp(self.eye_open, openness, 0.15) * 1.0
        self.mouth = lerp(self.mouth, st.mouth if st.status == "speaking" else 0.0, 0.5)
        self.mic = lerp(self.mic, min(1.0, st.mic_level * 12), 0.4)
        return blink

    def draw(self, now):
        st = self.state
        blink = self._update(now)
        scr = self.screen
        W, H = scr.get_size()
        scr.fill(BG)
        # layout unit: face lives in the top ~62% of the screen, subtitles below
        u = min(W / 100, H * 0.62 / 50)

        face_cy = H * 0.30
        eye_w, eye_h = 17 * u, 22 * u * self.eye_open * max(0.08, blink)
        gap = 15 * u
        lx, ly = self.look[0] * 4 * u, self.look[1] * 4 * u
        glow = mix(BG, self.color, 0.25)

        for side in (-1, 1):
            cx = W / 2 + side * (gap + eye_w / 2) + lx
            cy = face_cy + ly
            r = pygame.Rect(0, 0, eye_w, max(3, eye_h))
            r.center = (cx, cy)
            pygame.draw.rect(scr, glow, r.inflate(2.5 * u, 2.5 * u), border_radius=int(6 * u))
            pygame.draw.rect(scr, self.color, r, border_radius=int(5 * u))
            # highlight
            if eye_h > 6 * u:
                pygame.draw.circle(scr, mix(self.color, WHITE, 0.7),
                                   (int(r.left + eye_w * 0.3), int(r.top + eye_h * 0.25)), int(1.8 * u))
            if st.mode == "debate":  # determined eyebrows, slanting down toward the centre
                bx0 = cx - side * eye_w * 0.6
                bx1 = cx + side * eye_w * 0.6
                by = cy - 22 * u * self.eye_open / 2 - 4 * u
                pygame.draw.line(scr, self.color, (bx0, by + 3 * u), (bx1, by - 1.5 * u), int(2.2 * u))

        # mouth
        mouth_y = face_cy + 20 * u
        mw = 22 * u
        if st.status == "speaking" or self.mouth > 0.03:
            mh = 2 * u + self.mouth * 9 * u
            r = pygame.Rect(0, 0, mw * (0.75 + 0.25 * self.mouth), mh)
            r.center = (W / 2, mouth_y)
            pygame.draw.rect(scr, self.color, r, border_radius=int(mh / 2))
        elif st.status == "thinking":
            for i in range(3):
                a = 0.5 + 0.5 * math.sin(now * 5 - i * 0.9)
                pygame.draw.circle(scr, mix(BG, self.color, 0.3 + 0.7 * a),
                                   (int(W / 2 + (i - 1) * 6 * u), int(mouth_y)), int(1.6 * u + a * u))
        elif st.status == "listening":
            bars = 7
            for i in range(bars):
                h = (1.5 + 8 * self.mic * (0.5 + 0.5 * math.sin(now * 14 + i * 1.3))) * u
                x = W / 2 + (i - bars // 2) * 3.4 * u
                pygame.draw.line(scr, self.color, (x, mouth_y - h / 2), (x, mouth_y + h / 2), int(1.6 * u))
        else:  # gentle smile, drawn as a thick polyline with round caps
            pts = [(W / 2 + (t - 0.5) * mw, mouth_y - 2 * u + 4 * u * math.sin(math.pi * t))
                   for t in [i / 16 for i in range(17)]]
            th = max(2, int(1.8 * u))
            pygame.draw.lines(scr, self.color, False, pts, th)
            for p in (pts[0], pts[-1]):
                pygame.draw.circle(scr, self.color, (int(p[0]), int(p[1])), th // 2)

        self._draw_badges(scr, W, H, u)
        self._draw_subtitles(scr, W, H, u)

    def _draw_badges(self, scr, W, H, u):
        st = self.state
        f = self.font(4.2 * u)
        y = 2.5 * u
        if st.mode == "debate":
            label = "DEBATE" + (f": {st.debate_topic}" if st.debate_topic else "")
            txt = f.render(self._clip(label, f, W * 0.8), True, BG)
            r = txt.get_rect(midtop=(W / 2, y)).inflate(3 * u, 1.2 * u)
            pygame.draw.rect(scr, DEBATE, r, border_radius=int(2 * u))
            scr.blit(txt, txt.get_rect(center=r.center))
        tags = []
        if st.moving:
            tags.append("MOVING")
        if st.info:
            tags.append(st.info)
        if tags:
            t = self.font(3.6 * u).render("  ·  ".join(tags), True, DIM)
            scr.blit(t, (2.5 * u, H - t.get_height() - 1.5 * u))

    def _draw_subtitles(self, scr, W, H, u):
        st = self.state
        top = H * 0.66
        f_user = self.font(4.0 * u)
        f_bot = self.font(4.8 * u)
        y = top
        if st.user_text:
            for line in self._wrap("“" + st.user_text + "”", f_user, W * 0.9)[-1:]:
                s = f_user.render(line, True, DIM)
                scr.blit(s, s.get_rect(midtop=(W / 2, y)))
                y += s.get_height() + 0.6 * u
        if st.reply_text:
            for line in self._wrap(st.reply_text, f_bot, W * 0.9)[-3:]:
                s = f_bot.render(line, True, WHITE)
                scr.blit(s, s.get_rect(midtop=(W / 2, y)))
                y += s.get_height() + 0.3 * u

    @staticmethod
    def _wrap(text, font, width):
        words, lines, cur = text.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if font.size(t)[0] <= width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    @staticmethod
    def _clip(text, font, width):
        while text and font.size(text)[0] > width:
            text = text[:-2]
        return text
