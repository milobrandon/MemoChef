#!/usr/bin/env python3
"""Generate an animated GIF of a cute cartoon shrimp cooking in a kitchen.

Run once to produce assets/cooking_shrimp.gif, then used by app.py at runtime.
"""

from PIL import Image, ImageDraw, ImageFont
import math, os

WIDTH, HEIGHT = 400, 400
FRAMES = 24
DURATION = 100  # ms per frame
BG_COLOR = (255, 245, 230)  # warm kitchen background

# Color palette
SHRIMP_BODY = (255, 130, 100)       # coral/salmon
SHRIMP_LIGHT = (255, 170, 140)      # lighter highlight
SHRIMP_DARK = (220, 90, 65)         # darker segments
SHRIMP_TAIL = (255, 110, 80)
EYE_WHITE = (255, 255, 255)
EYE_BLACK = (30, 30, 30)
CHEF_HAT = (255, 255, 255)
CHEF_HAT_BAND = (60, 60, 60)
PAN_COLOR = (80, 80, 85)
PAN_INSIDE = (50, 50, 55)
PAN_HANDLE = (100, 80, 60)
FLAME_ORANGE = (255, 160, 30)
FLAME_YELLOW = (255, 220, 50)
FLAME_RED = (255, 80, 20)
STEAM_COLOR = (220, 220, 220, 180)
COUNTER_COLOR = (180, 140, 100)
COUNTER_TOP = (200, 165, 125)
WALL_TILE = (230, 240, 235)
TILE_LINE = (210, 220, 215)
SPATULA_COLOR = (170, 170, 175)
SPATULA_HANDLE = (120, 90, 60)
CHEEK_COLOR = (255, 160, 150, 100)


def draw_kitchen_bg(draw: ImageDraw.Draw, frame: int):
    """Draw kitchen background with tiled wall and counter."""
    # Wall tiles
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=WALL_TILE)
    for y in range(0, 240, 50):
        draw.line([(0, y), (WIDTH, y)], fill=TILE_LINE, width=1)
    for x in range(0, WIDTH, 60):
        offset = 30 if (x // 60) % 2 == 0 else 0
        for y in range(0, 240, 50):
            draw.line([(x + offset, y), (x + offset, y + 50)], fill=TILE_LINE, width=1)

    # Counter
    draw.rectangle([0, 280, WIDTH, HEIGHT], fill=COUNTER_COLOR)
    draw.rectangle([0, 275, WIDTH, 290], fill=COUNTER_TOP)


def draw_flames(draw: ImageDraw.Draw, frame: int):
    """Draw animated flames under the pan."""
    flame_positions = [(145, 0), (170, 0), (195, 0), (220, 0), (245, 0)]
    for i, (bx, _) in enumerate(flame_positions):
        phase = (frame + i * 3) % FRAMES
        h = 12 + 8 * math.sin(phase * 2 * math.pi / FRAMES)
        top = int(278 - h)
        # Outer flame (orange/red)
        pts = [(bx, 278), (bx + 10, top - 3), (bx + 20, 278)]
        draw.polygon(pts, fill=FLAME_ORANGE)
        # Inner flame (yellow)
        inner_h = h * 0.6
        inner_top = int(278 - inner_h)
        pts2 = [(bx + 4, 278), (bx + 10, inner_top), (bx + 16, 278)]
        draw.polygon(pts2, fill=FLAME_YELLOW)


def draw_pan(draw: ImageDraw.Draw, frame: int):
    """Draw a frying pan on the counter."""
    # Pan body
    draw.ellipse([120, 255, 270, 285], fill=PAN_COLOR)
    draw.ellipse([130, 258, 260, 282], fill=PAN_INSIDE)
    # Handle
    draw.rounded_rectangle([260, 260, 340, 275], radius=4, fill=PAN_HANDLE)
    draw.rounded_rectangle([262, 263, 338, 272], radius=3, fill=(140, 110, 80))


def draw_steam(draw: ImageDraw.Draw, frame: int):
    """Draw rising steam wisps."""
    steam_img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    sd = ImageDraw.Draw(steam_img)
    for i, base_x in enumerate([160, 195, 230]):
        phase = (frame + i * 8) % FRAMES
        t = phase / FRAMES
        y = int(250 - t * 80)
        x = base_x + int(10 * math.sin(t * 4 * math.pi + i))
        alpha = int(150 * (1 - t))
        size = int(6 + t * 10)
        sd.ellipse(
            [x - size, y - size, x + size, y + size],
            fill=(230, 230, 230, alpha),
        )
    return steam_img


def draw_shrimp(draw: ImageDraw.Draw, frame: int):
    """Draw the cute cartoon shrimp chef character."""
    # Animation: gentle bob + arm wave
    bob = int(4 * math.sin(frame * 2 * math.pi / FRAMES))
    arm_angle = 15 * math.sin(frame * 2 * math.pi / FRAMES)
    cy = 210 + bob  # center y of shrimp body

    # --- Tail (left side) ---
    tail_pts = [
        (95, cy + 20),
        (75, cy + 30),
        (65, cy + 22),
        (75, cy + 15),
        (70, cy + 8),
        (80, cy + 10),
        (95, cy + 5),
    ]
    draw.polygon(tail_pts, fill=SHRIMP_TAIL)
    # Tail fin details
    draw.line([(80, cy + 10), (72, cy + 15)], fill=SHRIMP_DARK, width=2)
    draw.line([(80, cy + 18), (70, cy + 25)], fill=SHRIMP_DARK, width=2)

    # --- Body (curved shrimp shape) ---
    # Main body ellipse
    draw.ellipse([90, cy - 15, 200, cy + 35], fill=SHRIMP_BODY)
    # Segments (curved lines across body)
    for j in range(4):
        sx = 110 + j * 20
        draw.arc([sx - 8, cy - 10, sx + 8, cy + 30], 0, 180, fill=SHRIMP_DARK, width=2)

    # Lighter belly
    draw.ellipse([100, cy + 5, 190, cy + 30], fill=SHRIMP_LIGHT)

    # --- Head ---
    head_cx, head_cy = 205, cy - 5
    draw.ellipse([head_cx - 30, head_cy - 30, head_cx + 30, head_cy + 30], fill=SHRIMP_BODY)
    # Face highlight
    draw.ellipse([head_cx - 22, head_cy - 10, head_cx + 22, head_cy + 25], fill=SHRIMP_LIGHT)

    # Cheeks (blush)
    draw.ellipse([head_cx - 25, head_cy + 2, head_cx - 12, head_cy + 14], fill=(255, 170, 155))
    draw.ellipse([head_cx + 12, head_cy + 2, head_cx + 25, head_cy + 14], fill=(255, 170, 155))

    # Eyes
    for ex in [-12, 12]:
        # White
        draw.ellipse(
            [head_cx + ex - 8, head_cy - 18, head_cx + ex + 8, head_cy + 2],
            fill=EYE_WHITE,
            outline=(180, 180, 180),
            width=1,
        )
        # Pupil
        draw.ellipse(
            [head_cx + ex - 4, head_cy - 12, head_cx + ex + 4, head_cy - 2],
            fill=EYE_BLACK,
        )
        # Shine
        draw.ellipse(
            [head_cx + ex - 2, head_cy - 11, head_cx + ex + 2, head_cy - 7],
            fill=(255, 255, 255),
        )

    # Happy mouth
    draw.arc([head_cx - 10, head_cy + 2, head_cx + 10, head_cy + 18], 0, 180,
             fill=(180, 70, 50), width=2)

    # --- Antennae ---
    ant_base_y = head_cy - 28
    # Left antenna
    pts_l = [(head_cx - 8, ant_base_y + 5), (head_cx - 20, ant_base_y - 25),
             (head_cx - 15, ant_base_y - 30)]
    draw.line(pts_l, fill=SHRIMP_DARK, width=2)
    draw.ellipse([head_cx - 18, ant_base_y - 34, head_cx - 12, ant_base_y - 28],
                 fill=SHRIMP_BODY)
    # Right antenna
    pts_r = [(head_cx + 8, ant_base_y + 5), (head_cx + 20, ant_base_y - 25),
             (head_cx + 15, ant_base_y - 30)]
    draw.line(pts_r, fill=SHRIMP_DARK, width=2)
    draw.ellipse([head_cx + 12, ant_base_y - 34, head_cx + 18, ant_base_y - 28],
                 fill=SHRIMP_BODY)

    # --- Chef Hat ---
    hat_cy = head_cy - 28
    # Hat band
    draw.rectangle([head_cx - 22, hat_cy, head_cx + 22, hat_cy + 10], fill=CHEF_HAT)
    draw.rectangle([head_cx - 22, hat_cy + 6, head_cx + 22, hat_cy + 10], fill=CHEF_HAT_BAND)
    # Puffy top
    draw.ellipse([head_cx - 20, hat_cy - 28, head_cx + 20, hat_cy + 4], fill=CHEF_HAT)
    draw.ellipse([head_cx - 24, hat_cy - 18, head_cx - 4, hat_cy + 2], fill=CHEF_HAT)
    draw.ellipse([head_cx + 4, hat_cy - 18, head_cx + 24, hat_cy + 2], fill=CHEF_HAT)

    # --- Arms/Claws ---
    # Right arm holding spatula (animated)
    import math as m
    arm_end_x = head_cx + 35 + int(8 * m.sin(arm_angle * m.pi / 180))
    arm_end_y = cy + 5 + int(8 * m.cos(arm_angle * m.pi / 180))
    # Arm
    draw.line([(head_cx + 20, cy + 10), (arm_end_x, arm_end_y)],
              fill=SHRIMP_DARK, width=4)
    # Claw
    draw.ellipse([arm_end_x - 6, arm_end_y - 6, arm_end_x + 6, arm_end_y + 6],
                 fill=SHRIMP_BODY)

    # Spatula in right hand
    spat_x = arm_end_x + 5
    draw.line([(spat_x, arm_end_y), (spat_x + 5, arm_end_y + 40)],
              fill=SPATULA_HANDLE, width=3)
    draw.rounded_rectangle(
        [spat_x - 2, arm_end_y + 35, spat_x + 12, arm_end_y + 55],
        radius=3,
        fill=SPATULA_COLOR,
    )

    # Left arm (waving)
    wave_x = 90 + int(5 * m.sin(frame * 2 * m.pi / FRAMES + 1))
    wave_y = cy - 5 + int(8 * m.sin(frame * 2 * m.pi / FRAMES + 1))
    draw.line([(head_cx - 20, cy + 10), (wave_x, wave_y)],
              fill=SHRIMP_DARK, width=4)
    draw.ellipse([wave_x - 6, wave_y - 6, wave_x + 6, wave_y + 6],
                 fill=SHRIMP_BODY)

    # --- Little legs (under body) ---
    for lx in [120, 140, 160, 180]:
        ly = cy + 32
        wiggle = int(2 * math.sin(frame * 2 * math.pi / FRAMES + lx * 0.1))
        draw.line([(lx, cy + 28), (lx + wiggle, ly + 8)], fill=SHRIMP_DARK, width=2)


def draw_food_particles(draw: ImageDraw.Draw, frame: int):
    """Draw little food bits jumping in the pan."""
    particles = [(155, 0), (180, 1), (210, 2), (240, 3)]
    for bx, phase_off in particles:
        phase = (frame * 2 + phase_off * 6) % FRAMES
        t = phase / FRAMES
        # Parabolic hop
        py = int(265 - 20 * math.sin(t * math.pi))
        px = bx + int(5 * math.sin(t * 2 * math.pi))
        colors = [(255, 200, 50), (200, 100, 50), (120, 180, 80), (255, 150, 50)]
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=colors[phase_off])


def draw_sparkles(draw: ImageDraw.Draw, frame: int):
    """Draw little sparkle effects around the cooking."""
    sparkle_positions = [(130, 230), (270, 200), (300, 240), (100, 190)]
    for i, (sx, sy) in enumerate(sparkle_positions):
        phase = (frame + i * 6) % FRAMES
        if phase < FRAMES // 2:
            t = phase / (FRAMES // 2)
            size = int(3 + 4 * t)
            alpha_t = 1.0 - abs(t - 0.5) * 2
            c = int(255 * alpha_t)
            if c > 0:
                col = (255, 255, min(255, 100 + c))
                draw.line([(sx - size, sy), (sx + size, sy)], fill=col, width=1)
                draw.line([(sx, sy - size), (sx, sy + size)], fill=col, width=1)


def generate_gif(output_path: str = "assets/cooking_shrimp.gif"):
    frames = []
    for f in range(FRAMES):
        # Base frame (RGBA for steam compositing)
        img = Image.new("RGBA", (WIDTH, HEIGHT), (*BG_COLOR, 255))
        draw = ImageDraw.Draw(img)

        draw_kitchen_bg(draw, f)
        draw_flames(draw, f)
        draw_pan(draw, f)
        draw_food_particles(draw, f)
        draw_shrimp(draw, f)
        draw_sparkles(draw, f)

        # Composite steam (semi-transparent)
        steam = draw_steam(draw, f)
        img = Image.alpha_composite(img, steam)

        # Convert to P mode for GIF
        rgb = img.convert("RGB")
        frames.append(rgb)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=0,
        optimize=True,
    )
    print(f"Saved animated GIF to {output_path} ({len(frames)} frames)")


if __name__ == "__main__":
    generate_gif()
