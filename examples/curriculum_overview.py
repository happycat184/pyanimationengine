"""初中数学六册教材的编者视角总览，使用 PyAnimationEngine 重写。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import ImageFont

from pyanimation import (
    Animation,
    AnimationGroup,
    Arrow,
    BLUE,
    Circle,
    Create,
    DOWN as _DOWN,
    Dot,
    FadeIn,
    FadeOut,
    GREEN,
    LEFT as _LEFT,
    Line,
    Mobject,
    ORIGIN as _ORIGIN,
    PURPLE,
    RED,
    Rectangle,
    RIGHT as _RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    UP as _UP,
    VGroup,
    WHITE,
    YELLOW,
    linear,
)
from pyanimation.fonts import resolve_font_file


# Use array vectors here so the scene can keep Manim's LEFT * n style.
ORIGIN = np.array(_ORIGIN, dtype=float)
UP = np.array(_UP, dtype=float)
DOWN = np.array(_DOWN, dtype=float)
LEFT = np.array(_LEFT, dtype=float)
RIGHT = np.array(_RIGHT, dtype=float)

BG = "#080a12"
GRID = "#60677a"
PRIMARY = BLUE
SECONDARY = GREEN
ACCENT = YELLOW
NEGATIVE = RED
PURPLE_ACCENT = PURPLE
TEXT = WHITE
GREY_B = "#777d8d"
BOLD = "bold"


def _select_cjk_font():
    """Choose an installed CJK font when available; cache freezes the result."""
    for family in (
        "Noto Serif CJK SC",
        "STSong",
        "SimSun",
        "Noto Sans SC",
        "Microsoft YaHei",
        "WenQuanYi Zen Hei",
    ):
        try:
            path = resolve_font_file(family)
            ImageFont.truetype(path, 24)
            return path
        except OSError:
            continue
    return None


CJK_FONT = _select_cjk_font()


def ctext(value, size=30, color=TEXT, weight=None, **kwargs):
    """Adapt the source video's point-sized Chinese labels to world units."""
    kwargs.pop("line_spacing", None)
    kwargs.pop("font", None)
    return Text(
        str(value),
        font_size=float(size) / 72.0,
        color=color,
        font_path=CJK_FONT,
        **kwargs,
    )


def _leaf_mobjects(items):
    leaves = []
    for item in items:
        if isinstance(item, VGroup):
            leaves.extend(_leaf_mobjects(item.submobjects))
        elif isinstance(item, Mobject):
            leaves.append(item)
        else:
            raise TypeError(f"expected Mobject or VGroup, got {type(item).__name__}")
    return leaves


def fade_in(*items):
    animations = [FadeIn(mob) for mob in _leaf_mobjects(items)]
    if len(animations) == 1:
        return animations[0]
    return AnimationGroup(*animations, lag_ratio=0.0)


def fade_out(*items):
    animations = [FadeOut(mob) for mob in _leaf_mobjects(items)]
    if len(animations) == 1:
        return animations[0]
    return AnimationGroup(*animations, lag_ratio=0.0)


def create_group(group):
    animations = [Create(mob) for mob in _leaf_mobjects((group,))]
    if len(animations) == 1:
        return animations[0]
    return AnimationGroup(*animations, lag_ratio=0.0)


def fit_width(mobject, max_width):
    """Shrink overlong text to a box width without enlarging short headings."""
    if 0 < mobject.width > max_width:
        mobject.set_width(max_width)
    return mobject


def make_grid(x_radius=7.0, y_radius=3.85, step=0.5):
    lines = VGroup()
    x = -x_radius
    while x <= x_radius + 1e-8:
        lines.add(Line((x, -y_radius), (x, y_radius), color=GRID, stroke_width=0.45, opacity=0.16))
        x += step
    y = -y_radius
    while y <= y_radius + 1e-8:
        lines.add(Line((-x_radius, y), (x_radius, y), color=GRID, stroke_width=0.45, opacity=0.16))
        y += step
    return lines


def vertical_brace(x, top, bottom, *, side="right", color=PURPLE_ACCENT, width=2.0):
    """A small vector bracket built from line segments instead of a Manim Brace."""
    inward = -1.0 if side == "right" else 1.0
    arm = 0.16
    middle = (top + bottom) / 2
    points = [
        (x, top),
        (x + inward * arm, top),
        (x + inward * arm, middle + 0.10),
        (x, middle),
        (x + inward * arm, middle - 0.10),
        (x + inward * arm, bottom),
        (x, bottom),
    ]
    return VGroup(*[
        Line(points[i], points[i + 1], color=color, stroke_width=width)
        for i in range(len(points) - 1)
    ])


class MoveAlongSegment(Animation):
    """Move a dot along a straight segment using the engine's Animation API."""

    def __init__(self, mobject, start, end, run_time=1.0, rate_func=linear):
        super().__init__(mobject, run_time=run_time, rate_func=rate_func)
        self.start_point = np.asarray(start, dtype=float)
        self.end_point = np.asarray(end, dtype=float)

    def begin(self, scene):
        super().begin(scene)
        self.mobject.move_to(self.start_point)

    def interpolate(self, alpha):
        t = self.rate_func(max(0.0, min(1.0, alpha)))
        self.mobject.move_to(self.start_point * (1 - t) + self.end_point * t)


class ChapterScene(Scene):
    """Reusable black canvas, subtle grid, and section transitions."""

    def setup(self):
        self.config.background_color = BG
        self.grid = make_grid()
        self.add(self.grid)

    def _grid_ids(self):
        return {id(mob) for mob in _leaf_mobjects((self.grid,))}

    def _stage_mobjects(self, keep_grid=True):
        grid_ids = self._grid_ids() if keep_grid else set()
        return [mob for mob in self.mobjects if id(mob) not in grid_ids]

    def clear_stage(self, keep_grid=True, run_time=0.45):
        stage = self._stage_mobjects(keep_grid=keep_grid)
        if stage:
            self.play(fade_out(*stage), run_time=run_time)

    def hold_summary(self):
        self.wait(2.0)

    def chapter_break(self, label):
        self.clear_stage(keep_grid=False, run_time=0.35)
        cover = Rectangle(width=14.35, height=8.15, color=BG, fill_opacity=1.0, stroke_width=0)
        self.add(cover)
        text = ctext(label, size=36, color=TEXT, weight=BOLD)
        self.play(fade_in(text), run_time=0.5)
        self.wait(0.35)
        self.play(fade_out(text), run_time=0.4)
        self.clear()
        self.add(self.grid)


class CurriculumOverview(ChapterScene):
    """七上到九下六册教材的总设计逻辑。"""

    def construct(self):
        self.opening()
        self.overall_view()

        volumes = [
            {
                "book": "七上",
                "title": "从小学走进初中：先学会用字母和方程",
                "position": ["小学算术 → 初中代数", "用正负数、字母和等号说清问题"],
                "core": ["有理数、整式的加减", "一元一次方程、线段和角"],
                "design": ["先把数的范围扩大到负数", "再用字母表示未知数", "最后用方程找出未知数"],
                "link": ["以后学方程、函数都要用这里的运算", "线段和角是后面几何的第一步"],
                "algebra": "数 → 字母 → 方程",
                "geometry": "线段 · 角 · 几何语言",
            },
            {
                "book": "七下",
                "title": "几何入门 + 二元一次方程：开始学着说明理由",
                "position": ["从会算到会解释", "用图形和方程一起解决问题"],
                "core": ["相交线与平行线", "二元一次方程组、实数、不等式、坐标系"],
                "design": ["看到图形时，不只说答案，还要说为什么", "两个未知数一起出现，问题更贴近实际", "坐标系先搭好，后面函数才能画图"],
                "link": ["这里的‘因为……所以……’会变成八上的证明", "方程组和坐标系会在函数中继续使用"],
                "algebra": "实数 · 方程组 · 坐标",
                "geometry": "相交线 · 平行线 · 简单说理",
            },
            {
                "book": "八上",
                "title": "难度上升：从说理由到写证明",
                "position": ["学会完整证明", "每一步都要有根据"],
                "core": ["三角形、全等、轴对称", "整式乘法、因式分解、分式"],
                "design": ["先学会写已知和求证", "再用全等证明两个图形相同", "代数上学会把式子变形"],
                "link": ["因式分解会帮助你解二次方程"],
                "algebra": "整式变形 · 因式分解 · 分式",
                "geometry": "三角形 · 全等 · 轴对称 · 证明",
            },
            {
                "book": "八下",
                "title": "初中分水岭：第一次系统学习函数",
                "position": ["知识主干册", "代数和几何都开始连成体系"],
                "core": ["二次根式、勾股定理、平行四边形", "一次函数、数据分析"],
                "design": ["根式让实数计算更完整", "一次函数教你看变量、图像和关系式", "四边形把证明和计算放到一起"],
                "link": ["一次函数的学法会迁移给二次函数", "勾股和四边形会反复出现在综合题"],
                "algebra": "二次根式 · 一次函数 · 数据分析",
                "geometry": "勾股定理 · 平行四边形",
            },
            {
                "book": "九上",
                "title": "综合难度上升：二次函数、圆和概率",
                "position": ["综合提升册", "开始遇到需要分类讨论的问题"],
                "core": ["一元二次方程、二次函数", "旋转、圆、概率初步"],
                "design": ["二次方程比一元一次更复杂", "二次函数要同时看图像和式子", "概率要判断各种可能，容易混淆"],
                "link": ["二次函数和圆会与九下知识合在一起", "概率训练你整理信息、分类思考"],
                "algebra": "一元二次方程 · 二次函数 · 概率",
                "geometry": "旋转 · 圆 · 综合几何",
            },
            {
                "book": "九下",
                "title": "收口：投影与视图 + 总复习",
                "position": ["收官复习册", "把六册学过的方法连起来"],
                "core": ["相似三角形、锐角三角函数", "投影与视图、总复习"],
                "design": ["相似把图形的边按比例联系起来", "三角函数帮助计算角和边", "投影与视图帮助你看懂立体图形"],
                "link": ["总复习把知识整理成方法清单", "中考题会把函数、圆、相似、三角函数放在一起"],
                "algebra": "三角函数 · 综合建模 · 总复习",
                "geometry": "相似 · 投影与视图 · 综合复习",
            },
        ]

        for index, volume in enumerate(volumes):
            self.present_volume(index, volume)
        self.closing_summary()

    def opening(self):
        title = ctext("一个视频带你看透初中数学", size=48, color=TEXT, weight=BOLD)
        rule = Line(LEFT * 3.6, RIGHT * 3.6, color=PRIMARY, stroke_width=5)
        title.move_to(UP * 0.45)
        rule.next_to(title, DOWN, buff=0.3)
        self.play(fade_in(title), Create(rule), run_time=1.3)
        self.wait(1.0)
        self.play(fade_out(title, rule), run_time=0.7)
        self.mind_map()

    def mind_map(self):
        root = RoundedRectangle(width=2.3, height=0.72, corner_radius=0.14, color=ACCENT, fill_opacity=0.12)
        root.move_to(LEFT * 5.7)
        root_text = ctext("初中数学教材", size=22, color=ACCENT, weight=BOLD).move_to(root)
        self.play(Create(root), fade_in(root_text), run_time=0.8)

        grade_brace = vertical_brace(-4.15, 2.15, -2.15, side="left", color=PRIMARY)
        self.play(create_group(grade_brace), run_time=0.7)

        grades = [
            ("七年级", 1.75, "七上：代数入门", "学习目标：从小学算术过渡到代数语言", "七下：说理起步", "学习目标：学会说明理由和列方程", "主要难点：从会算到会解释"),
            ("八年级", 0.0, "八上：严格证明", "学习目标：每一步都写出根据", "八下：函数与图形主干", "学习目标：学会研究变量和图像", "主要难点：从说理到严格证明，函数首次成体系"),
            ("九年级", -1.75, "九上：综合难点", "学习目标：处理二次函数、圆和概率", "九下：投影视图与总复习", "学习目标：把六册方法连成解题网络", "主要难点：综合建模与知识收口"),
        ]
        for grade, row_y, upper_title, upper_goal, lower_title, lower_goal, difficulty in grades:
            grade_box = RoundedRectangle(width=1.35, height=0.58, corner_radius=0.1, color=PRIMARY)
            grade_box.move_to(LEFT * 3.25 + UP * row_y)
            grade_text = ctext(grade, size=23, color=PRIMARY, weight=BOLD).move_to(grade_box)
            self.play(Create(grade_box), fade_in(grade_text), run_time=0.55)

            upper = self.goal_card(upper_title, upper_goal, SECONDARY)
            lower = self.goal_card(lower_title, lower_goal, SECONDARY)
            pair = VGroup(upper, lower).arrange(DOWN, buff=0.12).move_to(UP * row_y)
            branch = Line(grade_box.get_right(), pair.get_left(), color=SECONDARY, stroke_width=2)
            brace = vertical_brace(pair.get_right()[0] + 0.12, pair.get_top()[1], pair.get_bottom()[1], side="right", color=PURPLE_ACCENT)
            difficulty_text = ctext(difficulty, size=16, color=PURPLE_ACCENT)
            fit_width(difficulty_text, 3.15)
            difficulty_text.next_to(brace, RIGHT, buff=0.16)
            self.play(Create(branch), fade_in(upper), run_time=0.55)
            self.play(fade_in(lower), create_group(brace), fade_in(difficulty_text), run_time=0.7)

        self.hold_summary()
        stage = self._stage_mobjects(keep_grid=True)
        if stage:
            self.play(fade_out(*stage), run_time=0.7)

    def goal_card(self, title, goal, color):
        box = RoundedRectangle(width=3.55, height=0.62, corner_radius=0.1, color=color, fill_opacity=0.08)
        heading = ctext(title, size=16, color=color, weight=BOLD)
        body = ctext(goal, size=12, color=TEXT)
        fit_width(heading, 3.28)
        fit_width(body, 3.28)
        center = np.asarray(box.get_center(), dtype=float)
        heading.move_to(center + UP * 0.14)
        body.move_to(center + DOWN * 0.14)
        return VGroup(box, heading, body)

    def make_roadmap(self, active_index):
        algebra_y = 1.55
        geometry_y = 0.78
        left_x, right_x = -5.7, 5.7
        algebra_line = Arrow((left_x + 0.05, algebra_y), (right_x - 0.15, algebra_y), color=PRIMARY, stroke_width=4)
        geometry_line = Arrow((left_x + 0.05, geometry_y), (right_x - 0.15, geometry_y), color=SECONDARY, stroke_width=4)
        algebra_label = ctext("代数主线", size=22, color=PRIMARY).move_to((left_x + 0.5, algebra_y + 0.35))
        geometry_label = ctext("几何主线", size=22, color=SECONDARY).move_to((left_x + 0.5, geometry_y - 0.35))
        positions = [-4.8, -2.9, -1.0, 1.0, 2.9, 4.8]
        nodes = VGroup()
        for index, x in enumerate(positions):
            color = ACCENT if index == active_index else GREY_B
            nodes.add(Dot((x, algebra_y), radius=0.1 if index == active_index else 0.07, color=color))
            nodes.add(Dot((x, geometry_y), radius=0.1 if index == active_index else 0.07, color=color))
            nodes.add(ctext(["七上", "七下", "八上", "八下", "九上", "九下"][index], size=16, color=color).move_to((x, 0.42)))
        roadmap = VGroup(algebra_line, geometry_line, algebra_label, geometry_label, nodes)
        return roadmap, algebra_line, geometry_line, positions[active_index]

    def overall_view(self):
        heading = ctext("编者总视角：两条主线，六次递进", size=32, color=ACCENT, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(fade_in(heading), run_time=0.8)
        roadmap, algebra_line, geometry_line, _ = self.make_roadmap(-1)
        self.play(Create(algebra_line), Create(geometry_line), run_time=1.2)
        self.play(fade_in(roadmap[2:]), run_time=0.8)
        moving_a = Dot(algebra_line.start, color=ACCENT, radius=0.09)
        moving_g = Dot(geometry_line.start, color=ACCENT, radius=0.09)
        self.add(moving_a, moving_g)
        self.play(
            MoveAlongSegment(moving_a, algebra_line.start, algebra_line.end, rate_func=linear),
            MoveAlongSegment(moving_g, geometry_line.start, geometry_line.end, rate_func=linear),
            run_time=2.2,
        )
        self.remove(moving_a, moving_g)

        lines = VGroup(
            ctext("代数主线：数  →  式  →  方程  →  函数", size=25, color=PRIMARY),
            ctext("几何主线：基本图形  →  推理  →  全等  →  四边形  →  圆  →  相似", size=25, color=SECONDARY),
        ).arrange(DOWN, buff=0.25).move_to(DOWN * 1.55)
        self.play(fade_in(lines[0]), run_time=0.8)
        self.play(fade_in(lines[1]), run_time=0.8)
        self.wait(2.0)
        self.play(fade_out(heading, roadmap, lines), run_time=0.7)

    def text_block(self, title, body_lines, color, center):
        box = RoundedRectangle(width=6.2, height=1.35, corner_radius=0.12, color=color, stroke_width=2, fill_opacity=0.06)
        box.move_to(center)
        heading = ctext(title, size=16, color=color, weight=BOLD)
        body = VGroup(*[ctext(line, size=13, color=TEXT) for line in body_lines]).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        fit_width(heading, 5.82)
        fit_width(body, 5.82)
        center = np.asarray(box.get_center(), dtype=float)
        heading.move_to(center + UP * 0.49)
        body.move_to(center + DOWN * 0.21)
        return VGroup(box, heading, body)

    def branch_tags(self, active_x, algebra_text, geometry_text):
        tag_a = ctext(algebra_text, size=18, color=PRIMARY)
        tag_g = ctext(geometry_text, size=18, color=SECONDARY)
        tag_x = max(-2.6, min(2.6, active_x))
        tag_a.move_to((tag_x, 2.18))
        tag_g.move_to((tag_x, 0.12))
        return tag_a, tag_g

    def present_volume(self, index, volume):
        self.chapter_break(f"{volume['book']}  ·  {volume['title']}")
        heading = ctext(f"{volume['book']}：{volume['title']}", size=30, color=ACCENT, weight=BOLD).to_edge(UP, buff=0.25)
        if heading.width > 13.2:
            heading.set_width(13.2)
        roadmap, algebra_line, geometry_line, active_x = self.make_roadmap(index)
        tag_a, tag_g = self.branch_tags(active_x, volume["algebra"], volume["geometry"])
        self.play(fade_in(heading), Create(algebra_line), Create(geometry_line), fade_in(roadmap[2:]), run_time=1.0)
        self.play(fade_in(tag_a), fade_in(tag_g), run_time=0.8)

        blocks = VGroup(
            self.text_block("这一册要学会什么", volume["position"], ACCENT, LEFT * 3.35 + DOWN * 0.72),
            self.text_block("主要学什么", volume["core"], PRIMARY, RIGHT * 3.35 + DOWN * 0.72),
            self.text_block("为什么这样安排", volume["design"], SECONDARY, LEFT * 3.35 + DOWN * 2.28),
            self.text_block("以后用在哪里", volume["link"], PURPLE_ACCENT, RIGHT * 3.35 + DOWN * 2.28),
        )
        for block in blocks:
            self.play(fade_in(block), run_time=0.55)
        self.hold_summary()

    def closing_summary(self):
        self.chapter_break("六册教材：从基础语言走向综合建模")
        title = ctext("教材整套编写逻辑", size=34, color=ACCENT, weight=BOLD)
        lines = VGroup(
            ctext("七上学会用数学语言，七下学会说明理由；", size=27, color=TEXT),
            ctext("八上学会写严格证明，八下学会研究函数；", size=27, color=TEXT),
            ctext("九上处理概率和压轴难点，九下把方法连起来。", size=27, color=TEXT),
            ctext("六册从会算、会说理，走到会证明、会做综合题。", size=25, color=SECONDARY),
        ).arrange(DOWN, buff=0.24)
        group = VGroup(title, lines).arrange(DOWN, buff=0.42).move_to(ORIGIN)
        self.play(fade_in(title), run_time=0.8)
        for line in lines:
            self.play(fade_in(line), run_time=0.6)
        self.hold_summary()


if __name__ == "__main__":
    CurriculumOverview().render("curriculum_overview.mp4")
