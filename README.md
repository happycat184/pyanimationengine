# PyAnimationEngine

一个用 Python 编写、受 Manim 启发的 2D / 3D 动画引擎。它把简洁的场景 API 与显式的像素渲染管线分开，便于逐层替换和扩展。

## 渲染结构

```text
User API -> Converter -> ObjectCache -> Renderer -> FrameGenerator -> FrameBuffer -> Drawer
```

- **User API**：`Scene`、几何对象与动画对象组成场景。
- **Converter**：把对象几何变换到世界坐标和内部像素坐标。
- **ObjectCache**：先转换完整个场景时间轴，再将每帧的对象快照写入 `cache/<脚本名>.objects`。
- **Renderer**：按 `z_index` 与添加顺序排序。
- **Renderer3D**：对三角网格执行视锥裁剪、透视或正交投影、分块、逐像素深度比较、透明面排序与合成；与 2D 渲染器共用帧缓冲区。
- **FrameGenerator**：默认优先使用 WebGPU 计算着色器并行处理像素；不可用时回退到 CPU 扫描线填充与线段距离计算。
- **FrameBuffer**：保存一张 supersampled RGB 帧。
- **Drawer**：所有对象帧渲染完成后，再统一缩放并通过一次 FFmpeg 编码会话写出 MP4。

`.objects` 使用版本化的小端二进制格式：文件头记录分辨率、帧率、超采样、帧数和背景色；帧记录包含对象样式、float32 路径点，以及普通文字的 PNG 字形遮罩。v4 另存相机空间 float32 三维顶点、uint32 三角形索引、每面 RGB 和投影参数；仍可读取 v1–v3 缓存。TeX 文字已转成矢量路径。新格式缓存不需要目标机器安装原字体或 TeX 工具链即可重渲染；它保留转换时的文字外观和缓存中的分辨率设置。临时 RGB 帧在 MP4 编码完成后删除；`.objects` 缓存会保留。

将 `.objects` 文件复制到另一台机器后，可直接生成 MP4：

```bash
pyanimation --objects-cache cache/intro.objects -o intro.mp4
```

## 安装

```bash
python -m pip install -e .
```

依赖为 NumPy、Pillow、`imageio-ffmpeg` 和 wgpu-py。wgpu-py 通过 Vulkan / DirectX 12 等后端访问 AMD、NVIDIA 等 GPU；需要安装可用的 GPU 驱动。MP4 编码使用 H.264；若环境无法加载捆绑的 FFmpeg，可在 `PATH` 中安装 FFmpeg。

## 最小示例

```python
from pyanimation import BLUE, Circle, Create, Scene, Text


class FirstScene(Scene):
    def construct(self):
        title = Text("Hello animation", color="#f5f5f5")
        circle = Circle(radius=1.2, color=BLUE, stroke_width=3)
        self.play(Create(circle), run_time=1.5)
        self.play(title.animate.shift((0, 2)))
        self.wait(1)


FirstScene().render("hello.mp4")
```

也可以从命令行选择输出精度：

```bash
pyanimation examples/intro.py Intro -o intro.mp4 --width 1920 --height 1080 --fps 60 --antialias 2 --workers 16
```

已迁移的六册教材总览示例可直接运行：

```bash
python examples/curriculum_overview.py
```

`width`、`height` 和 `fps` 决定 MP4 输出精度；`antialias` 是内部 supersampling 倍数，增大后边缘更细腻，但每帧光栅计算量约按倍数的平方增长。自定义宽高需为偶数，以适配常见 H.264 4:2:0 输出。默认 `workers=16`，对象转换和独立帧渲染分别使用 16 个线程；`--workers 1` 可关闭并行，`--workers 0` 使用最多 4 个线程的自动模式。渲染帧会按原顺序写出。

默认 `--device auto` 会选择可用的硬件 GPU，AMD 与 NVIDIA 由 wgpu 后端自动适配；没有兼容 GPU 时显示原因并回退 CPU。可用 `--device gpu` 强制 GPU（不可用时直接报错），或用 `--device cpu` 关闭 GPU。Python API 对应 `RenderConfig(device="auto" | "gpu" | "cpu")`。2D 和 3D 均支持这些选项；转换与三角形准备在 CPU 上进行，GPU 逐像素并行光栅化。CPU 模式默认使用 16 个独立帧工作线程；GPU 模式按帧提交计算任务，避免多线程争用设备。wgpu-py 使用 Vulkan、DirectX 12 或 Metal，由平台和驱动选择后端。[wgpu-py 平台说明](https://wgpu-py.readthedocs.io/en/stable/start.html)

渲染期间会按阶段输出进度，例如 `[INFO In Rendering] 30/120 pixels=1920x1080`。阶段依次为 `Converting`、`Calculating`、`In Rendering`、`Writing`；默认每推进约 10% 输出一次，避免逐帧刷屏。

## 3D 对象与 2D 混合

```python
from pyanimation import (
    ThreeDScene, Group3D, Cube, Sphere, Text, Create, Rotate,
    Y_AXIS, PI, BLUE, TEAL,
)

class Solids(ThreeDScene):
    def construct(self):
        self.set_camera(position=(6, 4, 10), target=(0, 0, 0))
        shapes = Group3D(
            Cube(side_length=2, color=BLUE),
            Sphere(radius=1, color=TEAL),
        ).arrange_in_grid(columns=2)
        title = Text("2D title + 3D objects", font_size=0.45)
        title.move_to((0, 3)).set_z_index(10)
        self.add(title)
        self.play(Create(shapes), run_time=1)
        self.play(shapes.animate.rotate(2*PI, axis=Y_AXIS), run_time=3)
        self.wait(1)

Solids().render("solids.mp4")
```

简短示例可以直接运行：

```bash
python examples/hello_3d.py
```

完整示例包含六种几何体、绕不同轴旋转、遮挡与固定的二维标注：

```bash
python examples/three_d.py
python examples/three_d.py --device gpu --width 1920 --height 1080 --fps 60 --antialias 2
python examples/three_d.py --device cpu --workers 16 -o three_d_cpu.mp4
```

默认输出 `three_d.mp4`，转换缓存为 `cache/three_d.objects`。还可以使用统一 CLI：

```bash
pyanimation examples/three_d.py ThreeDGallery -o three_d.mp4 --device auto
pyanimation --objects-cache cache/three_d.objects -o three_d_replay.mp4
```

三维对象：`Cube`、`Cuboid`、`Sphere`、`Cylinder`、`Cone`、`Torus`、`Plane`、`Tetrahedron`、`Octahedron`、`Line3D`、`ThreeDAxes`；`Mesh3D(vertices, faces)` 接受三角网格，`Polyhedron(vertices, faces)` 对凸多边形面作扇形三角化，`Surface(function, u_range, v_range, resolution)` 采样参数曲面。用 `Group3D(a, b, ...).arrange_in_grid(columns=3)` 整理一组模型；`self.play(Create(group))` 会展开到组内模型，`group.animate.rotate(angle, axis=...)` 则能作为一个动画整体旋转。

- 坐标为右手系，Y 轴向上；`X_AXIS`、`Y_AXIS`、`Z_AXIS` 是三维方向，旋转角度以弧度计。二维 `(x, y)` 输入在网格方法中解释为 `(x, y, 0)`。
- `self.set_camera(...)` 在当前时间轴位置切换相机；`projection="orthographic"` 使用正交投影。相机支持 `position`、`target`、`up`、`near`、`far`、`focal_distance`、`light_direction`、`ambient`。未指定焦距时，目标平面的可见宽度等于 `frame_width`。
- 同一个 `z_index` 内的所有网格共享深度比较，再绘制该层的二维对象。较大的 `z_index` 整层覆盖较小的层。二维文字是固定屏幕标注，不会参与三维遮挡。
- 底层自行计算旋转矩阵、视图基、面法线、Lambert 漫反射、六平面裁剪、重心坐标、透视深度和 alpha 混合。CPU 使用 NumPy 分块计算；GPU 使用自编 [WGSL 计算着色器](https://www.w3.org/TR/WGSL/)，没有调用外部 3D 场景渲染器。透明三角面按每个像素的深度混合，密集透明几何体比不透明网格更慢。
- 网格支持 `Create` / `Uncreate`（按三角面顺序展开）、淡入淡出、缩放强调、`Rotate(..., axis=..., about_point=...)` 和 `.animate`。`Transform` 要求源和目标有相同顶点数量及三角形拓扑；完整旋转可用 `Rotate` 或 `.animate.rotate(...)`，两者都使用旋转矩阵。
- 三维面用 `fill_opacity` 设置透明度；`stroke_width` 不生成三维网格边线，需要线条时使用有实际半径的 `Line3D`。当前提供面级光照，没有纹理、投影阴影、PBR 材质或自动相机运动。

三维阶段日志保留统一的前缀：

```text
[INFO 3D Converting] 30/120 meshes=6 triangles=1364 cache=cache/three_d.objects
[INFO 3D Calculating] 30/120 meshes=6 triangles=1364
[INFO 3D In Rendering] 30/120 pixels=960x540 device=GPU
[INFO 3D Writing] 30/120 mixed_2d_3d_mp4_frame
```

运行几何、遮挡、裁剪、缓存、并行帧输出及 GPU/CPU 一致性检查：`python -m unittest discover -s tests -v`。没有可用 GPU 时会跳过 GPU 检查，其余使用 CPU。

## 字体与 TeX

`Text` 支持字体文件和字体族名，也可设置全局默认字体：

```python
from pyanimation import RenderConfig, Text

config = RenderConfig(font_path=r"C:\Windows\Fonts\arial.ttf")
label = Text("Custom font", font_path=r"C:\Windows\Fonts\arial.ttf")
family_label = Text("Arial family", font_family="Arial")
```

命令行可用 `--font "字体文件或字体族名"` 设置全局默认字体，对象级 `font_path` / `font_family` 优先级更高。

`Tex` 与 `MathTex` 会在转换阶段调用本机 TeX 编译器和 `dvisvgm`，将结果转成路径后进入 `.objects` 缓存与普通光栅流程。`tex_compiler="auto"` 会发现可用的 `latex`、`xelatex`、pTeX、`latexmk`、`pdflatex`、`lualatex` 或 `tectonic`，并在自动模式下尝试其他已安装引擎；也可显式设置编译器或 `dvisvgm_path`。DVI/XDV 是直接转 SVG 的首选链路；PDF 输出引擎需要所安装的 dvisvgm 带 PDF 处理器。安装 TeX Live、MiKTeX、MacTeX 或 Tectonic，并确保编译器和 `dvisvgm` 在 `PATH` 中：

```python
from pyanimation import MathTex, Tex

formula = MathTex(r"\int_0^1 x^2 \, dx = \frac{1}{3}")
heading = Tex("Vectorized by TeX")
unicode_heading = Tex(
    "你好",
    tex_compiler="xelatex",
    preamble=r"\usepackage{fontspec}\setmainfont{Microsoft YaHei}",
)
```

默认使用 `auto`；使用 XeLaTeX 的系统字体时，在 `preamble` 中通过 `fontspec` 选择字体。TeX 输出会被矢量化为路径，因而不依赖运行时 SVG 字体渲染器；dvisvgm 官方手册说明了字体转路径、DVI/XDV 输入以及 PDF 处理器要求。[dvisvgm 官方手册](https://dvisvgm.de/Manpage/)

## 当前 API 范围

2D 对象：`Circle`、`Ellipse`、`Square`、`Rectangle`、`RoundedRectangle`、`Polygon`、`RegularPolygon`、`Triangle`、`Star`、`Line`、`Arrow`、`DashedLine`、`Dot`、`Arc`、`Sector`、`Annulus`、`Text`、`Tex`、`MathTex`、`VGroup`。

3D 对象与相机：`Mesh3D`、`Polyhedron`、`Cube`、`Cuboid`、`Sphere`、`Cylinder`、`Cone`、`Torus`、`Plane`、`Tetrahedron`、`Octahedron`、`Surface`、`Line3D`、`ThreeDAxes`、`Group3D`、`Camera3D`、`ThreeDScene` 和 `Renderer3D`。

动画：`Create`、`Write`、`Uncreate`、`FadeIn`、`FadeOut`、`GrowFromCenter`、`ShrinkToCenter`、`DrawBorderThenFill`、`Indicate`、`Transform`、`ReplacementTransform`、`TransformFromCopy`、`Rotate`、`.animate`、`AnimationGroup`、`LaggedStart` 和 `Succession`；内置 `linear`、`smooth`、`there_and_back` 时间函数。

这是支持 2D / 3D 的引擎核心，不是 Manim 的完整兼容实现。当前没有 2D 相机帧控制、updater 系统、完整的 Manim 对象/动画集合，也未实现 Manim 插件兼容。`ThreeDScene` 已接入专用网格渲染器；`MovingCameraScene` 仍是扩展入口。普通 `Text` 使用 Pillow 字体绘制；`Tex` / `MathTex` 需要外部 TeX 工具链。

## 扩展渲染器

新增对象可以继承 `Mobject` 并提供本地坐标路径；`Converter` 将其转换成 `RenderObject`。新增光栅效果可以扩展 `FrameGenerator`，编码/输出设备则可替换 `Drawer`。场景构建只依赖这些接口，不绑定某个视频编码器。
