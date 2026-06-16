"""
Road Map Annotation Platform v5
修复清单：
1. 路径配置：JSONL_PATH 指定单文件，IMAGE_ROOT 指定图片根目录，自动拼接避免重复
2. 批次改为下拉选择 + 卡片展示，不再用大表格
3. 彻底修复"点击开始标注卡住"：根本原因是 gr.Tabs(selected=...) 返回值在 Gradio 6
   里不能作为 outputs 使用，改用 gr.update() 不切换tab，让用户手动点标注工作台
4. 加入详细 debug 日志，终端打印每一步操作
5. share=True 修复远程访问
6. f-string 无反斜杠
"""

import gradio as gr
import json, os, threading, time, base64, io, math, traceback
from pathlib import Path
from PIL import Image, ImageDraw

# ══════════════════════════════════════════════════════════════
#  ① 路径配置 —— 按实际情况修改这里
# ══════════════════════════════════════════════════════════════
JSONL_PATH  = Path("data/sample_data.jsonl")   # ← 改成你的 jsonl 绝对路径
                                                 #   例如 Path("/cache/data/train.jsonl")
IMAGE_ROOT  = Path("")                           # ← 图片根目录，例如 Path("/cache/data")
                                                 #   jsonl 里的 image 字段已含 images/train/...
                                                 #   最终路径 = IMAGE_ROOT / record["image"]
                                                 #   若 jsonl 里已是绝对路径则留空 Path("")
OUTPUT_DIR  = Path("output")
STATE_DIR   = Path("state")
BATCH_SIZE  = 10000    # 每批数量，数据量小时自动调小

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

CATEGORIES = ["简单", "中等", "困难", "空白", "丢弃"]
CAT_COLORS = {"简单":"#22c55e","中等":"#f59e0b","困难":"#ef4444","空白":"#94a3b8","丢弃":"#6b7280"}
CAT_ICONS  = {"简单":"✅","中等":"⚡","困难":"🔥","空白":"⬜","丢弃":"🗑️"}

def dbg(msg):
    """统一 debug 输出，带时间戳"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════
#  全局状态
# ══════════════════════════════════════════════════════════════
_lock         = threading.Lock()
batch_locks   : dict = {}
user_progress : dict = {}
annotations   : dict = {}
all_data      : list = []
batches       : list = []

_load_pct     : int  = 0
_load_msg     : str  = "等待启动…"
_startup_done = threading.Event()
_startup_err  : str  = ""

# ══════════════════════════════════════════════════════════════
#  持久化
# ══════════════════════════════════════════════════════════════
def _save():
    try:
        with _lock:
            st = {"batch_locks":   dict(batch_locks),
                  "user_progress": dict(user_progress),
                  "annotations":   {k: dict(v) for k,v in annotations.items()}}
        (STATE_DIR/"state.json").write_text(
            json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        dbg(f"[WARN] 保存状态失败: {e}")

def _load_state():
    p = STATE_DIR/"state.json"
    if not p.exists():
        dbg("state.json 不存在，全新开始")
        return
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
        with _lock:
            batch_locks.update(st.get("batch_locks", {}))
            user_progress.update(st.get("user_progress", {}))
            for k, v in st.get("annotations", {}).items():
                annotations.setdefault(k, {}).update(v)
        dbg(f"历史进度恢复：{len(user_progress)} 用户，"
            f"{sum(len(v) for v in annotations.values())} 条标注")
    except Exception as e:
        dbg(f"[WARN] 加载状态失败: {e}")

# ══════════════════════════════════════════════════════════════
#  数据加载（后台线程）
# ══════════════════════════════════════════════════════════════
def _prog(pct, msg):
    global _load_pct, _load_msg
    _load_pct, _load_msg = pct, msg
    bar = "█"*(pct//5) + "░"*(20-pct//5)
    print(f"\r[{bar}] {pct:3d}%  {msg}", end="", flush=True)

def _resolve_image_path(img_field: str) -> str:
    """
    把 jsonl 里的 image 字段解析成绝对路径。
    - 若 IMAGE_ROOT 非空：path = IMAGE_ROOT / img_field
      （IMAGE_ROOT 已包含 images/ 的上级目录，img_field 以 images/ 开头）
    - 若 IMAGE_ROOT 为空且 img_field 是绝对路径：直接用
    - 否则相对当前工作目录
    """
    if not img_field:
        return ""
    p = Path(img_field)
    if IMAGE_ROOT and str(IMAGE_ROOT) != ".":
        resolved = IMAGE_ROOT / img_field
    elif p.is_absolute():
        resolved = p
    else:
        resolved = Path.cwd() / img_field
    return str(resolved)

def _load_data():
    global all_data, batches, _startup_err
    try:
        _prog(0, f"检查文件: {JSONL_PATH}")
        dbg(f"JSONL_PATH = {JSONL_PATH.resolve()}")
        dbg(f"IMAGE_ROOT = {IMAGE_ROOT.resolve() if str(IMAGE_ROOT) not in ('', '.') else '(使用jsonl中的路径)'}")

        if not JSONL_PATH.exists():
            msg = f"❌ 找不到文件: {JSONL_PATH.resolve()}"
            _prog(100, msg); _startup_err = msg
            dbg(msg); _startup_done.set(); return

        _prog(10, "读取 JSONL 文件…")
        lines = JSONL_PATH.read_text(encoding="utf-8").splitlines()
        dbg(f"文件行数: {len(lines)}")

        records = []
        bad = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                bad += 1
            if i % 10000 == 0 and i > 0:
                _prog(10 + int(i/len(lines)*50), f"解析中 {i}/{len(lines)}…")

        _prog(60, f"解析完成：{len(records)} 条有效，{bad} 条解析失败")
        dbg(f"有效记录: {len(records)}，解析失败: {bad}")

        if records:
            sample = records[0]
            img_field = sample.get("image","")
            resolved  = _resolve_image_path(img_field)
            exists    = os.path.exists(resolved) if resolved else False
            dbg(f"[路径检查] image字段='{img_field}'")
            dbg(f"[路径检查] 解析后路径='{resolved}'")
            dbg(f"[路径检查] 文件存在={exists}")
            if not exists:
                dbg("[WARN] 第一条样本图片不存在，将用占位图。"
                    "请检查 IMAGE_ROOT 配置是否正确。")

        all_data = records
        bsz = BATCH_SIZE if len(records) > BATCH_SIZE else max(10, len(records))
        batches.clear()
        nb = math.ceil(len(records)/bsz) if records else 0
        for i in range(nb):
            bid = f"batch_{i+1:04d}"
            s   = i * bsz
            e   = min(s + bsz, len(records))
            batches.append((bid, s, e))
            batch_locks.setdefault(bid, None)
            annotations.setdefault(bid, {})
        dbg(f"批次: {len(batches)} 个，每批最多 {bsz} 条")

        _prog(80, "恢复历史进度…")
        _load_state()

        _prog(100, f"✅ 就绪！{len(records)} 条 / {len(batches)} 批次")
        print()
        dbg("=== 启动完成，等待用户操作 ===")

    except Exception as e:
        err = traceback.format_exc()
        _startup_err = str(e)
        _prog(100, f"❌ 加载出错: {e}")
        print()
        dbg(f"[ERROR] 启动异常:\n{err}")
    finally:
        _startup_done.set()

threading.Thread(target=_load_data, daemon=True).start()

# ══════════════════════════════════════════════════════════════
#  渲染
# ══════════════════════════════════════════════════════════════
def _pct(a, b): return int(a/b*100) if b else 0

PLACEHOLDER_IMG = None  # lazy init

def _placeholder():
    global PLACEHOLDER_IMG
    if PLACEHOLDER_IMG is None:
        img = Image.new("RGB", (256,256), (30,40,52))
        draw = ImageDraw.Draw(img)
        draw.text((80,110), "无图片", fill=(100,116,139))
        buf = io.BytesIO(); img.save(buf, format="PNG")
        PLACEHOLDER_IMG = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return PLACEHOLDER_IMG

def render_sample(record) -> str:
    img_field = record.get("image", "")
    resolved  = _resolve_image_path(img_field)
    img = None
    if resolved and os.path.exists(resolved):
        try:
            img = Image.open(resolved).resize((256,256)).convert("RGB")
        except Exception as ex:
            dbg(f"[WARN] 图片读取失败 {resolved}: {ex}")
    if img is None:
        img = Image.new("RGB", (256,256), (30,40,52))

    draw = ImageDraw.Draw(img, "RGBA")
    gpt  = next((c["value"] for c in record.get("conversations",[])
                 if c.get("from") == "gpt"), "{}")
    try:
        road = json.loads(gpt)
    except Exception:
        road = {}

    def n2p(pt):
        return (int(pt[0]/1000*256), int(pt[1]/1000*256))

    for ln in road.get("lines", []):
        cat = ln.get("category", "centerline")
        pts = [n2p(p) for p in ln.get("points", [])]
        if len(pts) < 2:
            continue
        if cat == "intersection":
            draw.polygon(pts, fill=(255,109,0,55), outline=(255,109,0,200))
        else:
            for i in range(len(pts)-1):
                draw.line([pts[i], pts[i+1]], fill=(0,229,255,230), width=2)
            for pt in pts:
                r = 3
                draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r],
                             fill=(255,255,0,255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def render_at(bid, idx):
    """返回 (img_html, meta_dict, progress_html, debug_html)"""
    dbg(f"render_at: bid={bid}, idx={idx}")
    try:
        with _lock:
            bls = list(batches)
            ann = dict(annotations.get(bid, {}))
        bi = next((b for b in bls if b[0] == bid), None)
        if not bi:
            dbg(f"[ERROR] render_at: 批次 {bid} 不存在，当前批次={[b[0] for b in bls]}")
            return _err_html("批次不存在"), {}, "", f"批次 {bid} 不存在"
        _, s, e = bi
        recs = all_data[s:e]
        if not recs:
            dbg(f"[ERROR] render_at: 批次 {bid} 为空")
            return _err_html("批次为空"), {}, "", "批次数据为空"
        idx = max(0, min(int(idx), len(recs)-1))
        rec = recs[idx]
        rid = rec.get("id", str(idx))
        dbg(f"render_at: 渲染 rid={rid}, img={rec.get('image','')}")

        b64   = render_sample(rec)
        cat   = ann.get(rid, "—")
        badge = (
            f'<span style="background:{CAT_COLORS.get(cat,"#334155")};color:#fff;'
            f'padding:2px 9px;border-radius:10px;font-size:11px">'
            f'{CAT_ICONS.get(cat,"")} {cat}</span>'
            if cat != "—" else
            '<span style="color:#475569;font-size:11px;border:1px solid #334155;'
            'padding:2px 8px;border-radius:10px">未标注</span>'
        )
        img_html = (
            f'<div style="display:flex;flex-direction:column;gap:8px">'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
            f'<span style="color:#94a3b8;font-size:13px">'
            f'样本 <b style="color:#e2e8f0">{idx}</b> / {len(recs)-1}</span>'
            f'{badge}</div>'
            f'<img src="{b64}" style="width:256px;height:256px;border-radius:8px;'
            f'border:2px solid #334155;image-rendering:pixelated"/>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap">'
            f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;'
            f'color:#00e5ff;font-size:11px">━ 中心线</span>'
            f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;'
            f'color:#ff6d00;font-size:11px">⬡ 交叉口</span>'
            f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;'
            f'color:#ffff00;font-size:11px">● 节点</span>'
            f'</div></div>'
        )
        done  = len(ann)
        total = len(recs)
        p2    = _pct(done, total)
        prog  = (
            f'<div style="display:flex;align-items:center;gap:10px;margin-top:4px">'
            f'<div style="background:#1e293b;border-radius:6px;height:8px;width:180px">'
            f'<div style="background:#6366f1;height:8px;border-radius:6px;width:{p2}%"></div>'
            f'</div>'
            f'<span style="color:#94a3b8;font-size:13px">{done}/{total} ({p2}%)</span>'
            f'</div>'
        )
        img_path = _resolve_image_path(rec.get("image",""))
        debug = (
            f'<span style="color:#475569;font-size:11px">'
            f'rid={rid} | img_exists={os.path.exists(img_path) if img_path else False}'
            f'</span>'
        )
        return img_html, rec.get("meta", {}), prog, debug

    except Exception as ex:
        err = traceback.format_exc()
        dbg(f"[ERROR] render_at 异常:\n{err}")
        return _err_html(str(ex)), {}, "", str(ex)

def _err_html(msg):
    return (f'<div style="width:256px;padding:16px;background:#1e293b;border-radius:8px;'
            f'border:2px solid #ef4444;color:#ef4444;font-size:13px">❌ {msg}</div>')

# ══════════════════════════════════════════════════════════════
#  批次操作
# ══════════════════════════════════════════════════════════════
def _acquire(bid, user):
    with _lock:
        lk = batch_locks.get(bid)
        if lk is None or lk.get("user") == user:
            batch_locks[bid] = {"user": user, "since": time.time()}
            _save()
            dbg(f"[LOCK] {user} 锁定批次 {bid}")
            return True
    dbg(f"[LOCK] {user} 无法锁定 {bid}，当前持有者={batch_locks[bid].get('user')}")
    return False

def _release(bid, user):
    with _lock:
        lk = batch_locks.get(bid)
        if lk and lk.get("user") == user:
            batch_locks[bid] = None
            _save()
            dbg(f"[LOCK] {user} 释放批次 {bid}")

def _status_cell(blk, bid):
    lk = blk.get(bid)
    if lk:
        return f'<span style="color:#ef4444;font-weight:600">🔴 {lk["user"]}</span>'
    return '<span style="color:#22c55e;font-weight:600">🟢 空闲</span>'

def batch_cards_html(selected_bid=""):
    """批次状态卡片（横向滚动）"""
    with _lock:
        blk = dict(batch_locks)
        ann = {k: len(v) for k,v in annotations.items()}
        bls = list(batches)
    if not bls:
        return '<p style="color:#64748b;padding:12px">暂无批次，请检查 JSONL_PATH 配置</p>'
    cards = []
    for bid, s, e in bls:
        done  = ann.get(bid, 0)
        total = e - s
        p     = _pct(done, total)
        lk    = blk.get(bid)
        is_sel = bid == selected_bid
        if lk:
            status = f'🔴 {lk["user"]}'
            border = "#ef4444"
        else:
            status = "🟢 空闲"
            border = "#22c55e" if not is_sel else "#6366f1"
        bg = "#2d3f55" if is_sel else "#1e293b"
        cards.append(
            f'<div style="background:{bg};border:2px solid {border};border-radius:10px;'
            f'padding:12px 16px;min-width:160px;flex-shrink:0">'
            f'<div style="color:#e2e8f0;font-weight:600;font-size:14px">{bid}</div>'
            f'<div style="color:#64748b;font-size:11px;margin:2px 0">{s}–{e-1}</div>'
            f'<div style="background:#0f172a;border-radius:4px;height:8px;margin:8px 0">'
            f'<div style="background:#6366f1;height:8px;border-radius:4px;width:{p}%"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:11px">'
            f'<span style="color:#94a3b8">{done}/{total}</span>'
            f'<span style="color:#94a3b8">{status}</span>'
            f'</div></div>'
        )
    return (
        '<div style="display:flex;gap:10px;overflow-x:auto;padding:8px 0;flex-wrap:wrap">'
        + "".join(cards) + "</div>"
    )

def stats_html():
    with _lock:
        ann_all = {k: dict(v) for k,v in annotations.items()}
    total  = len(all_data)
    done   = sum(len(v) for v in ann_all.values())
    counts = {c: 0 for c in CATEGORIES}
    for v in ann_all.values():
        for cat in v.values():
            if cat in counts: counts[cat] += 1
    bars = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
        f'<span style="width:36px;color:#94a3b8;font-size:12px">{c}</span>'
        f'<div style="flex:1;background:#0f172a;border-radius:4px;height:16px">'
        f'<div style="background:{CAT_COLORS[c]};height:16px;border-radius:4px;'
        f'width:{_pct(counts[c],done)}%"></div></div>'
        f'<span style="color:#e2e8f0;font-size:12px;min-width:36px;text-align:right">'
        f'{counts[c]}</span></div>'
        for c in CATEGORIES
    )
    cards = "".join(
        f'<div style="background:#1e293b;border-radius:10px;padding:14px 20px;'
        f'min-width:90px;text-align:center">'
        f'<div style="font-size:26px;font-weight:700;color:{col}">{val}</div>'
        f'<div style="color:#64748b;font-size:11px;margin-top:4px">{lbl}</div></div>'
        for val,col,lbl in [
            (total, "#6366f1", "总样本"),
            (done,  "#22c55e", "已标注"),
            (total-done, "#f59e0b", "待标注"),
            (f"{_pct(done,total)}%", "#e2e8f0", "完成度"),
        ]
    )
    return (
        f'<div style="color:#e2e8f0">'
        f'<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">{cards}</div>'
        f'<div style="background:#1e293b;border-radius:10px;padding:16px;margin-bottom:16px">'
        f'<div style="color:#94a3b8;font-size:13px;font-weight:600;margin-bottom:10px">'
        f'各类别分布</div>{bars}</div>'
        f'<div style="background:#1e293b;border-radius:10px;padding:16px">'
        f'{batch_cards_html()}</div></div>'
    )

# ══════════════════════════════════════════════════════════════
#  加载进度 overlay
# ══════════════════════════════════════════════════════════════
def _overlay_html():
    if _startup_done.is_set():
        return ""  # 空字符串 → overlay 消失
    pct = _load_pct
    msg = _load_msg
    col = "#22c55e" if pct == 100 else "#6366f1"
    return (
        f'<div id="startup-overlay" style="position:fixed;top:0;left:0;'
        f'width:100vw;height:100vh;background:#0f172a;z-index:99999;'
        f'display:flex;align-items:center;justify-content:center">'
        f'<div style="background:#1e293b;border-radius:16px;padding:40px 48px;'
        f'text-align:center;border:1px solid #334155;min-width:380px;max-width:480px">'
        f'<div style="font-size:40px;margin-bottom:16px">🗺️</div>'
        f'<h2 style="color:#e2e8f0;font-size:20px;font-weight:700;margin:0 0 8px">'
        f'道路地图标注平台</h2>'
        f'<p style="color:#475569;font-size:13px;margin:0 0 24px">正在加载数据，请稍候…</p>'
        f'<div style="background:#0f172a;border-radius:8px;height:18px;overflow:hidden;margin-bottom:12px">'
        f'<div style="background:{col};height:18px;border-radius:8px;'
        f'width:{pct}%;transition:width 0.5s ease"></div></div>'
        f'<div style="color:#94a3b8;font-size:13px;margin-bottom:6px">{msg}</div>'
        f'<div style="color:#475569;font-size:12px">{pct}% 完成</div>'
        f'</div></div>'
    )

# ══════════════════════════════════════════════════════════════
#  Gradio App
# ══════════════════════════════════════════════════════════════
CSS = """
body,.gradio-container{background:#0f172a !important;color:#e2e8f0 !important}
footer{display:none!important}
.cat-btn{font-size:15px!important;font-weight:600!important;
         border-radius:8px!important;min-height:48px!important;width:100%!important}
label{color:#94a3b8!important}
.section-box{background:#1e293b;border-radius:12px;padding:16px;margin-bottom:12px;
             border:1px solid #334155}
"""

def build():
    with gr.Blocks(title="道路地图标注平台") as demo:
        s_user  = gr.State("")
        s_batch = gr.State("")
        s_idx   = gr.State(0)

        # ── 全屏 overlay（position:fixed 盖住所有内容）──
        overlay = gr.HTML(value=_overlay_html())

        # ── Header ──
        gr.HTML(
            '<div style="text-align:center;padding:20px 0 8px">'
            '<h1 style="color:#e2e8f0;font-size:26px;font-weight:700;margin:0">🗺️ 道路地图标注平台</h1>'
            '<p style="color:#475569;font-size:13px;margin:4px 0 0">'
            'Road Map BEV Annotation &nbsp;·&nbsp; 多用户并发 &nbsp;·&nbsp; 进度自动保存</p></div>'
        )

        with gr.Tabs() as tabs:

            # ════════════ Tab 1: 主页 ════════════
            with gr.Tab("🏠 主页 & 批次", id="home"):

                # 登录区
                with gr.Group():
                    gr.Markdown("### 👤 登录")
                    with gr.Row():
                        inp_user  = gr.Textbox(label="用户名", placeholder="输入用户名后点登录…",
                                               scale=3)
                        btn_login = gr.Button("登录 →", variant="primary", scale=1)
                    login_msg = gr.HTML("")

                gr.Markdown("---")

                # 批次选择区
                gr.Markdown("### 📦 选择批次")
                with gr.Row():
                    dd_batch  = gr.Dropdown(label="选择批次", choices=[], value=None, scale=3)
                    btn_enter = gr.Button("🚀 开始标注", variant="primary", scale=1)
                enter_msg = gr.HTML("")

                # 批次状态卡片
                gr.Markdown("#### 批次概览")
                btn_ref  = gr.Button("🔄 刷新状态", size="sm")
                batch_cards = gr.HTML("")

            # ════════════ Tab 2: 标注工作台 ════════════
            with gr.Tab("✏️ 标注工作台", id="annotate"):

                # 顶部状态栏
                with gr.Row():
                    lbl_batch_info = gr.HTML(
                        '<span style="color:#475569;font-size:13px">请先在主页选择批次</span>')
                    lbl_progress   = gr.HTML("")

                with gr.Row():
                    # 左：图片区
                    with gr.Column(scale=3, min_width=290):
                        img_disp = gr.HTML(
                            '<div style="width:256px;height:256px;background:#1e293b;'
                            'border:2px solid #334155;border-radius:8px;display:flex;'
                            'align-items:center;justify-content:center;color:#475569;font-size:13px">'
                            '请先选择批次</div>'
                        )
                        lbl_debug = gr.HTML("")       # debug 信息
                        meta_json = gr.JSON(label="样本元数据")

                    # 右：操作区
                    with gr.Column(scale=2):
                        gr.Markdown("### 🏷️ 分类标注")
                        gr.HTML('<p style="color:#64748b;font-size:12px;margin:0 0 12px">'
                               '点击分类后自动跳到下一个样本</p>')
                        cat_btns = {}
                        for cat in CATEGORIES:
                            cat_btns[cat] = gr.Button(
                                f"{CAT_ICONS[cat]}  {cat}",
                                elem_classes=["cat-btn"],
                                variant="secondary",
                            )

                        gr.Markdown("---")
                        gr.Markdown("### 🧭 导航")
                        with gr.Row():
                            btn_prev = gr.Button("◀ 上一个")
                            btn_next = gr.Button("▶ 下一个")
                        with gr.Row():
                            inp_jump = gr.Number(label="跳转到编号", value=0,
                                                 precision=0, scale=2)
                            btn_jump = gr.Button("跳转", scale=1)

                        gr.Markdown("---")
                        gr.Markdown("### 📤 导出 & 管理")
                        btn_export  = gr.Button("导出当前批次", variant="secondary")
                        export_out  = gr.Textbox(label="导出结果", lines=4,
                                                 interactive=False)
                        btn_release = gr.Button("🔓 释放批次锁", variant="stop", size="sm")
                        release_msg = gr.HTML("")

            # ════════════ Tab 3: 统计 ════════════
            with gr.Tab("📊 总体统计", id="stats"):
                btn_ref_s = gr.Button("🔄 刷新")
                stats_out = gr.HTML("")

        # ── Timer：每秒轮询启动状态 ──
        timer = gr.Timer(value=1.0, active=True)

        # ════════════════ Callbacks ════════════════

        def poll_startup():
            ov = _overlay_html()
            if _startup_done.is_set():
                choices = [b[0] for b in batches]
                val     = choices[0] if choices else None
                dbg(f"poll_startup: 就绪，{len(choices)} 批次")
                return (
                    ov,
                    gr.update(active=False),
                    gr.update(choices=choices, value=val),
                    batch_cards_html(),
                    stats_html(),
                )
            return (ov, gr.update(active=True),
                    gr.update(), gr.update(), gr.update())

        timer.tick(poll_startup,
                   outputs=[overlay, timer, dd_batch, batch_cards, stats_out])

        # ── 登录 ──
        def cb_login(user):
            user = (user or "").strip()
            if not user:
                return "", "<span style='color:#ef4444'>⚠️ 请输入用户名</span>"
            dbg(f"登录: {user}")
            with _lock:
                prog = user_progress.get(user)
            hint = ""
            if prog:
                hint = (f"<br><span style='color:#64748b;font-size:12px'>"
                        f"上次进度：{prog['batch_id']} 第 {prog['index']} 个样本</span>")
            return user, f"<span style='color:#22c55e'>✅ 欢迎，{user}！</span>{hint}"

        # ── 进入批次 ──
        def cb_enter(user, bid):
            dbg(f"cb_enter: user={user!r}, bid={bid!r}")
            if not user:
                return (gr.update(), 0,
                        "<span style='color:#ef4444'>⚠️ 请先登录</span>",
                        _err_html("未登录"), {}, "", "",
                        batch_cards_html())
            if not bid:
                return (gr.update(), 0,
                        "<span style='color:#ef4444'>⚠️ 请先选择批次</span>",
                        _err_html("未选批次"), {}, "", "",
                        batch_cards_html())
            ok = _acquire(bid, user)
            if not ok:
                with _lock:
                    lk = batch_locks.get(bid) or {}
                owner = lk.get("user", "其他人")
                msg = f"<span style='color:#ef4444'>🔴 批次 {bid} 正被 <b>{owner}</b> 处理中，请选其他批次</span>"
                return (gr.update(), 0, msg,
                        _err_html(f"批次已被 {owner} 锁定"), {}, "", "",
                        batch_cards_html(bid))

            with _lock:
                prog = user_progress.get(user, {})
            idx = prog.get("index", 0) if prog.get("batch_id") == bid else 0
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": idx}
            _save()

            ih, m, pg, dbg_html = render_at(bid, idx)
            batch_info = (
                f'<span style="color:#6366f1;font-weight:600">👤 {user}</span>'
                f' &nbsp;·&nbsp; <span style="color:#e2e8f0">📦 {bid}</span>'
            )
            cards = batch_cards_html(bid)
            dbg(f"cb_enter 成功：{bid} idx={idx}")
            return (
                bid, idx,
                f"<span style='color:#22c55e'>✅ 已锁定 {bid}，请切换到【标注工作台】开始标注</span>",
                ih, m, pg, batch_info,
                cards,
            )

        # 输出顺序：s_batch, s_idx, enter_msg, img_disp, meta_json,
        #           lbl_progress, lbl_batch_info, batch_cards
        ENTER_OUT = [s_batch, s_idx, enter_msg, img_disp, meta_json,
                     lbl_progress, lbl_batch_info, batch_cards]

        # ── 标注 ──
        def cb_annotate(user, bid, idx, cat):
            dbg(f"cb_annotate: user={user!r}, bid={bid!r}, idx={idx}, cat={cat!r}")
            if not bid:
                return idx, _err_html("请先选择批次"), {}, "", ""
            with _lock:
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return idx, _err_html(f"批次 {bid} 不存在"), {}, "", ""
            _, s, e = bi
            recs = all_data[s:e]
            idx  = max(0, min(int(idx), len(recs)-1))
            rid  = recs[idx].get("id", str(idx))
            with _lock:
                annotations.setdefault(bid, {})[rid] = cat
                ni = min(idx+1, len(recs)-1)
                user_progress[user] = {"batch_id": bid, "index": ni}
            _save()
            dbg(f"标注 {rid} → {cat}，下一个: {ni}")
            ih, m, pg, dh = render_at(bid, ni)
            return ni, ih, m, pg, dh

        ANNO_OUT = [s_idx, img_disp, meta_json, lbl_progress, lbl_debug]

        # ── 导航 ──
        def cb_nav(user, bid, idx, d):
            dbg(f"cb_nav: bid={bid!r}, idx={idx}, d={d}")
            if not bid:
                return idx, _err_html("请先选择批次"), {}, "", ""
            with _lock:
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return idx, _err_html("批次不存在"), {}, "", ""
            _, s, e = bi
            ni = max(0, min(int(idx)+d, e-s-1))
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": ni}
            _save()
            ih, m, pg, dh = render_at(bid, ni)
            return ni, ih, m, pg, dh

        def cb_jump(user, bid, j):
            j = max(0, int(j or 0))
            dbg(f"cb_jump: bid={bid!r}, j={j}")
            ih, m, pg, dh = render_at(bid, j)
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": j}
            _save()
            return j, ih, m, pg, dh

        # ── 导出 ──
        def cb_export(bid):
            dbg(f"cb_export: bid={bid!r}")
            if not bid:
                return "请先选择批次"
            with _lock:
                ann = dict(annotations.get(bid, {}))
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return "批次不存在"
            _, s, e = bi
            recs    = all_data[s:e]
            buckets = {c: [] for c in CATEGORIES}
            for r in recs:
                cat = ann.get(r.get("id", ""))
                if cat and cat in buckets:
                    buckets[cat].append(r)
            out = []
            for cat, lst in buckets.items():
                if lst:
                    p = OUTPUT_DIR / f"{bid}_{cat}.jsonl"
                    p.write_text(
                        "\n".join(json.dumps(r, ensure_ascii=False) for r in lst),
                        encoding="utf-8")
                    out.append(f"{cat}: {len(lst)} 条 → {p}")
                    dbg(f"导出 {cat}: {len(lst)} 条")
            return ("导出完成：\n" + "\n".join(out)) if out else "暂无已标注数据"

        # ── 释放 ──
        def cb_release(user, bid):
            dbg(f"cb_release: user={user!r}, bid={bid!r}")
            if not bid:
                return "<span style='color:#ef4444'>未选批次</span>"
            _release(bid, user)
            return f"<span style='color:#22c55e'>✅ 已释放 {bid}</span>"

        # ════════════════ Wire events ════════════════
        btn_login.click(cb_login, [inp_user], [s_user, login_msg])
        btn_ref.click(lambda: batch_cards_html(), outputs=[batch_cards])
        btn_ref_s.click(stats_html, outputs=[stats_out])

        btn_enter.click(cb_enter, [s_user, dd_batch], ENTER_OUT)

        for cat, btn in cat_btns.items():
            btn.click(
                lambda u, b, i, c=cat: cb_annotate(u, b, i, c),
                [s_user, s_batch, s_idx],
                ANNO_OUT,
            )

        btn_prev.click(lambda u,b,i: cb_nav(u,b,i,-1),
                       [s_user, s_batch, s_idx], ANNO_OUT)
        btn_next.click(lambda u,b,i: cb_nav(u,b,i,+1),
                       [s_user, s_batch, s_idx], ANNO_OUT)
        btn_jump.click(cb_jump, [s_user, s_batch, inp_jump], ANNO_OUT)
        btn_export.click(cb_export, [s_batch], [export_out])
        btn_release.click(cb_release, [s_user, s_batch], [release_msg])

    return demo


if __name__ == "__main__":
    dbg("=== 道路地图标注平台启动 ===")
    dbg(f"JSONL_PATH = {JSONL_PATH.resolve()}")
    dbg(f"IMAGE_ROOT = {IMAGE_ROOT}")
    dbg(f"OUTPUT_DIR = {OUTPUT_DIR.resolve()}")
    dbg(f"STATE_DIR  = {STATE_DIR.resolve()}")
    app = build()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,        # 远程服务器必须 True
        show_error=True,
        css=CSS,
    )
