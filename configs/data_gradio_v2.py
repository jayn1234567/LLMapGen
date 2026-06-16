"""
Road Map Annotation Platform v3.1
- Fixed: JS bridge button click issue (DOM removal due to visible=False)
- Fixed: share=True for remote servers
- Added: table inline "Enter" button with auto-redirect
- Added: separate images directory logic
"""

import gradio as gr
import json, os, threading, time, base64, io, math, sys
from pathlib import Path
from PIL import Image, ImageDraw

# ─────────────────────────── Config ───────────────────────────
DATA_DIR        = Path("data")
IMAGE_DIR       = Path("images")
OUTPUT_DIR      = Path("output")
STATE_DIR       = Path("state")
BATCH_SIZE      = 10000   

CATEGORIES = ["简单", "中等", "困难", "空白", "丢弃"]
CAT_COLORS = {"简单":"#22c55e","中等":"#f59e0b","困难":"#ef4444","空白":"#94a3b8","丢弃":"#6b7280"}
CAT_ICONS  = {"简单":"✅","中等":"⚡","困难":"🔥","空白":"⬜","丢弃":"🗑️"}

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

# ─────────────────────────── Global state ─────────────────────
_lock         = threading.Lock()
batch_locks   : dict = {}
user_progress : dict = {}
annotations   : dict = {}
all_data      : list = []
batches       : list = []

# startup progress (0-100)
_load_progress   : int = 0
_load_status_msg : str = "等待启动…"

# ─────────────────────────── Persistence ──────────────────────
def _save():
    with _lock:
        st = {
            "batch_locks":   {k: v for k, v in batch_locks.items()},
            "user_progress": dict(user_progress),
            "annotations":   {k: dict(v) for k, v in annotations.items()},
        }
    (STATE_DIR / "state.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=2))

def _load_state():
    p = STATE_DIR / "state.json"
    if not p.exists():
        return
    st = json.loads(p.read_text())
    with _lock:
        batch_locks.update(st.get("batch_locks", {}))
        user_progress.update(st.get("user_progress", {}))
        for k, v in st.get("annotations", {}).items():
            annotations.setdefault(k, {}).update(v)

# ─────────────────────────── Data loading (with progress) ─────
def _set_progress(pct: int, msg: str):
    global _load_progress, _load_status_msg
    _load_progress   = pct
    _load_status_msg = msg
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r[{bar}] {pct:3d}%  {msg}", end="", flush=True)

def load_all_data_with_progress():
    global all_data, batches

    _set_progress(0, "扫描 data/ 目录…")
    files = sorted(DATA_DIR.glob("*.jsonl"))
    if not files:
        _set_progress(100, "⚠️  data/ 目录下未找到 .jsonl 文件")
        return 0

    records = []
    total_files = len(files)
    for fi, f in enumerate(files):
        _set_progress(int(fi / total_files * 60),
                      f"读取文件 {f.name} ({fi+1}/{total_files})…")
        lines = f.read_text(encoding="utf-8").splitlines()
        total_lines = len(lines)
        for li, line in enumerate(lines):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
            if total_lines > 5000 and li % 5000 == 0:
                inner = int((fi + li / total_lines) / total_files * 60)
                _set_progress(inner,
                    f"解析 {f.name}: {li}/{total_lines} 行…")

    _set_progress(65, f"共加载 {len(records)} 条记录，构建批次…")
    all_data = records

    bsz = BATCH_SIZE
    batches.clear()
    n_batches = math.ceil(len(records) / bsz) if records else 0
    for i in range(n_batches):
        bid   = f"batch_{i+1:04d}"
        start = i * bsz
        end   = min(start + bsz, len(records))
        batches.append((bid, start, end))
        batch_locks.setdefault(bid, None)
        annotations.setdefault(bid, {})
        if n_batches > 0:
            _set_progress(65 + int(i / n_batches * 15),
                          f"构建批次 {bid}…")

    _set_progress(80, "加载历史进度…")
    _load_state()

    _set_progress(100, f"✅ 就绪！共 {len(records)} 条 / {len(batches)} 批次")
    print()  
    return len(records)

# ─────────────────────────── Rendering ───────────────────────
def resolve_image_path(img_path):
    if not img_path:
        return None
    p = Path(img_path)
    if p.exists():
        return p
    p2 = IMAGE_DIR / img_path
    if p2.exists():
        return p2
    p3 = IMAGE_DIR / p.name
    if p3.exists():
        return p3
    return None

def render_sample(record) -> str:
    SIZE = 256
    img = None
    real_path = resolve_image_path(
        record.get("image", "")
    )

    if real_path:
        try:
            img = (
                Image.open(real_path)
                .resize((SIZE, SIZE))
                .convert("RGB")
            )
        except Exception as e:
            print("图片加载失败:", real_path, e)

    if img is None:
        img = Image.new("RGB", (SIZE, SIZE), (30, 40, 52))

    draw = ImageDraw.Draw(img, "RGBA")
    gpt_val = next(
        (c["value"] for c in record.get("conversations", []) if c.get("from") == "gpt"),
        "{}")
    try:
        road = json.loads(gpt_val)
    except Exception:
        road = {}

    def n2p(pt):
        return (int(pt[0] / 1000 * SIZE), int(pt[1] / 1000 * SIZE))

    for ln in road.get("lines", []):
        cat  = ln.get("category", "centerline")
        pts  = [n2p(p) for p in ln.get("points", [])]
        if len(pts) < 2:
            continue
        if cat == "intersection":
            draw.polygon(pts, fill=(255, 109, 0, 55), outline=(255, 109, 0, 200))
        else:
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i+1]], fill=(0, 229, 255, 230), width=2)
            for pt in pts:
                r = 3
                draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r],
                             fill=(255, 255, 0, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────── Batch helpers ────────────────────
def _acquire(bid, user):
    with _lock:
        lk = batch_locks.get(bid)
        if lk is None or lk.get("user") == user:
            batch_locks[bid] = {"user": user, "since": time.time()}
            _save()
            return True
    return False

def _release(bid, user):
    with _lock:
        lk = batch_locks.get(bid)
        if lk and lk.get("user") == user:
            batch_locks[bid] = None
            _save()

def _status_cell(blk, bid):
    lk = blk.get(bid)
    if lk:
        return f'<span style="color:#ef4444;font-weight:600">🔴 {lk["user"]}</span>'
    return '<span style="color:#22c55e;font-weight:600">🟢 空闲</span>'

def _pct(done, total):
    return int(done / total * 100) if total else 0

def batch_table_html():
    with _lock:
        blk = dict(batch_locks)
        ann = {k: len(v) for k, v in annotations.items()}
        bls = list(batches)
    if not bls:
        return '<p style="color:#475569;padding:12px">暂无批次数据，请确认 data/ 目录下有 .jsonl 文件</p>'
    
    rows = "".join(
        f'<tr style="border-bottom:1px solid #1e293b">'
        f'<td style="padding:7px 14px;color:#e2e8f0">{bid}</td>'
        f'<td style="padding:7px 14px;color:#94a3b8">{s}–{e-1}</td>'
        f'<td style="padding:7px 14px;color:#94a3b8">{ann.get(bid,0)}/{e-s}</td>'
        f'<td style="padding:7px 14px">'
        f'  <div style="background:#0f172a;border-radius:4px;height:12px;width:120px;display:inline-block;vertical-align:middle">'
        f'    <div style="background:#6366f1;height:12px;border-radius:4px;width:{_pct(ann.get(bid,0),e-s)}%"></div>'
        f'  </div>'
        f'  <span style="color:#94a3b8;font-size:12px;margin-left:6px">{_pct(ann.get(bid,0),e-s)}%</span>'
        f'</td>'
        f'<td style="padding:7px 14px">{_status_cell(blk, bid)}</td>'
        f'<td style="padding:7px 14px">'
        f'  <button onclick="setBatchAndEnter(\'{bid}\')" '
        f'   style="background:#6366f1;color:#fff;border:none;padding:5px 12px;border-radius:4px;cursor:pointer;font-weight:600;font-size:12px;transition:0.2s">进入 →</button>'
        f'</td>'
        f'</tr>'
        for bid, s, e in bls
    )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<thead><tr style="background:#1e293b">'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">批次</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">范围</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">已标注</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">进度</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">状态</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">操作</th>'
        '</thead><tbody>' + rows + '</tbody></table>'
    )

def _stats_html():
    with _lock:
        ann_all = {k: dict(v) for k, v in annotations.items()}
    total  = len(all_data)
    done   = sum(len(v) for v in ann_all.values())
    pct    = _pct(done, total)
    counts = {c: 0 for c in CATEGORIES}
    for v in ann_all.values():
        for cat in v.values():
            if cat in counts:
                counts[cat] += 1
    bars = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
        f'<span style="width:36px;color:#94a3b8;font-size:12px">{c}</span>'
        f'<div style="flex:1;background:#0f172a;border-radius:4px;height:16px">'
        f'<div style="background:{CAT_COLORS[c]};height:16px;border-radius:4px;'
        f'width:{_pct(counts[c], done)}%"></div></div>'
        f'<span style="color:#e2e8f0;font-size:12px;min-width:36px;text-align:right">{counts[c]}</span>'
        f'</div>'
        for c in CATEGORIES
    )
    cards = "".join(
        f'<div style="background:#1e293b;border-radius:10px;padding:14px 20px;min-width:90px;text-align:center">'
        f'<div style="font-size:26px;font-weight:700;color:{col}">{val}</div>'
        f'<div style="color:#64748b;font-size:11px;margin-top:4px">{lbl}</div></div>'
        for val, col, lbl in [
            (total, "#6366f1", "总样本"),
            (done,  "#22c55e", "已标注"),
            (total - done, "#f59e0b", "待标注"),
            (f"{pct}%", "#e2e8f0", "完成度"),
        ]
    )
    return (
        f'<div style="color:#e2e8f0">'
        f'<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">{cards}</div>'
        f'<div style="background:#1e293b;border-radius:10px;padding:16px;margin-bottom:16px">'
        f'<div style="color:#94a3b8;font-size:13px;font-weight:600;margin-bottom:10px">各类别分布</div>'
        f'{bars}</div>'
        f'<div style="background:#1e293b;border-radius:10px;padding:16px">{batch_table_html()}</div>'
        f'</div>'
    )

LOADING_SPINNER = (
    '<div style="display:flex;align-items:center;justify-content:center;'
    'width:256px;height:256px;background:#1e293b;border-radius:8px;border:2px solid #334155">'
    '<div style="text-align:center;color:#6366f1">'
    '<div style="font-size:28px;animation:spin 1s linear infinite;display:inline-block">⏳</div>'
    '<div style="font-size:12px;color:#64748b;margin-top:8px">加载中…</div>'
    '</div></div>'
    '<style>@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}</style>'
)

def render_at(bid, idx):
    with _lock:
        bls = list(batches)
        ann = dict(annotations.get(bid, {}))
    bi = next((b for b in bls if b[0] == bid), None)
    if not bi:
        return '<div style="color:#ef4444;padding:12px">批次无效</div>', {}, ""
    _, s, e = bi
    recs = all_data[s:e]
    if not recs:
        return '<div style="color:#ef4444;padding:12px">批次为空</div>', {}, ""
    idx = max(0, min(int(idx), len(recs) - 1))
    rec = recs[idx]
    rid = rec.get("id", str(idx))
    b64 = render_sample(rec)
    cat = ann.get(rid, "—")
    if cat != "—":
        badge = (f'<span style="background:{CAT_COLORS.get(cat,"#334155")};color:#fff;'
                 f'padding:2px 9px;border-radius:10px;font-size:11px">{CAT_ICONS.get(cat,"")} {cat}</span>')
    else:
        badge = '<span style="color:#475569;font-size:11px;border:1px solid #334155;padding:2px 8px;border-radius:10px">未标注</span>'

    img_html = (
        f'<div style="display:flex;flex-direction:column;gap:8px">'
        f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
        f'<span style="color:#94a3b8;font-size:13px">样本 <b style="color:#e2e8f0">{idx}</b> / {len(recs)-1}</span>'
        f'{badge}</div>'
        f'<img src="{b64}" style="width:256px;height:256px;border-radius:8px;'
        f'border:2px solid #334155;image-rendering:pixelated"/>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap">'
        f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;color:#00e5ff;font-size:11px">━ 中心线</span>'
        f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;color:#ff6d00;font-size:11px">⬡ 交叉口</span>'
        f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;color:#ffff00;font-size:11px">● 节点</span>'
        f'</div></div>'
    )
    done  = len(ann)
    total = len(recs)
    p     = _pct(done, total)
    prog  = (
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="background:#1e293b;border-radius:6px;height:8px;width:180px">'
        f'<div style="background:#6366f1;height:8px;border-radius:6px;width:{p}%"></div></div>'
        f'<span style="color:#94a3b8;font-size:13px">{done}/{total} ({p}%)</span>'
        f'</div>'
    )
    return img_html, rec.get("meta", {}), prog

# ─────────────────────────── Startup loading in background ────
_startup_done = threading.Event()

def _startup():
    load_all_data_with_progress()
    _startup_done.set()

threading.Thread(target=_startup, daemon=True).start()

# ─────────────────────────── Build Gradio app ─────────────────

# 【修复处】：修改了JS逻辑，确保兼容Gradio的DOM渲染机制
JS_HEAD = """
<script>
function setBatchAndEnter(bid) {
    var container = document.getElementById('hidden_bid');
    if(!container) return;
    
    // 获取真实的输入框
    var el = container.querySelector('textarea') || container.querySelector('input');
    if (el) {
        // 1. 设置值并触发事件
        el.value = bid;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        
        // 2. 延迟等待 Gradio 状态同步后，点击隐藏按钮
        setTimeout(function(){
            var btnWrap = document.getElementById('hidden_btn');
            if(btnWrap) {
                // 有时候 id 挂在外层 div 上，需要往下找真实的 button
                var btn = btnWrap.tagName === 'BUTTON' ? btnWrap : btnWrap.querySelector('button');
                if(btn) btn.click();
            }
        }, 150);
    }
}
</script>
"""

def build():
    # 【修复处】：增加 .hidden-bridge 类，用于在前端隐藏组件但不将其从 DOM 删除
    CSS = """
    .gradio-container{background:#0f172a !important;color:#e2e8f0 !important}
    footer{display:none!important}
    .cat-btn{font-size:14px!important;font-weight:600!important;
             border-radius:8px!important;min-height:46px!important}
    label{color:#94a3b8!important}
    .progress-wrap{background:#1e293b;border-radius:10px;padding:16px;margin-bottom:12px}
    .hidden-bridge{display:none !important;}
    """

    with gr.Blocks(title="道路地图标注平台", css=CSS, head=JS_HEAD) as demo:
        s_user  = gr.State("")
        s_batch = gr.State("")
        s_idx   = gr.State(0)

        # ── Global header ──
        gr.HTML("""
        <div style="text-align:center;padding:20px 0 8px">
          <h1 style="color:#e2e8f0;font-size:26px;font-weight:700;margin:0">
            🗺️ 道路地图标注平台
          </h1>
          <p style="color:#475569;font-size:13px;margin:4px 0 0">
            Road Map BEV Patch Annotation &nbsp;·&nbsp; 多用户并发 &nbsp;·&nbsp; 进度自动保存
          </p>
        </div>""")

        loading_banner = gr.HTML(value=_startup_progress_html(), visible=True)
        main_content   = gr.Column(visible=False)

        # 【修复处】：去除了 visible=False，改用 css 隐藏，保证 JS 桥接能找到 DOM
        hidden_bid = gr.Textbox(elem_id="hidden_bid", elem_classes=["hidden-bridge"])
        hidden_btn = gr.Button(elem_id="hidden_btn", elem_classes=["hidden-bridge"])

        with main_content:
            with gr.Tabs() as tabs:

                # ════ Tab 1: Home ════
                with gr.Tab("🏠 主页 & 批次", id="home"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 👤 登录")
                            inp_user  = gr.Textbox(label="用户名", placeholder="输入用户名…")
                            btn_login = gr.Button("登录平台 →", variant="primary")
                            login_msg = gr.HTML("")

                        with gr.Column(scale=2):
                            gr.Markdown("### 📦 批次状态")
                            btn_ref = gr.Button("🔄 刷新状态", size="sm")
                            tbl     = gr.HTML("")
                            
                # ════ Tab 2: Annotate ════
                with gr.Tab("✏️ 标注工作台", id="annotate"):
                    with gr.Row():
                        lbl_status   = gr.HTML('<span style="color:#475569">未选择批次</span>')
                        lbl_progress = gr.HTML("")

                    with gr.Row():
                        with gr.Column(scale=3, min_width=280):
                            img_disp = gr.HTML(
                                '<div style="width:256px;height:256px;background:#1e293b;'
                                'border:2px solid #334155;border-radius:8px;display:flex;'
                                'align-items:center;justify-content:center;color:#475569;font-size:13px">'
                                '请先在主页选择批次进入</div>')
                            meta_json = gr.JSON(label="样本元数据")

                        with gr.Column(scale=2):
                            gr.Markdown("### 🏷️ 标注分类")
                            gr.HTML('<p style="color:#64748b;font-size:12px;margin:0 0 8px">'
                                   '点击类别后自动前进到下一个样本</p>')
                            cat_btns = {}
                            for cat in CATEGORIES:
                                cat_btns[cat] = gr.Button(
                                    f"{CAT_ICONS[cat]} {cat}",
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
                                btn_jump = gr.Button("跳转 →", scale=1)
                            gr.Markdown("---")
                            gr.Markdown("### 📤 导出 & 管理")
                            btn_export  = gr.Button("导出当前批次结果", variant="secondary")
                            export_out  = gr.Textbox(label="导出信息", lines=3,
                                                     interactive=False)
                            btn_release = gr.Button("🔓 释放批次锁", variant="stop",
                                                    size="sm")
                            release_msg = gr.HTML("")

                # ════ Tab 3: Stats ════
                with gr.Tab("📊 统计总览", id="stats"):
                    btn_ref_s = gr.Button("🔄 刷新统计")
                    stats_out = gr.HTML("")

        timer = gr.Timer(value=1.0, active=True)

        # ─────────────── Callbacks ───────────────────

        def poll_startup():
            html = _startup_progress_html()
            if _startup_done.is_set():
                return (
                    gr.update(value=html, visible=False),
                    gr.update(visible=True),
                    gr.update(active=False),
                    batch_table_html(),
                    _stats_html(),
                )
            return (
                gr.update(value=html, visible=True),
                gr.update(visible=False),
                gr.update(active=True),
                gr.update(),
                gr.update(),
            )

        timer.tick(
            poll_startup,
            outputs=[loading_banner, main_content, timer, tbl, stats_out],
        )

        def cb_login(user):
            if not user.strip():
                return "", "<span style='color:#ef4444'>请输入用户名</span>"
            with _lock:
                prog = user_progress.get(user)
            hint = (f"<br><span style='color:#64748b;font-size:12px'>"
                    f"上次：{prog['batch_id']} 第 {prog['index']} 个</span>"
                    if prog else "")
            return user, f"<span style='color:#22c55e'>✅ 欢迎，{user}！请在右侧点击批次的“进入”</span>{hint}"

        def cb_refresh():
            return batch_table_html()

        def cb_enter(user, bid):
            if not user:
                return (gr.update(), 0,
                        "<span style='color:#ef4444'>⚠️ 请先在主页左侧登录用户名</span>",
                        LOADING_SPINNER, {}, "", gr.update(selected="annotate"))
            ok = _acquire(bid, user)
            if not ok:
                with _lock:
                    lk = batch_locks.get(bid) or {}
                owner = lk.get("user", "其他人")
                return (
                    gr.update(), 0,
                    f"<span style='color:#ef4444'>🔴 {bid} 正被 <b>{owner}</b> 处理中，请选其他批次</span>",
                    '<div style="width:256px;height:256px;background:#1e293b;border-radius:8px;'
                    'display:flex;align-items:center;justify-content:center;color:#ef4444">'
                    '批次已锁定</div>',
                    {}, "", gr.update(selected="annotate"),
                )
            with _lock:
                prog = user_progress.get(user, {})
            idx = prog.get("index", 0) if prog.get("batch_id") == bid else 0
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": idx}
            _save()
            ih, m, pg = render_at(bid, idx)
            status = (f'<span style="color:#6366f1;font-weight:600">👤 {user}</span>'
                      f' &nbsp;|&nbsp; <span style="color:#e2e8f0">📦 {bid}</span>'
                      f' &nbsp;|&nbsp; <span style="color:#22c55e">✅ 已锁定并开始</span>')
            return (
                bid, idx,
                status,
                ih, m, pg,
                gr.update(selected="annotate"),  # 【修复处】使用 update 来安全跳转 Tab
            )

        def cb_annotate(user, bid, idx, cat):
            if not bid:
                return idx, LOADING_SPINNER, {}, ""
            with _lock:
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return idx, '<div style="color:#ef4444">批次无效</div>', {}, ""
            _, s, e = bi
            recs = all_data[s:e]
            if not recs:
                return idx, '<div>空批次</div>', {}, ""
            idx = max(0, min(int(idx), len(recs) - 1))
            rid = recs[idx].get("id", str(idx))
            with _lock:
                annotations.setdefault(bid, {})[rid] = cat
                ni = min(idx + 1, len(recs) - 1)
                user_progress[user] = {"batch_id": bid, "index": ni}
            _save()
            ih, m, pg = render_at(bid, ni)
            return ni, ih, m, pg

        def cb_nav(user, bid, idx, d):
            if not bid:
                return idx, LOADING_SPINNER, {}, ""
            with _lock:
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return idx, '<div style="color:#ef4444">批次无效</div>', {}, ""
            _, s, e = bi
            ni = max(0, min(int(idx) + d, e - s - 1))
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": ni}
            _save()
            ih, m, pg = render_at(bid, ni)
            return ni, ih, m, pg

        def cb_jump(user, bid, j):
            j = int(j or 0)
            ih, m, pg = render_at(bid, j)
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": j}
            _save()
            return j, ih, m, pg

        def cb_export(bid):
            if not bid:
                return "请先选择批次"
            with _lock:
                ann = dict(annotations.get(bid, {}))
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return "批次不存在"
            _, s, e = bi
            recs = all_data[s:e]
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
                        "\n".join(json.dumps(r, ensure_ascii=False) for r in lst))
                    out.append(f"{cat}: {len(lst)} 条 → {p.name}")
            return ("导出完成：\n" + "\n".join(out)) if out else "暂无已标注数据"

        def cb_release(user, bid):
            if not bid:
                return "<span style='color:#ef4444'>未选批次</span>"
            _release(bid, user)
            return f"<span style='color:#22c55e'>✅ 已释放 {bid}</span>"

        # ─── Wire events ───
        btn_login.click(cb_login, [inp_user], [s_user, login_msg])
        btn_ref.click(cb_refresh, outputs=[tbl])
        btn_ref_s.click(_stats_html, outputs=[stats_out])

        hidden_btn.click(
            cb_enter, [s_user, hidden_bid],
            [s_batch, s_idx, lbl_status, img_disp, meta_json, lbl_progress, tabs],
        )

        for cat, btn in cat_btns.items():
            btn.click(
                lambda u, b, i, c=cat: cb_annotate(u, b, i, c),
                [s_user, s_batch, s_idx],
                [s_idx, img_disp, meta_json, lbl_progress],
            )

        btn_prev.click(
            lambda u, b, i: cb_nav(u, b, i, -1),
            [s_user, s_batch, s_idx],
            [s_idx, img_disp, meta_json, lbl_progress],
        )
        btn_next.click(
            lambda u, b, i: cb_nav(u, b, i, +1),
            [s_user, s_batch, s_idx],
            [s_idx, img_disp, meta_json, lbl_progress],
        )
        btn_jump.click(cb_jump, [s_user, s_batch, inp_jump],
                       [s_idx, img_disp, meta_json, lbl_progress])
        btn_export.click(cb_export, [s_batch], [export_out])
        btn_release.click(cb_release, [s_user, s_batch], [release_msg])

    return demo, CSS


def _startup_progress_html():
    pct = _load_progress
    msg = _load_status_msg
    done = _startup_done.is_set()
    color = "#22c55e" if done else "#6366f1"
    spinner = "" if done else (
        '<span style="display:inline-block;animation:spin 1s linear infinite;'
        'margin-right:8px">⏳</span>'
        '<style>@keyframes spin{to{transform:rotate(360deg)}}</style>'
    )
    return (
        f'<div style="max-width:520px;margin:60px auto;background:#1e293b;'
        f'border-radius:16px;padding:32px;text-align:center;border:1px solid #334155">'
        f'<div style="font-size:32px;margin-bottom:16px">🗺️</div>'
        f'<h2 style="color:#e2e8f0;font-size:18px;margin:0 0 20px;font-weight:600">'
        f'道路地图标注平台</h2>'
        f'<div style="background:#0f172a;border-radius:8px;height:16px;overflow:hidden;margin-bottom:12px">'
        f'<div style="background:{color};height:16px;border-radius:8px;'
        f'width:{pct}%;transition:width 0.4s ease"></div>'
        f'</div>'
        f'<div style="color:#94a3b8;font-size:13px">{spinner}{msg}</div>'
        f'<div style="color:#475569;font-size:11px;margin-top:8px">{pct}%</div>'
        f'</div>'
    )


if __name__ == "__main__":
    app, css = build()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        css=css,
    )
