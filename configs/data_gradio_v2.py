"""
Road Map Annotation Platform v3.2
- ??: launch(css=...) ?? Gradio 5.x
- ??: ????????? jsonl
- ??: ??????????? images/images ???
- ??: GPT response ?? array/dict ????
- ??: category ??????
- ??: ?????????????????/??
"""

import gradio as gr
import json, os, threading, time, base64, io, math, sys
from pathlib import Path
from PIL import Image, ImageDraw

# ============================================================
# ???? - ??????
# ============================================================
# ???????????scripts/ ?????
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE   = _PROJECT_ROOT / "data/debug_phase_a_lane_intersection20/train.jsonl"  # jsonl ??
IMAGE_DIR   = _PROJECT_ROOT / "data/av2_patch_256_fullimage_cutflag_test_v2"       # ?????
OUTPUT_DIR  = Path("output")
STATE_DIR   = Path("state")
BATCH_SIZE  = 20                               # demo ??????? 10000
PORT        = 7863

CATEGORIES  = ["??", "??", "??", "??", "??"]
CAT_COLORS  = {"??":"#22c55e","??":"#f59e0b","??":"#ef4444",
               "??":"#94a3b8","??":"#6b7280"}
CAT_ICONS   = {"??":"?","??":"??","??":"???","??":"?","??":"?"}

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

# ============================================================
# ????
# ============================================================
def debug(msg: str):
    print(f"[DEBUG] {msg}", flush=True)

def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)

# ============================================================
# Global state
# ============================================================
_lock         = threading.RLock()
batch_locks   : dict = {}
user_progress : dict = {}
annotations   : dict = {}
all_data      : list = []
batches       : list = []
_load_progress   : int = 0
_load_status_msg : str = "????..."

# ============================================================
# Persistence
# ============================================================
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
        # ?????????????????
        batch_locks.clear()
        batch_locks.update({k: None for k in st.get("batch_locks", {})})
        user_progress.update(st.get("user_progress", {}))
        for k, v in st.get("annotations", {}).items():
            annotations.setdefault(k, {}).update(v)

# ============================================================
# Data loading (with progress + debug)
# ============================================================
def _set_progress(pct: int, msg: str):
    global _load_progress, _load_status_msg
    _load_progress   = pct
    _load_status_msg = msg

def load_all_data_with_progress():
    global all_data, batches

    data_file = DATA_FILE
    debug(f"????: {data_file.resolve()}")
    debug(f"?????: {IMAGE_DIR.resolve()}")

    records = []
    if data_file.is_file() and data_file.suffix == ".jsonl":
        files = [data_file]
        debug(f"?????: {data_file}")
    elif data_file.is_dir():
        files = sorted(data_file.rglob("*.jsonl"))
        debug(f"??????? {len(files)} ? jsonl ??")
        for f in files:
            debug(f"  ??: {f}")
    else:
        warn(f"???????? jsonl ??: {data_file}")
        _set_progress(100, f"??? .jsonl ????: {data_file}")
        return 0

    if not files:
        warn("data ???? .jsonl ??")
        _set_progress(100, "data ???? .jsonl ??")
        return 0

    total_files = len(files)
    for fi, f in enumerate(files):
        _set_progress(int(fi / total_files * 60),
                      f"?? {f.name} ({fi+1}/{total_files})...")
        lines = f.read_text(encoding="utf-8").splitlines()
        debug(f"?? {f.name}: {len(lines)} ?")
        for li, line in enumerate(lines):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    warn(f"???? {f.name}:{li}: {e}")
            if len(lines) > 5000 and li % 5000 == 0:
                _set_progress(int((fi + li / len(lines)) / total_files * 60),
                              f"?? {f.name}: {li}/{len(lines)} ?...")

    if not records:
        warn("???????")
        _set_progress(100, "???????")
        return 0

    debug(f"??? {len(records)} ???")
    all_data = records

    bsz = BATCH_SIZE
    batches.clear()
    n_batches = math.ceil(len(records) / bsz)
    for i in range(n_batches):
        bid   = f"batch_{i+1:04d}"
        start = i * bsz
        end   = min(start + bsz, len(records))
        batches.append((bid, start, end))
        batch_locks.setdefault(bid, None)
        annotations.setdefault(bid, {})
        debug(f"  ?? {bid}: [{start}, {end}) ? {end-start} ?")
        _set_progress(65 + int(i / n_batches * 15), f"???? {bid}...")

    debug("??????...")
    _load_state()
    _set_progress(100, f"???{len(records)} ? / {len(batches)} ??")
    return len(records)

# ============================================================
# ???????????? + ?????
# ============================================================
def resolve_image_path(img_path):
    if not img_path:
        warn("record ?? image ??")
        return None
    # 1. ????
    p = Path(img_path)
    if p.exists():
        debug(f"??: ?????? {p}")
        return p
    # 2. IMAGE_DIR + ?? img_path
    candidate = IMAGE_DIR / img_path
    if candidate.exists():
        debug(f"??: IMAGE_DIR/{img_path} ??")
        return candidate
    # 3. ?? images/ ???? images/images ???
    if img_path.startswith("images/"):
        stripped = img_path[len("images/"):]
        candidate = IMAGE_DIR / stripped
        if candidate.exists():
            debug(f"??: ???? {candidate} ??")
            return candidate
    # 4. ????
    candidate = IMAGE_DIR / p.name
    if candidate.exists():
        debug(f"??: ???? {candidate} ??")
        return candidate
    # 5. ???
    warn(f"?????: {img_path} (??? IMAGE_DIR={IMAGE_DIR})")
    return None

# ============================================================
# Rendering
# ============================================================
def render_sample(record) -> str:
    SIZE = 256
    img = None
    real_path = resolve_image_path(record.get("image", ""))
    if real_path:
        try:
            img = Image.open(real_path).resize((SIZE, SIZE)).convert("RGB")
        except Exception as e:
            warn(f"?????? {real_path}: {e}")
    if img is None:
        debug(f"??????? (?????)")
        img = Image.new("RGB", (SIZE, SIZE), (30, 40, 52))

    draw = ImageDraw.Draw(img, "RGBA")
    gpt_val = next(
        (c["value"] for c in record.get("conversations", []) if c.get("from") == "gpt"),
        "{}")

    try:
        parsed = json.loads(gpt_val)
    except Exception as e:
        warn(f"GPT response ????: {e}")
        parsed = {}

    # ??????: array ? {"lines": [...]}
    if isinstance(parsed, list):
        lines_data = parsed
        debug(f"GPT response ??: array, {len(lines_data)} ??")
    elif isinstance(parsed, dict):
        lines_data = parsed.get("lines", [])
        debug(f"GPT response ??: dict+lines, {len(lines_data)} ??")
    else:
        lines_data = []
        warn(f"GPT response ????: {type(parsed).__name__}")

    # ????????
    sample_pts = []
    for ln in lines_data:
        for p in ln.get("points", []):
            sample_pts.append(p)
    if sample_pts:
        max_val = max(max(p[0], p[1]) for p in sample_pts)
        coord_range = 255 if max_val <= 256 else 1000
        debug(f"????: max={max_val}, range={coord_range}")
    else:
        coord_range = 1000

    def n2p(pt):
        return (int(pt[0] / coord_range * SIZE),
                int(pt[1] / coord_range * SIZE))

    lines_drawn = 0
    for ln in lines_data:
        cat = str(ln.get("category", "centerline")).lower()
        pts = [n2p(p) for p in ln.get("points", [])]
        if len(pts) < 2:
            debug(f"  ??: ???2??")
            continue
        if cat == "intersection":
            draw.polygon(pts, fill=(255, 109, 0, 55), outline=(255, 109, 0, 200))
            lines_drawn += 1
        else:
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i+1]], fill=(0, 229, 255, 230), width=2)
            for pt in pts:
                r = 3
                draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r],
                             fill=(255, 255, 0, 255))
            lines_drawn += 1
    debug(f"??: ??? {lines_drawn} ??")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ============================================================
# Batch helpers
# ============================================================
def _acquire(bid, user):
    with _lock:
        lk = batch_locks.get(bid)
        if lk is None:
            batch_locks[bid] = {"user": user, "since": time.time()}
            debug(f"???: {bid} ? {user} ?? (????)")
            _save()
            return True
        if lk.get("user") == user:
            debug(f"???: {bid} ?? {user} ?????")
            return True
        debug(f"???: {bid} ? {user} ????? {lk['user']} ??")
    return False

def _release(bid, user):
    with _lock:
        lk = batch_locks.get(bid)
        if lk and lk.get("user") == user:
            batch_locks[bid] = None
            debug(f"???: {bid} ? {user} ??")
            _save()

def _status_cell(blk, bid):
    lk = blk.get(bid)
    if lk:
        return f'<span style="color:#ef4444;font-weight:600">? {lk["user"]}</span>'
    return '<span style="color:#22c55e;font-weight:600">? ??</span>'

def _pct(done, total):
    return int(done / total * 100) if total else 0

def batch_table_html():
    with _lock:
        blk = dict(batch_locks)
        ann = {k: len(v) for k, v in annotations.items()}
        bls = list(batches)
    if not bls:
        return '<p style="color:#64748b;padding:12px">??????</p>'
    rows = "".join(
        f'<tr style="border-bottom:1px solid #e2e8f0">'
        f'<td style="padding:7px 14px;color:#1e293b">{bid}</td>'
        f'<td style="padding:7px 14px;color:#64748b">{s}?{e-1}</td>'
        f'<td style="padding:7px 14px;color:#64748b">{ann.get(bid,0)}/{e-s}</td>'
        f'<td style="padding:7px 14px">'
        f'<div style="background:#f8fafc;border-radius:4px;height:12px;width:120px;display:inline-block;vertical-align:middle">'
        f'<div style="background:#6366f1;height:12px;border-radius:4px;width:{_pct(ann.get(bid,0),e-s)}%"></div></div>'
        f'<span style="color:#64748b;font-size:12px;margin-left:6px">{_pct(ann.get(bid,0),e-s)}%</span>'
        f'</td>'
        f'<td style="padding:7px 14px">{_status_cell(blk, bid)}</td>'
        f'<td style="padding:7px 14px">'
        f'  <button onclick="selectBatch(\'{bid}\')" '
        f'   style="background:#6366f1;color:#fff;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px">?? ?</button>'
        f'</td>'
        f'</tr>'
        for bid, s, e in bls
    )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<thead><tr style="background:#f1f5f9">'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">??</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">??</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">???</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">??</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">??</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">??</th>'
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
        f'<span style="width:36px;color:#64748b;font-size:12px">{c}</span>'
        f'<div style="flex:1;background:#f8fafc;border-radius:4px;height:16px">'
        f'<div style="background:{CAT_COLORS[c]};height:16px;border-radius:4px;'
        f'width:{_pct(counts[c], done)}%"></div></div>'
        f'<span style="color:#1e293b;font-size:12px;min-width:36px;text-align:right">{counts[c]}</span>'
        f'</div>'
        for c in CATEGORIES
    )
    cards = "".join(
        f'<div style="background:#f1f5f9;border-radius:10px;padding:14px 20px;min-width:90px;text-align:center">'
        f'<div style="font-size:26px;font-weight:700;color:{col}">{val}</div>'
        f'<div style="color:#64748b;font-size:11px;margin-top:4px">{lbl}</div></div>'
        for val, col, lbl in [
            (total, "#6366f1", "???"),
            (done,  "#22c55e", "???"),
            (total - done, "#f59e0b", "???"),
            (f"{pct}%", "#1e293b", "???"),
        ]
    )
    return (
        f'<div style="color:#1e293b">'
        f'<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">{cards}</div>'
        f'<div style="background:#f1f5f9;border-radius:10px;padding:16px;margin-bottom:16px">'
        f'<div style="color:#64748b;font-size:13px;font-weight:600;margin-bottom:10px">?????</div>'
        f'{bars}</div>'
        f'<div style="background:#f1f5f9;border-radius:10px;padding:16px">{batch_table_html()}</div>'
        f'</div>'
    )

LOADING_SPINNER = (
    '<div style="display:flex;align-items:center;justify-content:center;'
    'width:256px;height:256px;background:#f1f5f9;border-radius:8px;border:2px solid #e2e8f0">'
    '<div style="text-align:center;color:#6366f1">'
    '<div style="font-size:28px;animation:spin 1s linear infinite;display:inline-block">?</div>'
    '<div style="font-size:12px;color:#64748b;margin-top:8px">???...</div>'
    '</div></div>'
    '<style>@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}</style>'
)

def render_at(bid, idx):
    with _lock:
        bls = list(batches)
        ann = dict(annotations.get(bid, {}))
    bi = next((b for b in bls if b[0] == bid), None)
    if not bi:
        return '<div style="color:#ef4444;padding:12px">????</div>', {}, ""
    _, s, e = bi
    recs = all_data[s:e]
    if not recs:
        return '<div style="color:#ef4444;padding:12px">????</div>', {}, ""
    idx = max(0, min(int(idx), len(recs) - 1))
    rec = recs[idx]
    rid = rec.get("id", str(idx))
    debug(f"render_at: {bid}, idx={idx}, id={rid}")
    b64 = render_sample(rec)
    cat = ann.get(rid, "?")
    if cat != "?":
        badge = (f'<span style="background:{CAT_COLORS.get(cat,"#64748b")};color:#fff;'
                 f'padding:2px 9px;border-radius:10px;font-size:11px">{CAT_ICONS.get(cat,"")} {cat}</span>')
    else:
        badge = '<span style="color:#64748b;font-size:11px;border:1px solid #e2e8f0;padding:2px 8px;border-radius:10px">???</span>'

    img_html = (
        f'<div style="display:flex;flex-direction:column;gap:8px">'
        f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
        f'<span style="color:#64748b;font-size:13px">?? <b style="color:#1e293b">{idx}</b> / {len(recs)-1}</span>'
        f'{badge}</div>'
        f'<img src="{b64}" style="width:256px;height:256px;border-radius:8px;'
        f'border:2px solid #e2e8f0;image-rendering:pixelated"/>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap">'
        f'<span style="background:#f1f5f9;padding:3px 8px;border-radius:6px;color:#00e5ff;font-size:11px">? ???</span>'
        f'<span style="background:#f1f5f9;padding:3px 8px;border-radius:6px;color:#ff6d00;font-size:11px">? ???</span>'
        f'<span style="background:#f1f5f9;padding:3px 8px;border-radius:6px;color:#ffff00;font-size:11px">? ??</span>'
        f'</div></div>'
    )
    done  = len(ann)
    total = len(recs)
    p     = _pct(done, total)
    prog  = (
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="background:#f1f5f9;border-radius:6px;height:8px;width:180px">'
        f'<div style="background:#6366f1;height:8px;border-radius:6px;width:{p}%"></div></div>'
        f'<span style="color:#64748b;font-size:13px">{done}/{total} ({p}%)</span>'
        f'</div>'
    )
    return img_html, rec.get("meta", {}), prog

# JS bridge: ??????"??"?????? dropdown ???"????"
JS_HEAD = """
<script>
function selectBatch(bid) {
    var sel = document.querySelector('select[data-testid="dropdown"]');
    if (!sel) return;
    // ??? dropdown ???????
    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype, 'value'
    ).set;
    nativeInputValueSetter.call(sel, bid);
    sel.dispatchEvent(new Event('change', { bubbles: true }));
}
</script>
"""

# ============================================================
# Startup
# ============================================================
_startup_done = threading.Event()

def _startup():
    debug("=== ?????? ===")
    load_all_data_with_progress()
    debug("=== ?????? ===")
    _startup_done.set()

threading.Thread(target=_startup, daemon=True).start()

# ============================================================
# CSS
# ============================================================
CSS = """
.gradio-container{background:#ffffff !important;color:#1e293b !important}
footer{display:none!important}
.cat-btn{font-size:14px!important;font-weight:600!important;
         border-radius:8px!important;min-height:46px!important}
label{color:#64748b!important}
"""

def _startup_progress_html():
    pct = _load_progress
    msg = _load_status_msg
    done = _startup_done.is_set()
    color = "#22c55e" if done else "#6366f1"
    spinner = "" if done else (
        '<span style="display:inline-block;animation:spin 1s linear infinite;'
        'margin-right:8px">?</span>'
        '<style>@keyframes spin{to{transform:rotate(360deg)}}</style>'
    )
    return (
        f'<div style="max-width:520px;margin:60px auto;background:#f1f5f9;'
        f'border-radius:16px;padding:32px;text-align:center;border:1px solid #e2e8f0">'
        f'<div style="font-size:32px;margin-bottom:16px">??</div>'
        f'<h2 style="color:#1e293b;font-size:18px;margin:0 0 20px;font-weight:600">'
        f'????????</h2>'
        f'<div style="background:#f8fafc;border-radius:8px;height:16px;overflow:hidden;margin-bottom:12px">'
        f'<div style="background:{color};height:16px;border-radius:8px;'
        f'width:{pct}%;transition:width 0.4s ease"></div>'
        f'</div>'
        f'<div style="color:#64748b;font-size:13px">{spinner}{msg}</div>'
        f'<div style="color:#64748b;font-size:11px;margin-top:8px">{pct}%</div>'
        f'</div>'
    )

# ============================================================
# Gradio app
# ============================================================
def build():
    with gr.Blocks(title="????????", css=CSS, head=JS_HEAD) as demo:
        s_user  = gr.State("")
        s_batch = gr.State("")
        s_idx   = gr.State(0)

        gr.HTML("""
        <div style="text-align:center;padding:20px 0 8px">
          <h1 style="color:#1e293b;font-size:26px;font-weight:700;margin:0">
            ?? ????????
          </h1>
          <p style="color:#64748b;font-size:13px;margin:4px 0 0">
            Road Map BEV Patch Annotation &nbsp;?nbsp; ????? &nbsp;?nbsp; ??????
          </p>
        </div>""")

        loading_banner = gr.HTML(value=_startup_progress_html(), visible=True)
        main_content   = gr.Column(visible=False)

        with main_content:
            with gr.Tabs() as tabs:
                with gr.Tab("? ?? & ??", id="home"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### ? ??")
                            inp_user  = gr.Textbox(label="???", placeholder="?????...")
                            btn_login = gr.Button("???? ?", variant="primary")
                            login_msg = gr.HTML("")

                        with gr.Column(scale=2):
                            gr.Markdown("### ? ????")
                            btn_ref = gr.Button("? ????", size="sm")
                            tbl     = gr.HTML("")

                    gr.Markdown("---")
                    gr.Markdown("### ? ??????")
                    with gr.Row():
                        dd_batch  = gr.Dropdown(label="????", choices=[], value=None)
                        btn_enter = gr.Button("???? ?", variant="primary", scale=0)
                    enter_msg = gr.HTML("")

                with gr.Tab("?? ?????", id="annotate"):
                    with gr.Row():
                        lbl_status   = gr.HTML('<span style="color:#64748b">?????</span>')
                        lbl_progress = gr.HTML("")

                    with gr.Row():
                        with gr.Column(scale=3, min_width=280):
                            img_disp = gr.HTML(
                                '<div style="width:256px;height:256px;background:#f1f5f9;'
                                'border:2px solid #e2e8f0;border-radius:8px;display:flex;'
                                'align-items:center;justify-content:center;color:#64748b;font-size:13px">'
                                '?????????</div>')
                            meta_json = gr.JSON(label="?????")

                        with gr.Column(scale=2):
                            gr.Markdown("### ?? ????")
                            gr.HTML('<p style="color:#64748b;font-size:12px;margin:0 0 8px">'
                                   '???????????????</p>')
                            cat_btns = {}
                            for cat in CATEGORIES:
                                cat_btns[cat] = gr.Button(
                                    f"{CAT_ICONS[cat]} {cat}",
                                    elem_classes=["cat-btn"], variant="secondary")
                            gr.Markdown("---")
                            gr.Markdown("### ? ??")
                            with gr.Row():
                                btn_prev = gr.Button("? ???")
                                btn_next = gr.Button("??? ?")
                            with gr.Row():
                                inp_jump = gr.Number(label="?????", value=0,
                                                     precision=0, scale=2)
                                btn_jump = gr.Button("?? ?", scale=1)
                            gr.Markdown("---")
                            gr.Markdown("### ? ?? & ??")
                            btn_export  = gr.Button("????????", variant="secondary")
                            export_out  = gr.Textbox(label="????", lines=3,
                                                     interactive=False)
                            btn_release = gr.Button("? ?????", variant="stop",
                                                    size="sm")
                            release_msg = gr.HTML("")

                with gr.Tab("? ????", id="stats"):
                    btn_ref_s = gr.Button("? ????")
                    stats_out = gr.HTML("")

        timer = gr.Timer(value=1.0, active=True)

        def poll_startup():
            html = _startup_progress_html()
            if _startup_done.is_set():
                choices = [b[0] for b in batches]
                val     = choices[0] if choices else None
                debug(f"??: {len(choices)} ?????")
                return (
                    gr.update(value=html, visible=False),
                    gr.update(visible=True),
                    gr.update(active=False),
                    gr.update(choices=choices, value=val),
                    batch_table_html(),
                    _stats_html(),
                )
            return (
                gr.update(value=html, visible=True),
                gr.update(visible=False),
                gr.update(active=True),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        timer.tick(
            poll_startup,
            outputs=[loading_banner, main_content, timer, dd_batch, tbl, stats_out],
        )

        def cb_login(user):
            if not user.strip():
                debug("??: ?????")
                return "", "<span style='color:#ef4444'>??????</span>"
            with _lock:
                prog = user_progress.get(user)
            hint = (f"<br><span style='color:#64748b;font-size:12px'>"
                    f"???{prog['batch_id']} ? {prog['index']} ?</span>"
                    if prog else "")
            debug(f"??: {user} (prog={prog})")
            return user, f"<span style='color:#22c55e'>? ???{user}?</span>{hint}"

        def cb_refresh():
            return batch_table_html()

        def cb_enter(user, bid):
            if not user:
                debug(f"????: ?????")
                return (gr.update(), 0,
                        "<span style='color:#ef4444'>????</span>",
                        LOADING_SPINNER, {}, "", gr.update())
            if not bid:
                debug(f"????: ?????")
                return (gr.update(), 0,
                        "<span style='color:#ef4444'>??????</span>",
                        LOADING_SPINNER, {}, "", gr.update())
            debug(f"????: user={user}, bid={bid}")
            ok = _acquire(bid, user)
            if not ok:
                with _lock:
                    lk = batch_locks.get(bid) or {}
                owner = lk.get("user", "???")
                debug(f"????: {bid} ?? {owner} ??")
                return (
                    gr.update(), 0,
                    f"<span style='color:#ef4444'>? {bid} ??? <b>{owner}</b> ???</span>",
                    '<div style="width:256px;height:256px;background:#f1f5f9;border-radius:8px;'
                    'display:flex;align-items:center;justify-content:center;color:#ef4444">'
                    '?????</div>',
                    {}, "", gr.update(),
                )
            with _lock:
                prog = user_progress.get(user, {})
            idx = prog.get("index", 0) if prog.get("batch_id") == bid else 0
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": idx}
            _save()
            debug(f"render batch: {bid}, start_idx={idx}")
            ih, m, pg = render_at(bid, idx)
            return (
                bid, idx,
                f"<span style='color:#22c55e'>? ??? {bid}??????</span>",
                ih, m, pg,
                gr.Tabs(selected="annotate"),
            )

        def cb_annotate(user, bid, idx, cat):
            if not bid:
                return idx, LOADING_SPINNER, {}, ""
            debug(f"??: user={user}, bid={bid}, idx={idx}, cat={cat}")
            with _lock:
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return idx, '<div style="color:#ef4444">????</div>', {}, ""
            _, s, e = bi
            recs = all_data[s:e]
            if not recs:
                return idx, '<div>???</div>', {}, ""
            idx = max(0, min(int(idx), len(recs) - 1))
            rid = recs[idx].get("id", str(idx))
            with _lock:
                annotations.setdefault(bid, {})[rid] = cat
                ni = min(idx + 1, len(recs) - 1)
                user_progress[user] = {"batch_id": bid, "index": ni}
            _save()
            debug(f"  ??? {rid}={cat}, ??? {ni}")
            ih, m, pg = render_at(bid, ni)
            return ni, ih, m, pg

        def cb_nav(user, bid, idx, d):
            if not bid:
                return idx, LOADING_SPINNER, {}, ""
            with _lock:
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return idx, '<div style="color:#ef4444">????</div>', {}, ""
            _, s, e = bi
            ni = max(0, min(int(idx) + d, e - s - 1))
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": ni}
            _save()
            debug(f"??: {bid} idx {idx} -> {ni}")
            ih, m, pg = render_at(bid, ni)
            return ni, ih, m, pg

        def cb_jump(user, bid, j):
            j = int(j or 0)
            debug(f"??: {bid} -> {j}")
            ih, m, pg = render_at(bid, j)
            with _lock:
                user_progress[user] = {"batch_id": bid, "index": j}
            _save()
            return j, ih, m, pg

        def cb_export(bid):
            if not bid:
                return "??????"
            with _lock:
                ann = dict(annotations.get(bid, {}))
                bls = list(batches)
            bi = next((b for b in bls if b[0] == bid), None)
            if not bi:
                return "?????"
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
                    out.append(f"{cat}: {len(lst)} ? ? {p.name}")
            debug(f"?? {bid}: {out}")
            return ("?????\n" + "\n".join(out)) if out else "???????"

        def cb_release(user, bid):
            if not bid:
                return "<span style='color:#ef4444'>????</span>"
            debug(f"??: user={user}, bid={bid}")
            _release(bid, user)
            return f"<span style='color:#22c55e'>? ??? {bid}</span>"

        btn_login.click(cb_login, [inp_user], [s_user, login_msg])
        btn_ref.click(cb_refresh, outputs=[tbl])
        btn_ref_s.click(_stats_html, outputs=[stats_out])

        btn_enter.click(
            cb_enter, [s_user, dd_batch],
            [s_batch, s_idx, enter_msg, img_disp, meta_json, lbl_progress, tabs],
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

    return demo


if __name__ == "__main__":
    app = build()
    app.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=True,
        show_error=True,
    )
