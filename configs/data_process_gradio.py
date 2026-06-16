"""
Road Map Annotation Platform v2 – Gradio 6 compatible
Multi-user concurrent annotation tool with batch management
"""

import gradio as gr
import json
import os
import threading
import time
import base64
import io
import math
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

# ─────────────────────────── Config ───────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
STATE_DIR  = Path("state")
DEMO_BATCH_SIZE = 10   # small batches for demo; set to 10000 for production

CATEGORIES = ["简单", "中等", "困难", "空白", "丢弃"]
CAT_COLORS = {"简单":"#22c55e","中等":"#f59e0b","困难":"#ef4444","空白":"#94a3b8","丢弃":"#6b7280"}
CAT_ICONS  = {"简单":"✅","中等":"⚡","困难":"🔥","空白":"⬜","丢弃":"🗑️"}

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

# ─────────────────────────── Global mutable state ─────────────
_lock        = threading.Lock()
batch_locks  : dict = {}   # bid -> {"user":..,"since":..} | None
user_progress: dict = {}   # user -> {"batch_id":.., "index":..}
annotations  : dict = {}   # bid -> {sample_id: category}
all_data     : list = []
batches      : list = []   # [(bid, start, end), ...]

# ─────────────────────────── Persistence ──────────────────────
def _save():
    with _lock:
        st = {"batch_locks": {k:v for k,v in batch_locks.items()},
              "user_progress": dict(user_progress),
              "annotations":   {k:dict(v) for k,v in annotations.items()}}
    (STATE_DIR/"state.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))

def _load():
    p = STATE_DIR/"state.json"
    if not p.exists(): return
    st = json.loads(p.read_text())
    with _lock:
        batch_locks.update(st.get("batch_locks",{}))
        user_progress.update(st.get("user_progress",{}))
        for k,v in st.get("annotations",{}).items():
            annotations.setdefault(k,{}).update(v)

# ─────────────────────────── Data loading ─────────────────────
def load_all_data():
    global all_data, batches
    records = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                try: records.append(json.loads(line))
                except: pass
    all_data = records
    bsz = DEMO_BATCH_SIZE
    batches.clear()
    for i in range(math.ceil(len(records)/bsz) if records else 0):
        bid   = f"batch_{i+1:04d}"
        start = i*bsz
        end   = min(start+bsz, len(records))
        batches.append((bid, start, end))
        batch_locks.setdefault(bid, None)
        annotations.setdefault(bid, {})
    return len(records)

# ─────────────────────────── Rendering ───────────────────────
def render_sample(record) -> str:
    SIZE = 256
    img_path = record.get("image","")
    img = None
    if img_path and os.path.exists(img_path):
        try: img = Image.open(img_path).resize((SIZE,SIZE)).convert("RGB")
        except: pass
    if img is None:
        img = Image.new("RGB",(SIZE,SIZE),(30,40,52))

    draw = ImageDraw.Draw(img,"RGBA")
    gpt_val = next((c["value"] for c in record.get("conversations",[]) if c.get("from")=="gpt"), "{}")
    try: road = json.loads(gpt_val)
    except: road = {}

    def n2p(pt): return (int(pt[0]/1000*SIZE), int(pt[1]/1000*SIZE))

    for ln in road.get("lines",[]):
        cat  = ln.get("category","centerline")
        pts  = [n2p(p) for p in ln.get("points",[])]
        if len(pts)<2: continue
        if cat=="intersection":
            draw.polygon(pts, fill=(255,109,0,55), outline=(255,109,0,200))
        else:
            for i in range(len(pts)-1):
                draw.line([pts[i],pts[i+1]], fill=(0,229,255,230), width=2)
            for pt in pts:
                r=3; draw.ellipse([pt[0]-r,pt[1]-r,pt[0]+r,pt[1]+r], fill=(255,255,0,255))

    buf = io.BytesIO(); img.save(buf,format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────── Batch helpers ────────────────────
def acquire(bid, user):
    with _lock:
        lk = batch_locks.get(bid)
        if lk is None or lk.get("user")==user:
            batch_locks[bid]={"user":user,"since":time.time()}; _save(); return True
    return False

def release(bid, user):
    with _lock:
        lk = batch_locks.get(bid)
        if lk and lk.get("user")==user:
            batch_locks[bid]=None; _save()

def batch_table_html():
    with _lock:
        blk  = dict(batch_locks)
        ann  = {k:len(v) for k,v in annotations.items()}
        bls  = list(batches)
    rows="".join(
        f'<tr style="border-bottom:1px solid #1e293b">'
        f'<td style="padding:7px 14px;color:#e2e8f0">{bid}</td>'
        f'<td style="padding:7px 14px;color:#94a3b8">{s}–{e-1}</td>'
        f'<td style="padding:7px 14px;color:#94a3b8">{ann.get(bid,0)}/{e-s}</td>'
        f'<td style="padding:7px 14px"><div style="background:#0f172a;border-radius:4px;height:12px;width:120px;display:inline-block">'
        f'<div style="background:#6366f1;height:12px;border-radius:4px;width:{int(ann.get(bid,0)/(e-s)*100) if e>s else 0}%"></div></div>'
        f' <span style="color:#94a3b8;font-size:12px">{int(ann.get(bid,0)/(e-s)*100) if e>s else 0}%</span></td>'
        f'<td style="padding:7px 14px">'
        f'{"<span style=\\'color:#ef4444;font-weight:600\\'>🔴 "+blk[bid]["user"]+"</span>" if blk.get(bid) else "<span style=\\'color:#22c55e;font-weight:600\\'>🟢 空闲</span>"}'
        f'</td></tr>'
        for bid,s,e in bls
    )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<thead><tr style="background:#1e293b">'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">批次</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">范围</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">已标注</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">进度</th>'
        '<th style="padding:8px 14px;text-align:left;color:#64748b">状态</th>'
        '</thead><tbody>'+rows+'</tbody></table>'
    )

def stats_html():
    with _lock:
        ann_all = {k:dict(v) for k,v in annotations.items()}
    total  = len(all_data)
    done   = sum(len(v) for v in ann_all.values())
    pct    = int(done/total*100) if total else 0
    counts = {c:0 for c in CATEGORIES}
    for v in ann_all.values():
        for cat in v.values():
            if cat in counts: counts[cat]+=1
    bars = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
        f'<span style="width:36px;color:#94a3b8;font-size:12px">{c}</span>'
        f'<div style="flex:1;background:#0f172a;border-radius:4px;height:16px">'
        f'<div style="background:{CAT_COLORS[c]};height:16px;border-radius:4px;width:{int(counts[c]/max(done,1)*100)}%"></div></div>'
        f'<span style="color:#e2e8f0;font-size:12px;min-width:36px;text-align:right">{counts[c]}</span></div>'
        for c in CATEGORIES
    )
    return f"""
    <div style="color:#e2e8f0">
      <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap">
        {''.join(f'<div style="background:#1e293b;border-radius:10px;padding:14px 20px;min-width:100px;text-align:center">'
                 f'<div style="font-size:28px;font-weight:700;color:{c2}">{v2}</div>'
                 f'<div style="color:#64748b;font-size:12px;margin-top:4px">{l}</div></div>'
                 for v2,c2,l in [(total,"#6366f1","总样本"),(done,"#22c55e","已标注"),(total-done,"#f59e0b","待标注"),(f"{pct}%","#e2e8f0","进度")])}
      </div>
      <div style="background:#1e293b;border-radius:10px;padding:16px;margin-bottom:16px">
        <div style="color:#94a3b8;font-size:13px;margin-bottom:10px;font-weight:600">各类别分布</div>
        {bars}
      </div>
      <div style="background:#1e293b;border-radius:10px;padding:16px">{batch_table_html()}</div>
    </div>"""

def render_at(bid, idx):
    with _lock:
        bls = list(batches); ann = dict(annotations.get(bid,{}))
    bi = next((b for b in bls if b[0]==bid),None)
    if not bi: return '<div style="color:#ef4444">批次无效</div>',{},""
    _,s,e = bi; recs = all_data[s:e]
    if not recs: return '<div style="color:#ef4444">批次为空</div>',{},""
    idx = max(0,min(int(idx),len(recs)-1))
    rec = recs[idx]; rid = rec.get("id",str(idx))
    b64 = render_sample(rec); cat = ann.get(rid,"—")
    badge = (f'<span style="background:{CAT_COLORS.get(cat,"#334155")};color:#fff;padding:2px 9px;border-radius:10px;font-size:11px">{cat}</span>'
             if cat!="—" else '<span style="color:#475569;font-size:11px">未标注</span>')
    img_html = (
        f'<div style="display:flex;flex-direction:column;gap:8px">'
        f'<div style="display:flex;gap:8px;align-items:center">'
        f'<span style="color:#94a3b8;font-size:13px">样本 {idx} / {len(recs)-1}</span>{badge}</div>'
        f'<img src="{b64}" style="width:256px;height:256px;border-radius:8px;border:2px solid #334155;image-rendering:pixelated"/>'
        f'<div style="display:flex;gap:8px">'
        f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;color:#00e5ff;font-size:11px">━ 中心线</span>'
        f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;color:#ff6d00;font-size:11px">⬡ 交叉口</span>'
        f'<span style="background:#1e293b;padding:3px 8px;border-radius:6px;color:#ffff00;font-size:11px">● 节点</span>'
        f'</div></div>'
    )
    done=len(ann); total=len(recs); pct=int(done/total*100) if total else 0
    prog = (
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="background:#1e293b;border-radius:6px;height:8px;width:180px">'
        f'<div style="background:#6366f1;height:8px;border-radius:6px;width:{pct}%"></div></div>'
        f'<span style="color:#94a3b8;font-size:13px">{done}/{total} ({pct}%)</span></div>'
    )
    return img_html, rec.get("meta",{}), prog

# ─────────────────────────── Build app ────────────────────────
def build():
    load_all_data(); _load()

    css = """
    .gradio-container{background:#0f172a !important;color:#e2e8f0 !important}
    footer{display:none!important}
    .cat-btn{font-size:14px!important;font-weight:600!important;border-radius:8px!important;min-height:44px!important}
    .nav-btn{border-radius:8px!important}
    label{color:#94a3b8!important}
    """

    with gr.Blocks(title="道路地图标注平台") as demo:
        s_user  = gr.State("")
        s_batch = gr.State("")
        s_idx   = gr.State(0)

        gr.HTML("""<div style="text-align:center;padding:20px 0 4px;background:#0f172a">
          <h1 style="color:#e2e8f0;font-size:26px;font-weight:700;margin:0">🗺️ 道路地图标注平台</h1>
          <p style="color:#475569;font-size:13px;margin:4px 0 0">Road Map BEV Patch Annotation · 多用户并发 · 进度自动保存</p>
        </div>""")

        with gr.Tabs() as tabs:

            # ── Tab 1: Home ──
            with gr.Tab("🏠 主页 & 批次", id="home"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 👤 登录")
                        inp_user = gr.Textbox(label="用户名", placeholder="输入用户名…")
                        btn_login = gr.Button("进入平台 →", variant="primary")
                        login_msg = gr.HTML("")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📦 批次状态")
                        btn_ref = gr.Button("🔄 刷新", size="sm")
                        tbl = gr.HTML(batch_table_html())

                gr.Markdown("---")
                gr.Markdown("### 🎯 进入批次标注")
                with gr.Row():
                    dd_batch = gr.Dropdown(choices=[b[0] for b in batches],
                                           label="选择批次",
                                           value=batches[0][0] if batches else None)
                    btn_enter = gr.Button("开始标注 →", variant="primary", scale=0)
                enter_msg = gr.HTML("")

            # ── Tab 2: Annotate ──
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
                            '请先在主页选择批次</div>')
                        meta_json = gr.JSON(label="样本元数据")

                    with gr.Column(scale=2):
                        gr.Markdown("### 🏷️ 标注分类")
                        gr.HTML('<p style="color:#64748b;font-size:12px;margin:0 0 8px">点击类别后自动前进到下一个样本</p>')
                        cat_btns = {}
                        for cat in CATEGORIES:
                            cat_btns[cat] = gr.Button(
                                f"{CAT_ICONS[cat]} {cat}",
                                elem_classes=["cat-btn"],
                                variant="secondary"
                            )
                        gr.Markdown("---")
                        gr.Markdown("### 🧭 导航")
                        with gr.Row():
                            btn_prev = gr.Button("◀ 上一个", elem_classes=["nav-btn"])
                            btn_next = gr.Button("▶ 下一个", elem_classes=["nav-btn"])
                        with gr.Row():
                            inp_jump = gr.Number(label="跳转到编号", value=0, precision=0, scale=2)
                            btn_jump = gr.Button("跳转", scale=1)
                        gr.Markdown("---")
                        gr.Markdown("### 📤 导出 & 管理")
                        btn_export = gr.Button("导出当前批次结果", variant="secondary")
                        export_out = gr.Textbox(label="导出信息", lines=3, interactive=False)
                        btn_release = gr.Button("🔓 释放批次锁", variant="stop", size="sm")
                        release_msg = gr.HTML("")

            # ── Tab 3: Stats ──
            with gr.Tab("📊 统计总览", id="stats"):
                btn_ref_s = gr.Button("🔄 刷新统计")
                stats_out = gr.HTML(stats_html())

        # ─── Callbacks ───

        def cb_login(user):
            if not user.strip():
                return "", "<span style='color:#ef4444'>请输入用户名</span>"
            with _lock: prog = user_progress.get(user)
            hint = (f"<br><span style='color:#64748b;font-size:12px'>上次：{prog['batch_id']} 第{prog['index']}个</span>"
                    if prog else "")
            return user, f"<span style='color:#22c55e'>✅ 欢迎，{user}！</span>{hint}"

        def cb_refresh(): return batch_table_html()

        def cb_enter(user, bid):
            if not user:
                return (gr.update(), 0,
                        "<span style='color:#ef4444'>请先登录</span>",
                        '<div style="color:#ef4444">未登录</div>', {}, "",
                        gr.update())
            ok = acquire(bid, user)
            if not ok:
                with _lock: lk = batch_locks.get(bid,{})
                owner = (lk or {}).get("user","其他人")
                return (gr.update(), 0,
                        f"<span style='color:#ef4444'>🔴 {bid} 正被 {owner} 处理，请选其他批次</span>",
                        '<div style="color:#ef4444">批次锁定中</div>', {}, "",
                        gr.update())
            with _lock: prog = user_progress.get(user,{})
            idx = prog.get("index",0) if prog.get("batch_id")==bid else 0
            with _lock: user_progress[user]={"batch_id":bid,"index":idx}
            _save()
            ih,m,pg = render_at(bid,idx)
            sl = f'<span style="color:#6366f1;font-weight:600">👤 {user}</span> | <span style="color:#e2e8f0">📦 {bid}</span>'
            return (bid, idx,
                    f"<span style='color:#22c55e'>✅ 已锁定 {bid}，开始标注</span>",
                    ih, m, pg,
                    gr.Tabs(selected="annotate"))

        def cb_annotate(user, bid, idx, cat):
            if not bid: return idx,'<div>请先选批次</div>',{},""
            with _lock:
                bls=list(batches); bi=next((b for b in bls if b[0]==bid),None)
            if not bi: return idx,'<div>批次无效</div>',{},""
            _,s,e=bi; recs=all_data[s:e]
            if not recs: return idx,'<div>空批次</div>',{},""
            idx=max(0,min(int(idx),len(recs)-1))
            rid=recs[idx].get("id",str(idx))
            with _lock:
                annotations.setdefault(bid,{})[rid]=cat
                ni=min(idx+1,len(recs)-1)
                user_progress[user]={"batch_id":bid,"index":ni}
            _save()
            ih,m,pg=render_at(bid,ni)
            return ni,ih,m,pg

        def cb_nav(user,bid,idx,d):
            if not bid: return idx,'<div>请先选批次</div>',{},""
            with _lock: bls=list(batches)
            bi=next((b for b in bls if b[0]==bid),None)
            if not bi: return idx,'<div>批次无效</div>',{},""
            _,s,e=bi; ni=max(0,min(int(idx)+d,e-s-1))
            with _lock: user_progress[user]={"batch_id":bid,"index":ni}
            _save(); ih,m,pg=render_at(bid,ni); return ni,ih,m,pg

        def cb_jump(user,bid,j):
            j=int(j or 0)
            ih,m,pg=render_at(bid,j)
            with _lock: user_progress[user]={"batch_id":bid,"index":j}
            _save(); return j,ih,m,pg

        def cb_export(bid):
            if not bid: return "请先选择批次"
            with _lock:
                ann=dict(annotations.get(bid,{})); bls=list(batches)
            bi=next((b for b in bls if b[0]==bid),None)
            if not bi: return "批次不存在"
            _,s,e=bi; recs=all_data[s:e]
            buckets={c:[] for c in CATEGORIES}
            for r in recs:
                cat=ann.get(r.get("id",""))
                if cat and cat in buckets: buckets[cat].append(r)
            out=[]
            for cat,lst in buckets.items():
                if lst:
                    p=OUTPUT_DIR/f"{bid}_{cat}.jsonl"
                    p.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in lst))
                    out.append(f"{cat}: {len(lst)} 条 → {p.name}")
            return ("导出完成：\n"+"\n".join(out)) if out else "暂无已标注数据"

        def cb_release(user,bid):
            if not bid: return "<span style='color:#ef4444'>未选批次</span>"
            release(bid,user)
            return f"<span style='color:#22c55e'>已释放 {bid}</span>"

        # ─── Wire ───
        btn_login.click(cb_login,[inp_user],[s_user,login_msg])
        btn_ref.click(cb_refresh,outputs=[tbl])
        btn_ref_s.click(stats_html,outputs=[stats_out])
        btn_enter.click(cb_enter,[s_user,dd_batch],
                        [s_batch,s_idx,enter_msg,img_disp,meta_json,lbl_progress,tabs])

        for cat,btn in cat_btns.items():
            btn.click(lambda u,b,i,c=cat:cb_annotate(u,b,i,c),
                      [s_user,s_batch,s_idx],[s_idx,img_disp,meta_json,lbl_progress])

        btn_prev.click(lambda u,b,i:cb_nav(u,b,i,-1),
                       [s_user,s_batch,s_idx],[s_idx,img_disp,meta_json,lbl_progress])
        btn_next.click(lambda u,b,i:cb_nav(u,b,i,+1),
                       [s_user,s_batch,s_idx],[s_idx,img_disp,meta_json,lbl_progress])
        btn_jump.click(cb_jump,[s_user,s_batch,inp_jump],
                       [s_idx,img_disp,meta_json,lbl_progress])
        btn_export.click(cb_export,[s_batch],[export_out])
        btn_release.click(cb_release,[s_user,s_batch],[release_msg])
        demo.load(stats_html,outputs=[stats_out])

    return demo

if __name__ == "__main__":
    app = build()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css="""
        .gradio-container{background:#0f172a !important;color:#e2e8f0 !important}
        footer{display:none!important}
        .cat-btn{font-size:14px!important;font-weight:600!important;border-radius:8px!important;min-height:44px!important}
        label{color:#94a3b8!important}
        """,
    )
