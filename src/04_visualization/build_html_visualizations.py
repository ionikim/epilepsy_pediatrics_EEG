"""
build_html_visualizations.py
============================
Builds two HTML visualizations from the adjacency sparse matrix:

1. eeg_vg_network_inter.html
   - 23-electrode functional connectivity network
   - Node = electrode, size/color = mean VG degree per window
   - Edge = Pearson correlation of VG degree sequences (sliding 3-window)
   - Animates through 30 windows

2. eeg_visibility_graph_online_mini_timepoint.html
   - All 23 electrodes in head layout, each shown as small circular VG
   - Node = subsampled timepoint (16 per electrode per window)
   - Edge = actual HVG visibility connection (subsampled)
   - Animates through 30 windows

Usage:
    python build_html_visualizations.py
    python build_html_visualizations.py path/to/adjacency_sparse.npz
"""

import sys
import json
import math
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from pathlib import Path

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
N_ELEC   = 23
N_TIME   = 7680
WINDOW   = 256
N_WIN    = N_TIME // WINDOW   # 30
SEIZURE  = 15

LABELS = ['FP1','FP2','F7','F3','FZ','F4','F8','T7','C3','CZ','C4','T8',
          'P7','P3','PZ','P4','P8','O1','OZ','O2','A1','A2','T9']

# 10-20 head positions (normalized 0-1)
ELEC_POS = [
    [.5,.05],[.55,.05],[.15,.18],[.35,.15],[.5,.12],[.65,.15],[.85,.18],
    [.1,.45],[.3,.38],[.5,.35],[.7,.38],[.9,.45],
    [.1,.7],[.3,.62],[.5,.6],[.7,.62],[.9,.7],
    [.3,.85],[.5,.82],[.7,.85],
    [.05,.55],[.95,.55],[.05,.75]
]

# Subsampling for mini timepoint visualization
N_NODES_MINI = 16   # timepoints shown per electrode per window
STEP_MINI    = WINDOW // N_NODES_MINI  # 16 per window

# ──────────────────────────────────────────────
# Pipeline paths
# ──────────────────────────────────────────────
PIPELINE_DIR = Path.home() / "Desktop" / "epilepsy_pediatrics_EEG"
NPZ_DEFAULT  = (PIPELINE_DIR / "data" / "graphs" / "adjacency_sparse" /
                "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz")
OUT_DIR      = PIPELINE_DIR / "src" / "04_visualization"

# ──────────────────────────────────────────────
# Step 1: Load matrix & compute degrees
# ──────────────────────────────────────────────
def load_matrix(npz_path):
    print(f"Loading: {npz_path}")
    mat = sparse.load_npz(npz_path)
    cx  = mat.tocoo()

    rows_elec = cx.row // N_TIME
    rows_time = cx.row % N_TIME
    same      = rows_elec == (cx.col // N_TIME)

    intra_mat = csr_matrix(
        (np.ones(same.sum()), (cx.row[same], cx.col[same])),
        shape=mat.shape
    )
    intra_deg = (np.array(intra_mat.sum(axis=1)).flatten() +
                 np.array(intra_mat.sum(axis=0)).flatten())
    deg_full  = intra_deg.reshape(N_ELEC, N_TIME)   # (23, 7680)

    print(f"  Matrix shape : {mat.shape}")
    print(f"  Intra edges  : {same.sum():,}")
    print(f"  Pre mean deg : {deg_full[:, :SEIZURE*WINDOW].mean():.4f}")
    print(f"  Ict mean deg : {deg_full[:, SEIZURE*WINDOW:].mean():.4f}")

    return cx, same, rows_elec, rows_time, deg_full


# ──────────────────────────────────────────────
# Step 2: Compute per-window data
# ──────────────────────────────────────────────
def compute_window_data(cx, same, rows_elec, rows_time, deg_full):
    """
    Returns:
        deg_wins  : (N_ELEC, N_WIN)  — mean degree per electrode per window
        edges_wins: list[N_WIN][N_ELEC] — subsampled edges per window per electrode
        deg_mini  : (N_ELEC, N_WIN, N_NODES_MINI) — subsampled degree
    """
    deg_wins   = np.zeros((N_ELEC, N_WIN))
    edges_wins = [[[] for _ in range(N_ELEC)] for _ in range(N_WIN)]
    deg_mini   = np.zeros((N_ELEC, N_WIN, N_NODES_MINI))

    for w in range(N_WIN):
        t_start = w * WINDOW
        t_end   = (w + 1) * WINDOW

        for ei in range(N_ELEC):
            # Mean degree for this electrode in this window
            deg_wins[ei, w] = deg_full[ei, t_start:t_end].mean()

            # Subsampled degree (every STEP_MINI timepoints)
            sub_idx = np.arange(0, WINDOW, STEP_MINI)[:N_NODES_MINI]
            deg_mini[ei, w, :] = deg_full[ei, t_start:t_end][sub_idx]

            # Actual HVG edges (subsampled)
            mask  = (same & (rows_elec == ei) &
                     (rows_time >= t_start) & (rows_time < t_end))
            r_tp  = (cx.row[mask] % N_TIME) - t_start
            c_tp  = (cx.col[mask] % N_TIME) - t_start

            seen  = set()
            for ri, ci in zip(r_tp.tolist(), c_tp.tolist()):
                if ri >= ci or ri < 0 or ci >= WINDOW:
                    continue
                if (ri, ci) in seen:
                    continue
                seen.add((ri, ci))
                # Map to subsampled index
                ri_s = ri // STEP_MINI
                ci_s = ci // STEP_MINI
                if ri_s < N_NODES_MINI and ci_s < N_NODES_MINI and ri_s != ci_s:
                    edges_wins[w][ei].append([int(ri_s), int(ci_s)])

            # Deduplicate subsampled edges
            edges_wins[w][ei] = list({(min(a,b), max(a,b))
                                      for a, b in edges_wins[w][ei]})
            edges_wins[w][ei] = [[a, b] for a, b in edges_wins[w][ei]]

        if w % 5 == 0:
            print(f"  Window {w:2d} done")

    return deg_wins, edges_wins, deg_mini


# ──────────────────────────────────────────────
# Step 3: Compute sliding-window correlation (for network inter)
# ──────────────────────────────────────────────
def compute_sliding_corr(deg_wins, slide=3):
    """
    adj[w][i][j] = Pearson correlation of electrodes i and j
                   using deg_wins over windows [w-slide+1 .. w]
    Returns: adj (N_WIN, N_ELEC, N_ELEC)
    """
    adj = np.zeros((N_WIN, N_ELEC, N_ELEC))

    for w in range(N_WIN):
        w_start = max(0, w - slide + 1)
        seg     = deg_wins[:, w_start:w+1]   # (N_ELEC, slide)

        for i in range(N_ELEC):
            for j in range(i+1, N_ELEC):
                a, b = seg[i], seg[j]
                if len(a) < 2:
                    continue
                ma, mb = a.mean(), b.mean()
                na, nb = a - ma, b - mb
                denom  = np.sqrt((na**2).sum() * (nb**2).sum())
                if denom == 0:
                    continue
                r = max(0.0, float((na * nb).sum() / denom))
                adj[w, i, j] = adj[w, j, i] = round(r, 4)

    return adj


# ──────────────────────────────────────────────
# Step 4: Build eeg_vg_network_inter.html
# ──────────────────────────────────────────────
def build_network_inter_html(deg_wins, adj):
    """
    23-electrode functional connectivity network.
    - Node = electrode
    - Node size/color = mean VG degree per window
    - Edge = Pearson correlation (sliding 3-window)
    """

    # Pack data as JSON
    data = {
        'deg':       [[round(float(deg_wins[ei, w]), 4) for w in range(N_WIN)]
                      for ei in range(N_ELEC)],
        'adj':       [[[round(float(adj[w, i, j]), 4) for j in range(N_ELEC)]
                        for i in range(N_ELEC)]
                      for w in range(N_WIN)],
        'n_elec':    N_ELEC,
        'n_windows': N_WIN,
        'seizure':   SEIZURE,
        'labels':    LABELS
    }
    data_json = json.dumps(data)

    pos_js = json.dumps(ELEC_POS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EEG Visibility Graph Network</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a12;color:#ccc;font-family:'Courier New',monospace;display:flex;flex-direction:column;height:100vh}}
#hdr{{padding:8px 16px;border-bottom:1px solid #151525;display:flex;align-items:center;gap:12px;flex-shrink:0}}
#hdr h1{{font-size:11px;color:#444;letter-spacing:.1em;text-transform:uppercase}}
#phase{{padding:2px 8px;border-radius:3px;font-size:10px;transition:all .4s}}
.pre{{background:rgba(40,100,220,.15);color:#4d9fff;border:1px solid rgba(40,100,220,.3)}}
.ict{{background:rgba(220,60,30,.15);color:#ff5533;border:1px solid rgba(220,60,30,.3)}}
#stats{{margin-left:auto;display:flex;gap:16px;font-size:10px;color:#555}}
#stats b{{color:#888}}
#thr-wrap{{display:flex;align-items:center;gap:6px;font-size:10px;color:#555}}
#cw{{flex:1;position:relative}}
canvas{{display:block;width:100%;height:100%}}
#ctrl{{padding:8px 16px;border-top:1px solid #151525;display:flex;align-items:center;gap:12px;flex-shrink:0}}
#pb{{width:28px;height:28px;border-radius:50%;background:#4d9fff;border:none;cursor:pointer;font-size:13px;color:#fff}}
#tl{{flex:1;position:relative;height:26px;cursor:pointer}}
#tl-bg{{position:absolute;top:50%;transform:translateY(-50%);width:100%;height:2px;background:#151525}}
#tl-fill{{position:absolute;top:50%;transform:translateY(-50%);height:2px;background:#fff;opacity:.25;width:0%}}
#tl-sz{{position:absolute;top:0;height:100%;width:1px;background:rgba(255,85,51,.6);left:50%}}
#tl-cur{{position:absolute;top:50%;transform:translate(-50%,-50%);width:10px;height:10px;border-radius:50%;background:#fff;left:0%;transition:left .12s}}
.tlb{{position:absolute;bottom:0;font-size:8px;color:#333;transform:translateX(-50%)}}
#lbl{{font-size:10px;color:#555;min-width:50px}}
</style>
</head>
<body>
<div id="hdr">
  <h1>EEG Visibility Graph Network &middot; CHB01 &middot; 23 Electrodes</h1>
  <div id="phase" class="pre">PRE-ICTAL</div>
  <div id="stats">
    <span>t=<b id="sw">0s</b></span>
    <span>edges=<b id="se">0</b></span>
    <span>max corr=<b id="sc">0.00</b></span>
  </div>
  <div id="thr-wrap">
    threshold: <input type="range" id="thr" min="0" max="100" value="30" style="width:80px;accent-color:#4d9fff">
    <span id="thr-val">0.30</span>
  </div>
</div>
<div id="cw"><canvas id="c"></canvas></div>
<div id="ctrl">
  <button id="pb" onclick="togglePlay()">&#9654;</button>
  <div id="tl">
    <div id="tl-bg"></div><div id="tl-fill"></div>
    <div id="tl-sz"></div><div id="tl-cur"></div>
    <div class="tlb" style="left:0%">0s</div>
    <div class="tlb" style="left:16.7%">5s</div>
    <div class="tlb" style="left:33.3%">10s</div>
    <div class="tlb" style="left:50%;color:#ff5533">&#9889;15s</div>
    <div class="tlb" style="left:66.7%">20s</div>
    <div class="tlb" style="left:83.3%">25s</div>
    <div class="tlb" style="left:100%">30s</div>
  </div>
  <div id="lbl">t=0s</div>
</div>
<script>
const D={data_json};
const POS={pos_js};
const NW=D.n_windows,NE=D.n_elec,SZ=D.seizure,LB=D.labels;
let W=0,playing=false,timer=null;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
document.getElementById('thr').oninput=e=>{{
  document.getElementById('thr-val').textContent=(e.target.value/100).toFixed(2);
  draw(W);
}};
function resize(){{const cw=document.getElementById('cw');cv.width=cw.clientWidth;cv.height=cw.clientHeight;draw(W);}}
window.addEventListener('resize',resize);setTimeout(resize,50);
function getThreshold(){{return document.getElementById('thr').value/100;}}
function draw(w){{
  const CW=cv.width,CH=cv.height,ictal=w>=SZ;
  ctx.fillStyle='#0a0a12';ctx.fillRect(0,0,CW,CH);
  const PAD=60;
  const pos=POS.map(p=>([PAD+p[0]*(CW-2*PAD), PAD+p[1]*(CH-2*PAD)]));
  const deg=D.deg.map(d=>d[w]);
  const maxDeg=Math.max(...deg,0.001);
  const adj=D.adj[w];
  const thr=getThreshold();
  let nEdges=0,maxCorr=0;
  // Draw edges
  for(let i=0;i<NE;i++){{
    for(let j=i+1;j<NE;j++){{
      const r=adj[i][j];
      if(r<thr)continue;
      nEdges++;maxCorr=Math.max(maxCorr,r);
      const alpha=0.15+(r-thr)/(1-thr+0.001)*0.7;
      const t=(r-thr)/(1-thr+0.001);
      const R=Math.round(77+t*178),G=Math.round(159-t*100),B=Math.round(255-t*200);
      ctx.beginPath();ctx.moveTo(pos[i][0],pos[i][1]);ctx.lineTo(pos[j][0],pos[j][1]);
      ctx.strokeStyle=`rgba(${{R}},${{G}},${{B}},${{alpha}})`;
      ctx.lineWidth=0.5+r*4;ctx.stroke();
    }}
  }}
  // Draw head outline
  ctx.beginPath();ctx.ellipse(CW/2,CH/2,CW*0.44,CH*0.46,0,0,Math.PI*2);
  ctx.strokeStyle='rgba(255,255,255,0.03)';ctx.lineWidth=1;ctx.stroke();
  // Draw nodes
  pos.forEach((p,ei)=>{{
    const d=deg[ei],t=d/maxDeg,r=6+t*18;
    const grd=ctx.createRadialGradient(p[0],p[1],0,p[0],p[1],r*2);
    const nc=ictal?`rgba(255,85,51,${{0.3+t*0.5}})`:`rgba(77,159,255,${{0.3+t*0.5}})`;
    grd.addColorStop(0,nc);grd.addColorStop(1,'rgba(0,0,0,0)');
    ctx.beginPath();ctx.arc(p[0],p[1],r*2,0,Math.PI*2);ctx.fillStyle=grd;ctx.fill();
    ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);
    ctx.fillStyle=ictal?`rgba(255,85,51,${{0.5+t*0.5}})`:`rgba(77,159,255,${{0.5+t*0.5}})`;
    ctx.fill();
    ctx.font=`${{Math.max(8,r*0.7)}}px Courier New`;
    ctx.fillStyle=`rgba(255,255,255,${{0.4+t*0.5}})`;
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(LB[ei],p[0],p[1]);
  }});
  document.getElementById('sw').textContent=w<SZ?`${{w}}s`:`+${{w-SZ}}s`;
  document.getElementById('se').textContent=nEdges;
  document.getElementById('sc').textContent=maxCorr.toFixed(2);
  const badge=document.getElementById('phase');
  badge.textContent=ictal?'ICTAL':'PRE-ICTAL';badge.className=ictal?'ict':'pre';
  const pct=(w/(NW-1))*100;
  document.getElementById('tl-fill').style.width=pct+'%';
  document.getElementById('tl-cur').style.left=pct+'%';
  document.getElementById('lbl').textContent=`t=${{w<SZ?w+'s':'+'+(w-SZ)+'s'}}`;
}}
function togglePlay(){{playing=!playing;document.getElementById('pb').textContent=playing?'&#9646;&#9646;':'&#9654;';if(playing)step();else clearTimeout(timer);}}
function step(){{if(!playing)return;W=(W+1)%NW;draw(W);timer=setTimeout(step,800);}}
document.getElementById('tl').addEventListener('click',e=>{{
  const r=e.currentTarget.getBoundingClientRect();
  W=Math.round(((e.clientX-r.left)/r.width)*(NW-1));draw(W);
}});
draw(0);
</script>
</body>
</html>"""
    return html


# ──────────────────────────────────────────────
# Step 5: Build eeg_visibility_graph_online_mini_timepoint.html
# ──────────────────────────────────────────────
def build_mini_timepoint_html(deg_mini, edges_wins):
    """
    All 23 electrodes in head layout, each as small circular VG.
    Nodes = subsampled timepoints (16 per electrode per window).
    Edges = actual HVG connections (subsampled).
    """
    # Pack data
    data = {
        'deg':       [[[round(float(deg_mini[ei, w, k]), 4)
                        for k in range(N_NODES_MINI)]
                       for w in range(N_WIN)]
                      for ei in range(N_ELEC)],
        'edges':     edges_wins,
        'n_elec':    N_ELEC,
        'n_windows': N_WIN,
        'n_nodes':   N_NODES_MINI,
        'seizure':   SEIZURE,
        'labels':    LABELS
    }
    data_json = json.dumps(data)
    pos_js    = json.dumps(ELEC_POS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EEG Visibility Graph &mdash; Seizure Network</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a12;color:#ccc;font-family:'Courier New',monospace;display:flex;flex-direction:column;height:100vh}}
#hdr{{padding:8px 16px;border-bottom:1px solid #151525;display:flex;align-items:center;gap:10px;flex-shrink:0}}
#hdr h1{{font-size:11px;color:#444;letter-spacing:.1em;text-transform:uppercase}}
#phase{{padding:2px 8px;border-radius:3px;font-size:10px;transition:all .4s}}
.pre{{background:rgba(40,100,220,.15);color:#4d9fff;border:1px solid rgba(40,100,220,.3)}}
.ict{{background:rgba(220,60,30,.15);color:#ff5533;border:1px solid rgba(220,60,30,.3)}}
#stats{{margin-left:auto;display:flex;gap:16px;font-size:10px;color:#555}}
#stats b{{color:#888}}
#cw{{flex:1;position:relative}}
canvas{{display:block;width:100%;height:100%}}
#ctrl{{padding:8px 16px;border-top:1px solid #151525;display:flex;align-items:center;gap:12px;flex-shrink:0}}
#pb{{width:28px;height:28px;border-radius:50%;background:#4d9fff;border:none;cursor:pointer;font-size:13px;color:#fff}}
#tl{{flex:1;position:relative;height:26px;cursor:pointer}}
#tl-bg{{position:absolute;top:50%;transform:translateY(-50%);width:100%;height:2px;background:#151525}}
#tl-fill{{position:absolute;top:50%;transform:translateY(-50%);height:2px;background:#fff;opacity:.25;width:0%}}
#tl-sz{{position:absolute;top:0;height:100%;width:1px;background:rgba(255,85,51,.6);left:50%}}
#tl-cur{{position:absolute;top:50%;transform:translate(-50%,-50%);width:10px;height:10px;border-radius:50%;background:#fff;left:0%;transition:left .12s}}
.tlb{{position:absolute;bottom:0;font-size:8px;color:#333;transform:translateX(-50%)}}
#lbl{{font-size:10px;color:#555;min-width:50px}}
#metrics{{display:flex;gap:16px;font-size:10px;color:#555}}
#metrics b{{color:#888}}
</style>
</head>
<body>
<div id="hdr">
  <h1>EEG Visibility Graph &middot; CHB01 &middot; Seizure Onset</h1>
  <div id="phase" class="pre">PRE-ICTAL</div>
  <div id="stats">
    <span>t=<b id="sw">0s</b></span>
    <span>edges=<b id="se">0</b></span>
    <span>mean deg=<b id="sd">0.00</b></span>
    <span>top: <b id="st">-</b></span>
  </div>
</div>
<div id="cw"><canvas id="c"></canvas></div>
<div id="ctrl">
  <button id="pb" onclick="togglePlay()">&#9654;</button>
  <div id="tl">
    <div id="tl-bg"></div><div id="tl-fill"></div>
    <div id="tl-sz"></div><div id="tl-cur"></div>
    <div class="tlb" style="left:0%">0s</div>
    <div class="tlb" style="left:16.7%">5s</div>
    <div class="tlb" style="left:33.3%">10s</div>
    <div class="tlb" style="left:50%;color:#ff5533">&#9889;15s</div>
    <div class="tlb" style="left:66.7%">20s</div>
    <div class="tlb" style="left:83.3%">25s</div>
    <div class="tlb" style="left:100%">30s</div>
  </div>
  <div id="lbl">t=0s</div>
</div>
<script>
const DATA={data_json};
const POS={pos_js};
const NW=DATA.n_windows,NE=DATA.n_elec,NN=DATA.n_nodes,SZ=DATA.seizure,LB=DATA.labels;
let W=0,playing=false,timer=null;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
function resize(){{const cw=document.getElementById('cw');cv.width=cw.clientWidth;cv.height=cw.clientHeight;draw(W);}}
window.addEventListener('resize',resize);setTimeout(resize,50);
function edgeColor(i,j){{
  const dist=Math.abs(i-j),t=Math.min(1,dist/NN);
  const stops=[[58,12,163],[67,97,238],[76,201,240],[122,229,130],[248,150,30]];
  const idx=(1-t)*(stops.length-1),si=Math.min(Math.floor(idx),stops.length-2),f=idx-si;
  const c1=stops[si],c2=stops[si+1];
  return[Math.round(c1[0]*(1-f)+c2[0]*f),Math.round(c1[1]*(1-f)+c2[1]*f),Math.round(c1[2]*(1-f)+c2[2]*f)];
}}
function draw(w){{
  const CW=cv.width,CH=cv.height,ictal=w>=SZ;
  ctx.fillStyle='#0a0a12';ctx.fillRect(0,0,CW,CH);
  const PAD=30,R=Math.min(CW,CH)*0.065;
  // Head outline
  ctx.beginPath();ctx.ellipse(CW/2,CH/2,CW*0.44,CH*0.46,0,0,Math.PI*2);
  ctx.strokeStyle='rgba(255,255,255,0.03)';ctx.lineWidth=1;ctx.stroke();
  let totalEdges=0,totalDeg=0,topElec='',topDeg=0;
  POS.forEach((p,ei)=>{{
    const ex=PAD+p[0]*(CW-2*PAD),ey=PAD+p[1]*(CH-2*PAD);
    const deg=DATA.deg[ei][w];
    const edges=DATA.edges[w][ei];
    const maxDeg=Math.max(...deg,0.001);
    const meanDeg=deg.reduce((a,b)=>a+b,0)/NN;
    totalEdges+=edges.length;totalDeg+=meanDeg;
    if(meanDeg>topDeg){{topDeg=meanDeg;topElec=LB[ei];}}
    // Node positions on circle
    const pts=Array.from({{length:NN}},(_,k)=>{{
      const a=(k/NN)*Math.PI*2-Math.PI/2;
      return{{x:ex+R*0.82*Math.cos(a),y:ey+R*0.82*Math.sin(a)}};
    }});
    // Edges
    edges.forEach(([a,b])=>{{
      if(a<0||a>=NN||b<0||b>=NN)return;
      const[r,g,bl]=edgeColor(a,b),dist=Math.abs(a-b);
      const alpha=0.2+(dist/NN)*0.6;
      const p1=pts[a],p2=pts[b];
      const tc=0.3+(1-dist/NN)*0.4;
      ctx.beginPath();ctx.moveTo(p1.x,p1.y);
      ctx.quadraticCurveTo(ex*tc+(p1.x+p2.x)/2*(1-tc),ey*tc+(p1.y+p2.y)/2*(1-tc),p2.x,p2.y);
      ctx.strokeStyle=`rgba(${{r}},${{g}},${{bl}},${{alpha}})`;
      ctx.lineWidth=0.8+(dist/NN)*2;ctx.stroke();
    }});
    // Ring
    const meanT=meanDeg/maxDeg;
    ctx.beginPath();ctx.arc(ex,ey,R,0,Math.PI*2);
    ctx.strokeStyle=ictal?`rgba(255,85,51,${{0.06+meanT*0.12}})`:`rgba(77,159,255,${{0.04+meanT*0.08}})`;
    ctx.lineWidth=1+meanT*1.5;ctx.stroke();
    // Nodes
    pts.forEach((pt,k)=>{{
      const d=deg[k],t=d/maxDeg,nr=1.5+t*5;
      ctx.beginPath();ctx.arc(pt.x,pt.y,nr,0,Math.PI*2);
      ctx.fillStyle=d>0?(ictal?`rgba(255,85,51,${{0.4+t*0.6}})`:`rgba(77,159,255,${{0.4+t*0.6}})`):'rgba(255,255,255,0.08)';
      ctx.fill();
    }});
    // Label
    ctx.font=`bold ${{Math.max(6,R*0.22)}}px Courier New`;
    ctx.fillStyle=ictal?'rgba(255,85,51,.7)':'rgba(77,159,255,.7)';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(LB[ei],ex,ey);
  }});
  // Update stats
  document.getElementById('sw').textContent=w<SZ?`${{w}}s`:`+${{w-SZ}}s`;
  document.getElementById('se').textContent=totalEdges;
  document.getElementById('sd').textContent=(totalDeg/NE).toFixed(2);
  document.getElementById('st').textContent=topElec;
  const badge=document.getElementById('phase');
  badge.textContent=ictal?'ICTAL':'PRE-ICTAL';badge.className=ictal?'ict':'pre';
  const pct=(w/(NW-1))*100;
  document.getElementById('tl-fill').style.width=pct+'%';
  document.getElementById('tl-cur').style.left=pct+'%';
  document.getElementById('lbl').textContent=`t=${{w<SZ?w+'s':'+'+(w-SZ)+'s'}}`;
}}
function togglePlay(){{playing=!playing;document.getElementById('pb').textContent=playing?'&#9646;&#9646;':'&#9654;';if(playing)step();else clearTimeout(timer);}}
function step(){{if(!playing)return;W=(W+1)%NW;draw(W);timer=setTimeout(step,800);}}
document.getElementById('tl').addEventListener('click',e=>{{
  const r=e.currentTarget.getBoundingClientRect();
  W=Math.round(((e.clientX-r.left)/r.width)*(NW-1));draw(W);
}});
draw(0);
</script>
</body>
</html>"""
    return html


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else NPZ_DEFAULT
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    cx, same, rows_elec, rows_time, deg_full = load_matrix(npz_path)

    # Compute window data
    print("\nComputing per-window data...")
    deg_wins, edges_wins, deg_mini = compute_window_data(
        cx, same, rows_elec, rows_time, deg_full)

    # Compute sliding correlation
    print("\nComputing sliding-window correlation...")
    adj = compute_sliding_corr(deg_wins, slide=3)

    # Build HTML 1: network inter
    print("\nBuilding eeg_vg_network_inter.html...")
    html1 = build_network_inter_html(deg_wins, adj)
    out1 = OUT_DIR / 'eeg_vg_network_inter.html'
    with open(out1, 'w', encoding='utf-8') as f:
        f.write(html1)
    print(f"  ✓ {out1}  ({len(html1)//1024} KB)")

    # Build HTML 2: mini timepoint
    print("\nBuilding eeg_visibility_graph_online_mini_timepoint.html...")
    html2 = build_mini_timepoint_html(deg_mini, edges_wins)
    out2 = OUT_DIR / 'eeg_visibility_graph_online_mini_timepoint.html'
    with open(out2, 'w', encoding='utf-8') as f:
        f.write(html2)
    print(f"  ✓ {out2}  ({len(html2)//1024} KB)")

    print("\nDone! Open the HTML files in Chrome.")
