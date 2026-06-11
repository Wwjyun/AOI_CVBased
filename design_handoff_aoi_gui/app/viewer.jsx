// ============================================================
// AOI Console — image viewer（模擬影像 + 缺陷 overlay + 縮放/平移）
// ============================================================

const IMG_W = 1024;
const IMG_H = 768;

// 以 canvas 畫出模擬的 PCB pad 陣列當作檢測影像
function drawSimBoard(canvas) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  // substrate
  ctx.fillStyle = "#27343b";
  ctx.fillRect(0, 0, W, H);

  // deterministic pseudo-random
  let seed = 7;
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };

  // subtle substrate noise
  for (let i = 0; i < 2600; i++) {
    const x = rand() * W, y = rand() * H;
    ctx.fillStyle = `rgba(255,255,255,${rand() * 0.035})`;
    ctx.fillRect(x, y, 1.5, 1.5);
  }

  // pad grid 8 x 6
  const cols = 8, rows = 6;
  const cellW = W / cols, cellH = H / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cx = c * cellW + cellW / 2;
      const cy = r * cellH + cellH / 2;
      const pw = cellW * 0.62, ph = cellH * 0.58;

      // copper pad
      const grad = ctx.createLinearGradient(cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2);
      grad.addColorStop(0, "#8a7a52");
      grad.addColorStop(0.5, "#a8966a");
      grad.addColorStop(1, "#7d6e4a");
      ctx.fillStyle = grad;
      roundRect(ctx, cx - pw / 2, cy - ph / 2, pw, ph, 6);
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // inner circle (via)
      ctx.fillStyle = "#3d4a51";
      ctx.beginPath();
      ctx.arc(cx, cy, Math.min(pw, ph) * 0.16, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // fiducial corners
  ctx.fillStyle = "#cfd6cf";
  [[26, 26], [W - 26, 26], [26, H - 26], [W - 26, H - 26]].forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
  });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

const DEFECT_COLOR = { blob: "#ff5d52", scratch: "#ffb13d", uniformity: "#5db6ff" };

function ImageViewer({
  imageLoaded, imageName, defects, selectedDefect, onSelectDefect,
  showOverlay, onToggleOverlay, running, runPct,
}) {
  const wrapRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const [view, setView] = React.useState({ scale: 1, tx: 0, ty: 0, fitted: false });
  const [cursor, setCursor] = React.useState(null);
  const dragRef = React.useRef(null);

  // 畫模擬影像
  React.useEffect(() => {
    if (imageLoaded && canvasRef.current) drawSimBoard(canvasRef.current);
  }, [imageLoaded]);

  const fit = React.useCallback(() => {
    const el = wrapRef.current;
    if (!el) return;
    const pad = 28;
    const sw = (el.clientWidth - pad * 2) / IMG_W;
    const sh = (el.clientHeight - pad * 2) / IMG_H;
    const scale = Math.min(sw, sh);
    setView({
      scale,
      tx: (el.clientWidth - IMG_W * scale) / 2,
      ty: (el.clientHeight - IMG_H * scale) / 2,
      fitted: true,
    });
  }, []);

  React.useEffect(() => {
    if (imageLoaded) fit();
  }, [imageLoaded, fit]);

  React.useEffect(() => {
    const onResize = () => { if (imageLoaded) fit(); };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [imageLoaded, fit]);

  const zoomAt = (factor, px, py) => {
    setView((v) => {
      const scale = Math.min(8, Math.max(0.05, v.scale * factor));
      const k = scale / v.scale;
      return { scale, tx: px - (px - v.tx) * k, ty: py - (py - v.ty) * k, fitted: false };
    });
  };

  const onWheel = (e) => {
    if (!imageLoaded) return;
    e.preventDefault();
    const rect = wrapRef.current.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - rect.left, e.clientY - rect.top);
  };

  const onMouseDown = (e) => {
    if (!imageLoaded) return;
    dragRef.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false };
  };
  const onMouseMove = (e) => {
    const rect = wrapRef.current.getBoundingClientRect();
    const ix = (e.clientX - rect.left - view.tx) / view.scale;
    const iy = (e.clientY - rect.top - view.ty) / view.scale;
    setCursor(ix >= 0 && iy >= 0 && ix <= IMG_W && iy <= IMG_H ? { x: ix, y: iy } : null);

    const d = dragRef.current;
    if (d) {
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
      setView((v) => ({ ...v, tx: d.tx + dx, ty: d.ty + dy, fitted: false }));
    }
  };
  const onMouseUp = () => { dragRef.current = null; };

  const center = (zoomFn) => {
    const el = wrapRef.current;
    zoomFn(el.clientWidth / 2, el.clientHeight / 2);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* viewer toolbar */}
      <div style={{
        display: "flex", alignItems: "center", gap: 4,
        padding: "6px 10px",
        background: "var(--viewer-bg-2)",
        borderBottom: "1px solid rgba(255,255,255,0.07)",
        borderRadius: "var(--r-lg) var(--r-lg) 0 0",
      }}>
        <span style={{
          fontFamily: "var(--font-mono)", fontSize: "var(--fs-mono)",
          color: "rgba(255,255,255,0.6)",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          marginRight: "auto", paddingLeft: 4,
        }}>{imageLoaded ? imageName : "尚未載入影像"}</span>

        <div style={{ display: "flex", alignItems: "center", gap: 2, color: "rgba(255,255,255,0.75)" }}>
          <IconBtn title="縮小" onClick={() => imageLoaded && center((x, y) => zoomAt(1 / 1.25, x, y))} style={{ color: "inherit" }}><IcZoomOut size={16} /></IconBtn>
          <IconBtn title="放大" onClick={() => imageLoaded && center((x, y) => zoomAt(1.25, x, y))} style={{ color: "inherit" }}><IcZoomIn size={16} /></IconBtn>
          <IconBtn title="符合視窗" onClick={() => imageLoaded && fit()} style={{ color: "inherit" }}><IcFit size={16} /></IconBtn>
        </div>
        <div style={{ width: 1, height: 18, background: "rgba(255,255,255,0.12)", margin: "0 6px" }}></div>
        <button
          className="btn btn-sm"
          onClick={onToggleOverlay}
          style={{
            background: showOverlay ? "var(--accent)" : "rgba(255,255,255,0.08)",
            color: showOverlay ? "#fff" : "rgba(255,255,255,0.65)",
            border: "none",
          }}
        >
          <IcEye size={13} /> 缺陷 Overlay
        </button>
      </div>

      {/* stage */}
      <div
        ref={wrapRef}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        style={{
          flex: 1, position: "relative", overflow: "hidden",
          background: "var(--viewer-bg)",
          cursor: imageLoaded ? (dragRef.current ? "grabbing" : "grab") : "default",
          minHeight: 0,
        }}
      >
        {!imageLoaded && (
          <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>
            <EmptyState
              icon={<IcImage size={40} strokeWidth={1.2} />}
              title="尚未載入檢測影像"
              hint="從上方工具列載入影像，或將檔案拖曳到此處"
            />
          </div>
        )}

        {imageLoaded && (
          <div style={{
            position: "absolute", left: 0, top: 0,
            transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`,
            transformOrigin: "0 0",
            width: IMG_W, height: IMG_H,
          }}>
            <canvas ref={canvasRef} width={IMG_W} height={IMG_H}
              style={{ display: "block", boxShadow: "0 0 0 1px rgba(255,255,255,0.1), 0 12px 40px rgba(0,0,0,0.5)" }} />

            {/* defect boxes */}
            {showOverlay && defects.map((d) => {
              const sel = selectedDefect === d.id;
              const color = DEFECT_COLOR[d.type] || "#ff5d52";
              return (
                <div
                  key={d.id}
                  onClick={(e) => { e.stopPropagation(); if (!dragRef.current || !dragRef.current.moved) onSelectDefect(sel ? null : d.id); }}
                  title={`#${d.id} ${d.type} · detector ${d.detector}`}
                  style={{
                    position: "absolute",
                    left: d.x * IMG_W, top: d.y * IMG_H,
                    width: d.w * IMG_W, height: d.h * IMG_H,
                    border: `${sel ? 2.5 : 1.5}px solid ${color}`,
                    borderRadius: 2,
                    boxShadow: sel ? `0 0 0 3px ${color}55, 0 0 18px ${color}88` : `0 0 8px ${color}44`,
                    cursor: "pointer",
                    animation: "aoi-fade-in 0.3s ease",
                    zIndex: sel ? 3 : 2,
                  }}
                >
                  <span style={{
                    position: "absolute", top: -20, left: -2,
                    fontFamily: "var(--font-mono)", fontSize: 11, fontWeight: 600,
                    color: "#10171a", background: color,
                    padding: "1px 6px", borderRadius: 3,
                    whiteSpace: "nowrap",
                    display: sel ? "block" : "none",
                  }}>#{d.id} {d.type} {d.score.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* scan line while running */}
        {running && (
          <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden" }}>
            <div style={{
              position: "absolute", left: 0, right: 0, height: 2,
              background: "linear-gradient(90deg, transparent, var(--accent) 30%, var(--accent) 70%, transparent)",
              boxShadow: "0 0 14px var(--accent)",
              animation: "aoi-scan 1.6s linear infinite",
            }}></div>
            <div style={{
              position: "absolute", right: 14, top: 12,
              display: "flex", alignItems: "center", gap: 8,
              background: "rgba(13, 20, 24, 0.82)",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 6, padding: "6px 12px",
              color: "rgba(255,255,255,0.85)",
              fontFamily: "var(--font-mono)", fontSize: 12,
            }}>
              <span className="spinner"></span> 檢測中 {runPct}%
            </div>
          </div>
        )}
      </div>

      {/* status strip */}
      <div style={{
        display: "flex", alignItems: "center", gap: 16,
        padding: "5px 12px",
        background: "var(--viewer-bg-2)",
        borderTop: "1px solid rgba(255,255,255,0.07)",
        borderRadius: "0 0 var(--r-lg) var(--r-lg)",
        fontFamily: "var(--font-mono)", fontSize: 11,
        color: "rgba(255,255,255,0.5)",
      }}>
        <span>{imageLoaded ? `4096 × 3072 px` : "— × — px"}</span>
        <span>zoom {imageLoaded ? Math.round(view.scale * 100 * 4) / 4 * 1 : 0}%</span>
        <span style={{ marginLeft: "auto" }}>
          {cursor ? `x ${Math.round(cursor.x * 4)}  y ${Math.round(cursor.y * 4)}` : "x —  y —"}
        </span>
      </div>
    </div>
  );
}

Object.assign(window, { ImageViewer, DEFECT_COLOR, IMG_W, IMG_H });
