// ============================================================
// AOI Console — 結果 screen（摘要 + 缺陷表 + NG tile 圖庫）
// ============================================================

let _boardCache = null;
function getSimBoardCanvas() {
  if (!_boardCache) {
    _boardCache = document.createElement("canvas");
    _boardCache.width = IMG_W;
    _boardCache.height = IMG_H;
    drawSimBoard(_boardCache);
  }
  return _boardCache;
}

function TileThumb({ defect, size = 104, selected, onClick }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    const board = getSimBoardCanvas();
    const ctx = ref.current.getContext("2d");
    const cx = (defect.x + defect.w / 2) * IMG_W;
    const cy = (defect.y + defect.h / 2) * IMG_H;
    const crop = Math.max(defect.w * IMG_W, defect.h * IMG_H) * 3 + 56;
    const sx = Math.min(Math.max(cx - crop / 2, 0), IMG_W - crop);
    const sy = Math.min(Math.max(cy - crop / 2, 0), IMG_H - crop);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(board, sx, sy, crop, crop, 0, 0, size, size);
    // defect box
    const k = size / crop;
    ctx.strokeStyle = DEFECT_COLOR[defect.type] || "#ff5d52";
    ctx.lineWidth = 2;
    ctx.strokeRect((defect.x * IMG_W - sx) * k, (defect.y * IMG_H - sy) * k, defect.w * IMG_W * k, defect.h * IMG_H * k);
  }, [defect, size]);

  return (
    <button
      onClick={onClick}
      style={{
        border: selected ? "2px solid var(--accent)" : "1px solid var(--border)",
        borderRadius: "var(--r-md)",
        padding: 0, cursor: "pointer", background: "var(--viewer-bg)",
        overflow: "hidden", position: "relative",
        boxShadow: selected ? "0 0 0 3px var(--accent-soft)" : "var(--shadow-sm)",
        display: "flex", flexDirection: "column",
      }}
    >
      <canvas ref={ref} width={size} height={size} style={{ display: "block" }} />
      <span className="mono" style={{
        position: "absolute", left: 4, top: 4,
        background: "rgba(13,20,24,0.75)", color: "#fff",
        fontSize: 10, padding: "1px 5px", borderRadius: 3,
      }}>#{defect.id}</span>
      <span style={{
        display: "block", width: "100%",
        background: "var(--surface)", borderTop: "1px solid var(--border)",
        fontSize: 10, color: "var(--text-2)", padding: "3px 6px",
        textAlign: "left", fontFamily: "var(--font-mono)",
      }}>{defect.tile} · {defect.detector}</span>
    </button>
  );
}

function StatCard({ label, value, tone }) {
  return (
    <div className="panel" style={{ padding: "12px 16px", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</span>
      <span className="mono" style={{ fontSize: 22, fontWeight: 700, color: tone || "var(--text)" }}>{value}</span>
    </div>
  );
}

function ResultsScreen({ app }) {
  const { result, selectedDefect, setSelectedDefect } = app;
  const [filter, setFilter] = React.useState("all");

  if (!result) {
    return (
      <div style={{ height: "100%", display: "grid", placeItems: "center" }}>
        <EmptyState
          icon={<IcTable size={40} strokeWidth={1.2} />}
          title="尚無檢測結果"
          hint="到「檢測執行」載入影像與 Recipe 後執行檢測，結果會顯示在這裡。"
          action={<Btn variant="primary" size="sm" icon={<IcPlay size={14} />} onClick={() => app.setScreen("run")}>前往檢測執行</Btn>}
        />
      </div>
    );
  }

  const detectorIds = [...new Set(result.defects.map((d) => d.detector))];
  const defects = filter === "all" ? result.defects : result.defects.filter((d) => d.detector === filter);

  const viewInImage = (id) => {
    setSelectedDefect(id);
    app.setShowOverlay(true);
    app.setScreen("run");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", minHeight: 0 }}>
      {/* summary row */}
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr 1fr 1fr", gap: 12, flexShrink: 0 }}>
        <div className="panel" style={{
          padding: "12px 22px", justifyContent: "center",
          background: result.final === "NG" ? "var(--ng-soft)" : "var(--pass-soft)",
          borderColor: result.final === "NG" ? "#f3c6c3" : "#bfe5cc",
        }}>
          <span style={{
            fontSize: 30, fontWeight: 800, letterSpacing: "0.04em",
            fontFamily: "var(--font-mono)",
            color: result.final === "NG" ? "var(--ng)" : "var(--pass)",
          }}>{result.final}</span>
        </div>
        <StatCard label="Tiles" value={result.summary.tile_count} />
        <StatCard label="NG Tiles" value={result.summary.ng_count} tone="var(--ng)" />
        <StatCard label="缺陷數" value={result.summary.defect_count} tone="var(--ng)" />
        <StatCard label="耗時" value={result.dur} />
      </div>

      <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 0 }}>
        {/* defect table */}
        <Panel
          title={`缺陷清單（${defects.length}）`}
          flush
          style={{ flex: 3, minWidth: 0 }}
          actions={
            <div style={{ display: "flex", gap: 2 }}>
              <Segmented
                value={filter}
                onChange={setFilter}
                options={[{ value: "all", label: "全部" }, ...detectorIds.map((id) => ({ value: id, label: id }))]}
              />
            </div>
          }
        >
          <div style={{ overflowY: "auto", height: "100%" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th><th>Tile</th><th>Detector</th><th>類型</th>
                  <th>Global bbox</th><th style={{ textAlign: "right" }}>面積</th><th style={{ textAlign: "right" }}>分數</th><th></th>
                </tr>
              </thead>
              <tbody>
                {defects.map((d) => (
                  <tr
                    key={d.id}
                    className={"clickable" + (selectedDefect === d.id ? " selected" : "")}
                    onClick={() => setSelectedDefect(selectedDefect === d.id ? null : d.id)}
                  >
                    <td className="mono">{d.id}</td>
                    <td className="mono">{d.tile}</td>
                    <td className="mono">{d.detector}</td>
                    <td>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: DEFECT_COLOR[d.type], flexShrink: 0 }}></span>
                        {DEFECT_TYPE_LABEL[d.type] || d.type}
                      </span>
                    </td>
                    <td className="mono">[{Math.round(d.x * 4096)}, {Math.round(d.y * 3072)}, {Math.round(d.w * 4096)}, {Math.round(d.h * 3072)}]</td>
                    <td className="mono" style={{ textAlign: "right" }}>{d.area}</td>
                    <td className="mono" style={{ textAlign: "right" }}>{d.score.toFixed(4)}</td>
                    <td style={{ textAlign: "right" }}>
                      <Btn variant="ghost" size="sm" icon={<IcCrosshair size={13} />} onClick={(e) => { e.stopPropagation(); viewInImage(d.id); }}>
                        檢視
                      </Btn>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* NG tiles + outputs */}
        <div style={{ flex: 1.2, minWidth: 240, display: "flex", flexDirection: "column", gap: 12, minHeight: 0 }}>
          <Panel title="NG Tiles" style={{ flex: 1, minHeight: 0 }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
              {result.defects.map((d) => (
                <TileThumb
                  key={d.id} defect={d}
                  selected={selectedDefect === d.id}
                  onClick={() => viewInImage(d.id)}
                />
              ))}
            </div>
          </Panel>
          <Panel title="輸出檔案" flush>
            {[
              ["overlay/PRODUCT_A_LOT0612_017_overlay.png", "Overlay"],
              ["csv/PRODUCT_A_LOT0612_017.csv", "CSV"],
              ["json/PRODUCT_A_LOT0612_017.json", "JSON"],
            ].map(([path, kind]) => (
              <div key={path} className="row-item" title={`outputs/${path}`}>
                <IcFolder size={14} style={{ color: "var(--text-3)", flexShrink: 0 }} />
                <span className="mono" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text-2)" }}>{path}</span>
                <Badge kind="neutral">{kind}</Badge>
              </div>
            ))}
          </Panel>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { ResultsScreen });
