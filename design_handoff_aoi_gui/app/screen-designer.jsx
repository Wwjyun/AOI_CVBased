// ============================================================
// AOI Console — Recipe 設計 screen
// ============================================================

const TILE_MODES = [
  { value: "pattern_match", label: "Pattern Match" },
  { value: "grid", label: "Grid" },
  { value: "contour", label: "Contour" },
];

function TilePreviewCanvas({ previewed }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const board = getSimBoardCanvas();
    ctx.drawImage(board, 0, 0, canvas.width, canvas.height);
    if (previewed) {
      // pattern match 框（對齊 8x6 pad 陣列，取中間 24 個示意）
      const cols = 8, rows = 6;
      const cw = canvas.width / cols, ch = canvas.height / rows;
      ctx.strokeStyle = "#39d98a";
      ctx.lineWidth = 1.5;
      ctx.font = "8px monospace";
      ctx.fillStyle = "#39d98a";
      let n = 0;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = c * cw + cw * 0.16, y = r * ch + ch * 0.18;
          ctx.strokeRect(x, y, cw * 0.68, ch * 0.64);
          n += 1;
        }
      }
    }
  }, [previewed]);
  return (
    <canvas
      ref={ref} width={416} height={312}
      style={{ width: "100%", borderRadius: "var(--r-md)", display: "block", border: "1px solid var(--border)" }}
    />
  );
}

function DesignerScreen({ app }) {
  const [meta, setMeta] = React.useState({
    recipe_name: "PRODUCT_A_PATTERN_MATCH_000_AOI_01",
    product_id: "PRODUCT_A",
    machine_id: "AOI_01",
    version: "0.1.0",
  });
  const [tileMode, setTileMode] = React.useState("pattern_match");
  const [pm, setPm] = React.useState({
    template_path: "outputs_validation/pattern_template.png",
    match_threshold: 0.8, max_count: 999, nms_threshold: 0.3,
    crop_padding: 8, sort_row_tolerance: 20,
  });
  const [grid, setGrid] = React.useState({ width: 512, height: 512, overlap_x: 64, overlap_y: 64 });
  const [contour, setContour] = React.useState({ min_area: 4000, approx_epsilon: 0.01 });

  const [enabled, setEnabled] = React.useState({ "000": true });
  const [activeDet, setActiveDet] = React.useState("000");
  const [params, setParams] = React.useState(() => {
    const out = {};
    Object.entries(DETECTOR_DEFS).forEach(([id, def]) => { out[id] = { ...def.default_params }; });
    return out;
  });

  const [status, setStatus] = React.useState({ kind: "idle", text: "尚未預覽" });
  const [previewed, setPreviewed] = React.useState(false);

  const setMetaField = (k) => (e) => setMeta({ ...meta, [k]: e.target.value });

  const runPreview = () => {
    if (!app.imageLoaded) {
      setStatus({ kind: "error", text: "請先在「檢測執行」載入影像再預覽切圖" });
      return;
    }
    setStatus({ kind: "busy", text: "Pattern Match 預覽執行中…" });
    setPreviewed(false);
    setTimeout(() => {
      setStatus({ kind: "ok", text: "匹配 48 張小圖；最佳分數：0.9132" });
      setPreviewed(true);
    }, 900);
  };

  const saveRecipe = () => {
    const detectorIds = Object.keys(enabled).filter((id) => enabled[id]);
    if (!detectorIds.length) {
      setStatus({ kind: "error", text: "請至少啟用一個 detector" });
      return;
    }
    app.saveDesignedRecipe({
      file: `${meta.recipe_name}.yaml`,
      recipe_name: meta.recipe_name,
      product_id: meta.product_id,
      machine_id: meta.machine_id,
      version: meta.version,
      tile: { mode: tileMode, ...(tileMode === "pattern_match" ? pm : tileMode === "grid" ? grid : contour) },
      detectors: detectorIds,
    });
    setStatus({ kind: "ok", text: `Recipe 已儲存並載入：recipes/${meta.recipe_name}.yaml` });
  };

  const detectorIds = Object.keys(DETECTOR_DEFS).sort();
  const activeDef = DETECTOR_DEFS[activeDet];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0, gap: 12 }}>
      <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 0 }}>

        {/* 左：recipe + tiling + 預覽 */}
        <div style={{ width: 360, flexShrink: 0, display: "flex", flexDirection: "column", gap: 12, overflowY: "auto" }}>
          <Panel title="Recipe 資訊">
            <FormGrid>
              <FRow label="Recipe 名稱"><TextField mono value={meta.recipe_name} onChange={setMetaField("recipe_name")} /></FRow>
              <FRow label="產品 Product"><TextField mono value={meta.product_id} onChange={setMetaField("product_id")} /></FRow>
              <FRow label="機台 Machine"><TextField mono value={meta.machine_id} onChange={setMetaField("machine_id")} /></FRow>
              <FRow label="版本 Version"><TextField mono value={meta.version} onChange={setMetaField("version")} /></FRow>
            </FormGrid>
          </Panel>

          <Panel title="切圖 Tiling" actions={
            <Segmented options={TILE_MODES} value={tileMode} onChange={setTileMode} />
          }>
            {tileMode === "pattern_match" && (
              <FormGrid>
                <FRow label="Template">
                  <div style={{ display: "flex", gap: 6 }}>
                    <TextField mono value={pm.template_path} onChange={(e) => setPm({ ...pm, template_path: e.target.value })} />
                    <Btn variant="secondary" size="sm" style={{ height: "var(--row-h)" }} icon={<IcFolder size={13} />}
                      onClick={() => setPm({ ...pm, template_path: "templates/pad_template.png" })}>選擇</Btn>
                  </div>
                </FRow>
                <FRow label="匹配門檻"><NumField value={pm.match_threshold} onChange={(v) => setPm({ ...pm, match_threshold: v })} step={0.01} min={0} max={1} decimals={3} /></FRow>
                <FRow label="最大匹配數"><NumField value={pm.max_count} onChange={(v) => setPm({ ...pm, max_count: v })} min={1} max={100000} /></FRow>
                <FRow label="NMS 門檻"><NumField value={pm.nms_threshold} onChange={(v) => setPm({ ...pm, nms_threshold: v })} step={0.01} min={0} max={1} decimals={3} /></FRow>
                <FRow label="裁切外擴 px"><NumField value={pm.crop_padding} onChange={(v) => setPm({ ...pm, crop_padding: v })} min={0} /></FRow>
                <FRow label="排序列容差"><NumField value={pm.sort_row_tolerance} onChange={(v) => setPm({ ...pm, sort_row_tolerance: v })} min={1} /></FRow>
              </FormGrid>
            )}
            {tileMode === "grid" && (
              <FormGrid>
                <FRow label="Tile 寬"><NumField value={grid.width} onChange={(v) => setGrid({ ...grid, width: v })} min={32} /></FRow>
                <FRow label="Tile 高"><NumField value={grid.height} onChange={(v) => setGrid({ ...grid, height: v })} min={32} /></FRow>
                <FRow label="重疊 X"><NumField value={grid.overlap_x} onChange={(v) => setGrid({ ...grid, overlap_x: v })} min={0} /></FRow>
                <FRow label="重疊 Y"><NumField value={grid.overlap_y} onChange={(v) => setGrid({ ...grid, overlap_y: v })} min={0} /></FRow>
              </FormGrid>
            )}
            {tileMode === "contour" && (
              <FormGrid>
                <FRow label="最小面積"><NumField value={contour.min_area} onChange={(v) => setContour({ ...contour, min_area: v })} min={0} /></FRow>
                <FRow label="近似 ε"><NumField value={contour.approx_epsilon} onChange={(v) => setContour({ ...contour, approx_epsilon: v })} step={0.005} decimals={3} min={0} /></FRow>
              </FormGrid>
            )}
          </Panel>

          <Panel title="切圖預覽">
            <TilePreviewCanvas previewed={previewed} />
            <div style={{
              marginTop: 10, display: "flex", alignItems: "center", gap: 8,
              fontSize: "var(--fs-small)",
              color: status.kind === "error" ? "var(--ng)" : status.kind === "ok" ? "var(--accent-text)" : "var(--text-3)",
            }}>
              {status.kind === "busy" && <span className="spinner"></span>}
              <span>{status.text}</span>
            </div>
          </Panel>
        </div>

        {/* 右：detector 選用與參數 */}
        <Panel title="Detector 選用與參數" flush style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
            <div style={{ width: 280, borderRight: "1px solid var(--border)", overflowY: "auto", flexShrink: 0 }}>
              {detectorIds.map((id) => {
                const def = DETECTOR_DEFS[id];
                const on = !!enabled[id];
                return (
                  <div
                    key={id}
                    className={"row-item" + (activeDet === id ? " selected" : "")}
                    onClick={() => setActiveDet(id)}
                  >
                    <Toggle value={on} onChange={(v) => setEnabled({ ...enabled, [id]: v })} />
                    <div style={{ minWidth: 0 }}>
                      <div style={{ display: "flex", alignItems: "baseline", gap: 7 }}>
                        <span className="mono" style={{ fontWeight: 600 }}>{id}</span>
                        <span style={{ fontSize: "var(--fs-small)", color: "var(--text-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{def.zh}</span>
                      </div>
                      <div style={{ fontSize: 11, color: "var(--text-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{def.display_name}</div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ flex: 1, overflowY: "auto", padding: "var(--pad-panel)", minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                <span className="mono" style={{ fontWeight: 700, fontSize: 14 }}>{activeDet}</span>
                <span style={{ color: "var(--text-2)" }}>{activeDef.zh}</span>
                <Badge kind={enabled[activeDet] ? "accent" : "neutral"}>{enabled[activeDet] ? "啟用" : "停用"}</Badge>
              </div>
              <div style={{ maxWidth: 420 }}>
                <FormGrid>
                  {Object.entries(params[activeDet]).map(([k, v]) => (
                    <FRow key={k} label={<span className="mono">{k}</span>}>
                      <ParamControl
                        value={v}
                        onChange={(nv) => setParams({ ...params, [activeDet]: { ...params[activeDet], [k]: nv } })}
                      />
                    </FRow>
                  ))}
                </FormGrid>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      {/* action bar */}
      <div className="panel" style={{
        flexDirection: "row", alignItems: "center", gap: 10,
        padding: "10px var(--pad-panel)", flexShrink: 0,
      }}>
        <span style={{ fontSize: "var(--fs-small)", color: "var(--text-3)" }}>
          已啟用 {Object.values(enabled).filter(Boolean).length} 個 detector
        </span>
        <div style={{ flex: 1 }}></div>
        <Btn variant="secondary" icon={<IcEye size={15} />} onClick={runPreview} disabled={status.kind === "busy"}>
          預覽切圖
        </Btn>
        <Btn variant="primary" icon={<IcSave size={15} />} onClick={saveRecipe} disabled={status.kind === "busy"}>
          儲存 Recipe
        </Btn>
      </div>
    </div>
  );
}

Object.assign(window, { DesignerScreen });
