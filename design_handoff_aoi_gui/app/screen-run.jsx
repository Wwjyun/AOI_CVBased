// ============================================================
// AOI Console — 檢測執行 screen
// ============================================================

function DetectorRow({ id, def, enabled, expanded, onToggleExpand }) {
  return (
    <div style={{ borderBottom: "1px solid var(--surface-3)" }}>
      <div className="row-item" onClick={onToggleExpand} style={{ borderBottom: "none" }}>
        <span style={{ color: "var(--text-3)", display: "flex", transform: expanded ? "rotate(90deg)" : "none", transition: "transform 0.15s" }}>
          <IcChevronR size={13} />
        </span>
        <span className="mono" style={{ fontWeight: 600, color: "var(--text)" }}>{id}</span>
        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text-2)" }}>
          {def.zh}
        </span>
        <Badge kind={enabled ? "accent" : "neutral"}>{enabled ? "啟用" : "停用"}</Badge>
      </div>
      {expanded && (
        <div className="fade-in" style={{ padding: "4px 12px 12px 34px" }}>
          <div style={{ fontSize: "var(--fs-small)", color: "var(--text-3)", marginBottom: 8 }}>{def.display_name}</div>
          <FormGrid>
            {Object.entries(def.default_params).map(([k, v]) => (
              <FRow key={k} label={<span className="mono">{k}</span>}>
                <ParamControl value={v} readOnly />
              </FRow>
            ))}
          </FormGrid>
        </div>
      )}
    </div>
  );
}

function RecipeInfoPanel({ recipe, onOpenRecipe }) {
  const [expandedId, setExpandedId] = React.useState(null);
  if (!recipe) {
    return (
      <Panel title="Recipe">
        <EmptyState
          icon={<IcRecipe size={32} strokeWidth={1.3} />}
          title="尚未載入 Recipe"
          action={<Btn variant="secondary" size="sm" icon={<IcFolder size={14} />} onClick={onOpenRecipe}>載入 Recipe</Btn>}
        />
      </Panel>
    );
  }
  return (
    <Panel title="Recipe" flush actions={
      <Btn variant="ghost" size="sm" onClick={onOpenRecipe}>更換</Btn>
    }>
      <div style={{ padding: "12px var(--pad-panel)", borderBottom: "1px solid var(--surface-3)" }}>
        <div className="mono" style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, wordBreak: "break-all" }}>{recipe.recipe_name}</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Badge kind="neutral">{recipe.product_id}</Badge>
          <Badge kind="neutral">{recipe.machine_id}</Badge>
          <Badge kind="neutral">v{recipe.version}</Badge>
          <Badge kind="accent">{recipe.tile.mode}</Badge>
        </div>
      </div>
      <div style={{ padding: "8px var(--pad-panel) 4px", fontSize: 11, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Detectors（{recipe.detectors.length}）
      </div>
      <div>
        {recipe.detectors.map((id) => (
          <DetectorRow
            key={id} id={id} def={DETECTOR_DEFS[id]} enabled
            expanded={expandedId === id}
            onToggleExpand={() => setExpandedId(expandedId === id ? null : id)}
          />
        ))}
      </div>
    </Panel>
  );
}

function RunControlPanel({ app }) {
  const { imageLoaded, recipe, running, runPct, runMsg, result, startRun, goResults } = app;
  const ready = imageLoaded && recipe && !running;
  return (
    <Panel title="檢測控制">
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <Btn variant="primary" size="lg" icon={running ? <span className="spinner" style={{ borderColor: "rgba(255,255,255,0.35)", borderTopColor: "#fff" }}></span> : <IcPlay size={17} />}
          disabled={!ready} onClick={startRun} style={{ width: "100%" }}>
          {running ? "檢測執行中…" : "開始檢測"}
        </Btn>
        {!imageLoaded && !running && (
          <div style={{ fontSize: "var(--fs-small)", color: "var(--text-3)", textAlign: "center" }}>請先載入影像與 Recipe</div>
        )}

        {(running || result) && (
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <ProgressBar pct={runPct} />
            <span className="mono" style={{ color: "var(--text-2)", width: 38, textAlign: "right" }}>{runPct}%</span>
          </div>
        )}
        {running && <div className="mono" style={{ color: "var(--text-2)", fontSize: 11 }}>{runMsg}</div>}

        {result && !running && (
          <div className="fade-in" style={{
            border: "1px solid var(--border)", borderRadius: "var(--r-md)",
            overflow: "hidden",
          }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 10, padding: "10px 12px",
              background: result.final === "NG" ? "var(--ng-soft)" : "var(--pass-soft)",
            }}>
              <span style={{
                fontSize: 18, fontWeight: 700, letterSpacing: "0.04em",
                color: result.final === "NG" ? "var(--ng)" : "var(--pass)",
              }}>{result.final}</span>
              <span style={{ fontSize: "var(--fs-small)", color: "var(--text-2)" }}>檢測完成 · {result.dur}</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", borderTop: "1px solid var(--border)" }}>
              {[
                ["Tiles", result.summary.tile_count],
                ["NG Tiles", result.summary.ng_count],
                ["缺陷", result.summary.defect_count],
              ].map(([k, v]) => (
                <div key={k} style={{ padding: "8px 12px", textAlign: "center" }}>
                  <div className="mono" style={{ fontSize: 16, fontWeight: 600 }}>{v}</div>
                  <div style={{ fontSize: 10, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{k}</div>
                </div>
              ))}
            </div>
            <button
              onClick={goResults}
              style={{
                width: "100%", border: "none", borderTop: "1px solid var(--border)",
                background: "var(--surface-2)", color: "var(--accent-text)",
                padding: "8px", fontSize: "var(--fs-small)", fontWeight: 600, cursor: "pointer",
              }}
            >查看完整結果 →</button>
          </div>
        )}
      </div>
    </Panel>
  );
}

// OP 模式：大狀態 + 大按鈕，無參數細節
function OpModePanel({ app }) {
  const { imageLoaded, recipe, running, runPct, runMsg, result, startRun } = app;
  const ready = imageLoaded && recipe && !running;
  const stateColor = running ? "var(--accent)" : result ? (result.final === "NG" ? "var(--ng)" : "var(--pass)") : "var(--text-3)";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%" }}>
      <Panel>
        <div style={{ textAlign: "center", padding: "18px 0 22px" }}>
          <div style={{
            fontSize: 44, fontWeight: 800, letterSpacing: "0.05em",
            color: stateColor,
            fontFamily: "var(--font-mono)",
            animation: running ? "aoi-pulse 1.4s ease infinite" : "none",
          }}>
            {running ? `${runPct}%` : result ? result.final : "待機"}
          </div>
          <div style={{ color: "var(--text-3)", fontSize: "var(--fs-small)", marginTop: 6 }}>
            {running ? runMsg : result ? `缺陷 ${result.summary.defect_count} · NG tiles ${result.summary.ng_count}` : "載入影像後按下開始檢測"}
          </div>
        </div>
        <Btn variant="primary" size="lg" icon={<IcPlay size={18} />} disabled={!ready} onClick={startRun}
          style={{ width: "100%", height: 52, fontSize: 16 }}>
          {running ? "檢測中…" : "開始檢測"}
        </Btn>
      </Panel>

      <Panel title="本批紀錄" flush>
        <table className="data-table">
          <tbody>
            {(app.history).map((h, i) => (
              <tr key={i}>
                <td className="mono" style={{ color: "var(--text-3)" }}>{h.time}</td>
                <td><ResultBadge result={h.result} /></td>
                <td className="mono" style={{ textAlign: "right" }}>{h.defects} 缺陷</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

function RunScreen({ app }) {
  return (
    <div style={{ display: "flex", gap: 12, height: "100%", minHeight: 0 }}>
      <div className="panel" style={{ flex: 1, overflow: "hidden", border: "1px solid var(--border)" }}>
        <ImageViewer
          imageLoaded={app.imageLoaded}
          imageName={app.image ? app.image.file : ""}
          defects={app.result ? app.result.defects : []}
          selectedDefect={app.selectedDefect}
          onSelectDefect={app.setSelectedDefect}
          showOverlay={app.showOverlay}
          onToggleOverlay={() => app.setShowOverlay(!app.showOverlay)}
          running={app.running}
          runPct={app.runPct}
        />
      </div>

      <div style={{ width: 318, flexShrink: 0, display: "flex", flexDirection: "column", gap: 12, minHeight: 0, overflowY: "auto" }}>
        {app.mode === "op" ? (
          <OpModePanel app={app} />
        ) : (
          <React.Fragment>
            <RunControlPanel app={app} />
            <RecipeInfoPanel recipe={app.recipe} onOpenRecipe={app.openRecipePicker} />
          </React.Fragment>
        )}
      </div>
    </div>
  );
}

Object.assign(window, { RunScreen });
