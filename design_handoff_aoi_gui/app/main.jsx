// ============================================================
// AOI Console — app shell（rail / topbar / screens / settings / tweaks）
// ============================================================

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#0d9488",
  "density": "regular",
  "simResult": "NG"
}/*EDITMODE-END*/;

const ACCENT_MAP = {
  "#0d9488": { strong: "#0b7d73", soft: "#e3f3f1", softer: "#f0f9f8", text: "#0a6b62" },
  "#2563eb": { strong: "#1e50c4", soft: "#e5edfc", softer: "#f2f6fd", text: "#1c4fbe" },
  "#475569": { strong: "#374357", soft: "#e8ecf1", softer: "#f3f5f8", text: "#3c4a5e" },
};

const NAV = [
  { id: "run", label: "檢測執行", icon: IcPlay },
  { id: "designer", label: "Recipe 設計", icon: IcDesigner, engOnly: true },
  { id: "results", label: "檢測結果", icon: IcTable },
];

const SCREEN_TITLE = { run: "檢測執行", designer: "Recipe 設計", results: "檢測結果" };

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [screen, setScreen] = React.useState("run");
  const [mode, setMode] = React.useState("eng");

  const [image, setImage] = React.useState(null);
  const [imageLoading, setImageLoading] = React.useState(false);
  const [recipe, setRecipe] = React.useState(null);
  const [recipes, setRecipes] = React.useState(RECIPES);

  const [running, setRunning] = React.useState(false);
  const [runPct, setRunPct] = React.useState(0);
  const [runMsg, setRunMsg] = React.useState("Ready");
  const [result, setResult] = React.useState(null);
  const [selectedDefect, setSelectedDefect] = React.useState(null);
  const [showOverlay, setShowOverlay] = React.useState(true);
  const [history, setHistory] = React.useState(HISTORY_ROWS);

  const [picker, setPicker] = React.useState(null); // "image" | "recipe" | "settings" | null
  const [statusMsg, setStatusMsg] = React.useState("就緒");
  const [outputDir, setOutputDir] = React.useState("outputs");
  const [outputOpts, setOutputOpts] = React.useState({ overlay: true, ng_tiles: true, csv: true, json: true });

  const timersRef = React.useRef([]);
  React.useEffect(() => () => timersRef.current.forEach(clearTimeout), []);

  // accent tweak → CSS vars
  React.useEffect(() => {
    const a = ACCENT_MAP[t.accent] || ACCENT_MAP["#0d9488"];
    const root = document.documentElement;
    root.style.setProperty("--accent", t.accent);
    root.style.setProperty("--accent-strong", a.strong);
    root.style.setProperty("--accent-soft", a.soft);
    root.style.setProperty("--accent-softer", a.softer);
    root.style.setProperty("--accent-text", a.text);
  }, [t.accent]);

  React.useEffect(() => {
    document.documentElement.setAttribute("data-density", t.density);
  }, [t.density]);

  // OP 模式下離開 designer
  React.useEffect(() => {
    if (mode === "op" && screen === "designer") setScreen("run");
  }, [mode, screen]);

  const loadImage = (img) => {
    setPicker(null);
    setImageLoading(true);
    setStatusMsg(`影像載入中：${img.file}`);
    timersRef.current.push(setTimeout(() => {
      setImage(img);
      setImageLoading(false);
      setResult(null);
      setSelectedDefect(null);
      setStatusMsg(`影像已載入：${img.file}（${img.w} × ${img.h}, ${img.size}）`);
    }, 700));
  };

  const loadRecipe = (r) => {
    setPicker(null);
    setRecipe(r);
    setStatusMsg(`Recipe 已載入：recipes/${r.file}`);
  };

  const saveDesignedRecipe = (r) => {
    setRecipes((prev) => [r, ...prev.filter((x) => x.file !== r.file)]);
    setRecipe(r);
    setStatusMsg(`設計 Recipe 已儲存並載入：recipes/${r.file}`);
  };

  const startRun = () => {
    if (running || !image || !recipe) return;
    setRunning(true);
    setResult(null);
    setSelectedDefect(null);
    setRunPct(0);
    let acc = 0;
    RUN_STAGES.forEach((stage) => {
      acc += stage.ms;
      timersRef.current.push(setTimeout(() => {
        setRunPct(stage.pct);
        setRunMsg(stage.msg);
        setStatusMsg(`${stage.msg}（${stage.pct}%）`);
      }, acc));
    });
    timersRef.current.push(setTimeout(() => {
      const pass = t.simResult === "PASS";
      const defects = pass ? [] : SIM_DEFECTS;
      const res = {
        final: pass ? "PASS" : "NG",
        summary: pass ? { tile_count: 48, ng_count: 0, defect_count: 0 } : SIM_SUMMARY,
        defects,
        dur: "3.2s",
      };
      setResult(res);
      setRunning(false);
      setStatusMsg(`檢測完成：${res.final}`);
      setHistory((h) => [{
        time: new Date().toTimeString().slice(0, 8),
        image: image.file, recipe: recipe.recipe_name,
        result: res.final, defects: defects.length, dur: res.dur,
      }, ...h].slice(0, 6));
    }, acc + 200));
  };

  const app = {
    screen, setScreen, mode,
    image, imageLoaded: !!image, recipe, recipes,
    running, runPct, runMsg, result,
    selectedDefect, setSelectedDefect,
    showOverlay, setShowOverlay,
    history,
    startRun,
    goResults: () => setScreen("results"),
    openRecipePicker: () => setPicker("recipe"),
    saveDesignedRecipe,
  };

  const visibleNav = NAV.filter((n) => !(n.engOnly && mode === "op"));

  return (
    <div className="shell">
      {/* ---- icon rail ---- */}
      <nav className="rail">
        <div className="rail-logo" title="AOI 視覺檢測系統">A·I</div>
        {visibleNav.map((n) => {
          const Ic = n.icon;
          return (
            <button
              key={n.id}
              className={"rail-btn" + (screen === n.id ? " active" : "")}
              onClick={() => setScreen(n.id)}
            >
              <Ic size={19} />
              <span className="rail-tip">{n.label}</span>
            </button>
          );
        })}
        <div className="rail-spacer"></div>
        <button className={"rail-btn" + (picker === "settings" ? " active" : "")} onClick={() => setPicker("settings")}>
          <IcGear size={19} />
          <span className="rail-tip">設定</span>
        </button>
      </nav>

      <div className="main-col">
        {/* ---- top bar ---- */}
        <header className="topbar">
          <span className="topbar-title">{SCREEN_TITLE[screen]}</span>
          <div className="topbar-divider"></div>

          <Chip
            icon={imageLoading ? <span className="spinner"></span> : <IcImage size={14} />}
            label="影像" empty={!image}
            value={imageLoading ? "載入中…" : image ? image.file : "點擊載入"}
            onClick={() => !running && setPicker("image")}
          />
          <Chip
            icon={<IcRecipe size={14} />}
            label="Recipe" empty={!recipe}
            value={recipe ? recipe.recipe_name : "點擊載入"}
            onClick={() => !running && setPicker("recipe")}
          />

          {running && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, width: 180 }}>
              <ProgressBar pct={runPct} />
              <span className="mono" style={{ color: "var(--text-2)", flexShrink: 0 }}>{runPct}%</span>
            </div>
          )}

          <div style={{ flex: 1 }}></div>

          <Segmented
            value={mode}
            onChange={setMode}
            options={[{ value: "op", label: "操作員 OP" }, { value: "eng", label: "工程師" }]}
          />
        </header>

        {/* ---- screen ---- */}
        <main style={{ flex: 1, minHeight: 0, padding: 12 }} data-screen-label={SCREEN_TITLE[screen]}>
          {screen === "run" && <RunScreen app={app} />}
          {screen === "designer" && <DesignerScreen app={app} />}
          {screen === "results" && <ResultsScreen app={app} />}
        </main>

        {/* ---- status bar ---- */}
        <footer style={{
          height: 26, flexShrink: 0,
          background: "var(--surface)", borderTop: "1px solid var(--border)",
          display: "flex", alignItems: "center", gap: 14, padding: "0 14px",
          fontSize: 11, color: "var(--text-3)", fontFamily: "var(--font-mono)",
        }}>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{statusMsg}</span>
          <span style={{ marginLeft: "auto", flexShrink: 0 }}>AOI_01 · {mode === "op" ? "操作員模式" : "工程師模式"}</span>
        </footer>
      </div>

      {/* ---- pickers & settings ---- */}
      {picker === "image" && (
        <Drawer title="載入檢測影像" onClose={() => setPicker(null)}>
          <div style={{ fontSize: "var(--fs-small)", color: "var(--text-3)", marginBottom: 10 }}>images/（模擬檔案瀏覽）</div>
          <div className="panel" style={{ borderRadius: "var(--r-md)" }}>
            {IMAGES.map((img) => (
              <div key={img.file} className="row-item" onClick={() => loadImage(img)}>
                <IcImage size={15} style={{ color: "var(--text-3)", flexShrink: 0 }} />
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{img.file}</div>
                  <div style={{ fontSize: 11, color: "var(--text-3)" }}>{img.w} × {img.h} · {img.size}</div>
                </div>
                <IcChevronR size={14} style={{ color: "var(--text-3)" }} />
              </div>
            ))}
          </div>
        </Drawer>
      )}

      {picker === "recipe" && (
        <Drawer title="載入 Recipe" onClose={() => setPicker(null)}>
          <div style={{ fontSize: "var(--fs-small)", color: "var(--text-3)", marginBottom: 10 }}>recipes/*.yaml</div>
          <div className="panel" style={{ borderRadius: "var(--r-md)" }}>
            {recipes.map((r) => (
              <div key={r.file} className="row-item" onClick={() => loadRecipe(r)}>
                <IcRecipe size={15} style={{ color: "var(--text-3)", flexShrink: 0 }} />
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.file}</div>
                  <div style={{ fontSize: 11, color: "var(--text-3)" }}>
                    {r.tile.mode} · detectors: {r.detectors.join(", ")} · v{r.version}
                  </div>
                </div>
                {recipe && recipe.file === r.file
                  ? <Badge kind="accent"><IcCheck size={11} strokeWidth={2.5} />使用中</Badge>
                  : <IcChevronR size={14} style={{ color: "var(--text-3)" }} />}
              </div>
            ))}
          </div>
        </Drawer>
      )}

      {picker === "settings" && (
        <Drawer title="設定" onClose={() => setPicker(null)}>
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div>
              <div className="panel-title" style={{ marginBottom: 10 }}>輸出</div>
              <FormGrid>
                <FRow label="輸出目錄">
                  <div style={{ display: "flex", gap: 6 }}>
                    <TextField mono value={outputDir} onChange={(e) => setOutputDir(e.target.value)} />
                    <Btn variant="secondary" size="sm" style={{ height: "var(--row-h)" }} icon={<IcFolder size={13} />}>瀏覽</Btn>
                  </div>
                </FRow>
                {[
                  ["overlay", "儲存 overlay 影像"],
                  ["ng_tiles", "儲存 NG tiles"],
                  ["csv", "輸出 CSV 報表"],
                  ["json", "輸出 JSON 報表"],
                ].map(([k, label]) => (
                  <FRow key={k} label={label}>
                    <Toggle value={outputOpts[k]} onChange={(v) => setOutputOpts({ ...outputOpts, [k]: v })} />
                  </FRow>
                ))}
              </FormGrid>
            </div>
            <div>
              <div className="panel-title" style={{ marginBottom: 10 }}>機台</div>
              <FormGrid>
                <FRow label="Machine ID"><TextField mono value="AOI_01" readOnly /></FRow>
                <FRow label="Pipeline 版本"><TextField mono value="0.4.2 (MVP)" readOnly /></FRow>
              </FormGrid>
            </div>
          </div>
        </Drawer>
      )}

      {/* ---- Tweaks ---- */}
      <TweaksPanel>
        <TweakSection label="視覺" />
        <TweakColor label="主色 Accent" value={t.accent}
          options={["#0d9488", "#2563eb", "#475569"]}
          onChange={(v) => setTweak("accent", v)} />
        <TweakRadio label="密度" value={t.density}
          options={["regular", "compact"]}
          onChange={(v) => setTweak("density", v)} />
        <TweakSection label="模擬" />
        <TweakRadio label="檢測結果" value={t.simResult}
          options={["NG", "PASS"]}
          onChange={(v) => setTweak("simResult", v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
