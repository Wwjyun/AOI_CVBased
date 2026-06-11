// ============================================================
// AOI Console — 模擬資料（依實際 recipes/*.yaml 與 detectors/ 結構）
// ============================================================

const DETECTOR_DEFS = {
  "000": {
    display_name: "Binary contour area guard",
    zh: "二值化輪廓面積檢查",
    default_params: { threshold: 128, invert: false, min_area: 50, max_area: 200000, blur_size: 3 },
  },
  "001": {
    display_name: "Circle threshold NG detector",
    zh: "圓形閾值 NG 檢測",
    default_params: { threshold: 100, min_radius: 8, max_radius: 120, circularity: 0.78 },
  },
  "102": {
    display_name: "Scratch / thin line detector",
    zh: "刮痕／細線檢測",
    default_params: { canny_low: 40, canny_high: 120, min_length: 24, max_gap: 4, dilate_iter: 1 },
  },
  "305": {
    display_name: "Brightness / uniformity detector",
    zh: "亮度／均勻度檢測",
    default_params: { cell_size: 64, max_mean_diff: 18.0, max_std: 22.0 },
  },
  "777": {
    display_name: "Pattern match detector",
    zh: "Pattern 計數檢測",
    default_params: { template_path: "templates/pad.png", match_threshold: 0.8, expected_count: 24, tolerance: 0 },
  },
  "888": {
    display_name: "Texture / blur detector",
    zh: "紋理／模糊檢測",
    default_params: { lap_var_min: 60.0, local_std_min: 9.0, window: 32 },
  },
  "999": {
    display_name: "Dark / bright blob detector",
    zh: "暗點／亮點 blob 檢測",
    default_params: { threshold: 45, min_area: 20, max_area: 5000, blur_size: 3, invert: false, clahe_enabled: true },
  },
};

const RECIPES = [
  {
    file: "PRODUCT_A_AOI_01.yaml",
    recipe_name: "PRODUCT_A_AOI_01",
    product_id: "PRODUCT_A",
    machine_id: "AOI_01",
    version: "0.1.0",
    tile: { mode: "grid", width: 512, height: 512, overlap_x: 64, overlap_y: 64 },
    detectors: ["999"],
  },
  {
    file: "PRODUCT_A_PATTERN_MATCH_000_AOI_01.yaml",
    recipe_name: "PRODUCT_A_PATTERN_MATCH_000_AOI_01",
    product_id: "PRODUCT_A",
    machine_id: "AOI_01",
    version: "0.1.0",
    tile: { mode: "pattern_match", match_threshold: 0.8, max_count: 999, nms_threshold: 0.3, crop_padding: 8 },
    detectors: ["000", "999"],
  },
  {
    file: "PRODUCT_A_CONTOUR_TILE.yaml",
    recipe_name: "PRODUCT_A_CONTOUR_TILE",
    product_id: "PRODUCT_A",
    machine_id: "AOI_01",
    version: "0.2.1",
    tile: { mode: "contour", min_area: 4000, approx_epsilon: 0.01 },
    detectors: ["000", "102", "305"],
  },
];

const IMAGES = [
  { file: "PRODUCT_A_LOT0612_017.png", w: 4096, h: 3072, size: "11.4 MB" },
  { file: "PRODUCT_A_LOT0612_018.png", w: 4096, h: 3072, size: "11.2 MB" },
];

// 模擬缺陷（global bbox 以 0–1 正規化座標表示，渲染時乘上 viewer 尺寸）
const SIM_DEFECTS = [
  { id: 1, tile: "T012", detector: "999", type: "blob",    x: 0.31, y: 0.22, w: 0.030, h: 0.026, area: 412,  score: 0.93 },
  { id: 2, tile: "T012", detector: "999", type: "blob",    x: 0.36, y: 0.27, w: 0.018, h: 0.016, area: 105,  score: 0.81 },
  { id: 3, tile: "T031", detector: "102", type: "scratch", x: 0.58, y: 0.55, w: 0.110, h: 0.012, area: 887,  score: 0.88 },
  { id: 4, tile: "T044", detector: "999", type: "blob",    x: 0.76, y: 0.73, w: 0.024, h: 0.024, area: 298,  score: 0.95 },
  { id: 5, tile: "T044", detector: "305", type: "uniformity", x: 0.71, y: 0.66, w: 0.078, h: 0.072, area: 5210, score: 0.71 },
];

const RUN_STAGES = [
  { pct: 8,  msg: "載入圖片", ms: 350 },
  { pct: 22, msg: "切圖（tiling）", ms: 500 },
  { pct: 48, msg: "Detector 999 執行中", ms: 700 },
  { pct: 68, msg: "Detector 102 執行中", ms: 600 },
  { pct: 84, msg: "彙整結果（aggregate）", ms: 450 },
  { pct: 96, msg: "輸出 overlay / CSV / JSON", ms: 400 },
  { pct: 100, msg: "完成", ms: 250 },
];

const SIM_SUMMARY = { tile_count: 48, ng_count: 3, defect_count: SIM_DEFECTS.length };

const DEFECT_TYPE_LABEL = { blob: "Blob", scratch: "Scratch", uniformity: "Uniformity" };

const HISTORY_ROWS = [
  { time: "14:32:08", image: "PRODUCT_A_LOT0612_016.png", recipe: "PRODUCT_A_AOI_01", result: "PASS", defects: 0, dur: "2.8s" },
  { time: "14:28:51", image: "PRODUCT_A_LOT0612_015.png", recipe: "PRODUCT_A_AOI_01", result: "NG",   defects: 7, dur: "3.1s" },
  { time: "14:25:13", image: "PRODUCT_A_LOT0612_014.png", recipe: "PRODUCT_A_AOI_01", result: "PASS", defects: 0, dur: "2.9s" },
];

Object.assign(window, { DETECTOR_DEFS, RECIPES, IMAGES, SIM_DEFECTS, RUN_STAGES, SIM_SUMMARY, DEFECT_TYPE_LABEL, HISTORY_ROWS });
