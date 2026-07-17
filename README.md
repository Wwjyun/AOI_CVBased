# VisionFlow AOI

以 Python、OpenCV 與 PySide6 打造的配方驅動 AOI（自動光學檢測）系統。

VisionFlow AOI 不只是單一 Detector 範例，而是一套可實際延伸的檢測框架：同一條核心 Pipeline 可供 CLI、桌面 GUI、批次檢測與資料夾監控共用，並透過 YAML 配方調整切圖、Detector、判定及輸出行為。系統以 CPU 為正確性基準，另提供可選的 CUDA DLL 加速後端；未安裝 NVIDIA GPU 或 DLL 時，仍可完整使用 CPU 模式。

## 目前功能

- CLI 單張影像檢測。
- PySide6 桌面 GUI。
- YAML 配方載入、驗證、編輯與儲存。
- 固定網格、模板定位網格、輪廓及模板比對四種切圖方式。
- `401`、`401-1`、`401-2`、`900` 四個傳統電腦視覺 Detector。
- 單張檢測、批次資料夾檢測及新檔案監控。
- OP、Engineer、Admin 三種 GUI 操作模式。
- Overlay、NG 小圖、缺陷 CSV、矩陣 CSV、JSON 與輪替日誌。
- 批次統計 Dashboard 與散佈圖。
- PyInstaller Windows 執行檔打包。
- 一般 Windows CI，以及隔離在 RTX 3090 self-hosted runner 的 CUDA runtime workflow。
- 打包版非互動 smoke，涵蓋 CPU-only、缺少 DLL 時的安全 fallback 與 strict CUDA 失敗路徑。
- 可選 CUDA DLL、CPU fallback、效能觀測及 CPU/GPU 前處理抽象層。

目前仍待完成的重點包括：建立正式標註資料集、五份 production recipes 的完整 CPU/GPU 等價驗收、RTX 3090 實機編譯、長時間穩定度與效能測試，以及有 GPU 的打包版驗收。詳細進度以 [`Todo.md`](Todo.md) 為準。

## 設計目標

AOI 專案若將每種產品的規則硬寫在程式內，往往很快就難以維護。VisionFlow AOI 將可調整的檢測行為放進 YAML 配方，讓工程人員可在不改動核心 Pipeline 的情況下調整產品規格。

主要原則如下：

- GUI 與檢測核心分離，所有執行方式共用 `AOIPipeline`。
- Detector 參數可見、可調且可保存。
- 以影像、CSV、JSON、矩陣 CSV 與日誌保留追溯資料。
- 同時支援工程調機及產線 OP 工作流程。
- 新 Detector 可透過統一介面加入，不必修改 Pipeline。
- CPU-only 是完整支援的產品模式，也是結果正確性的基準。
- GPU 發生載入、初始化、kernel 或記憶體錯誤時，可依配方安全回退 CPU。

## 技術與環境

- Windows 10／11
- Python 3.13（CI、打包與目前部署環境的鎖定版本）
- OpenCV：影像處理與傳統 CV Detector
- NumPy：數值運算
- Pillow：大型影像載入與預覽轉換
- PyYAML：配方讀寫
- PySide6：桌面 GUI
- PyInstaller：Windows 打包
- CUDA Toolkit 與 NVIDIA GPU：僅 CUDA 加速功能需要

直接相依套件固定在 `requirements.txt`，完整 Windows transitive lock 位於 `requirements.lock.txt`；版本升級來源則保留在 `requirements.in`。CI、RTX runner 與打包環境一律安裝 lock：

```text
opencv-python==4.13.0.92
numpy==2.4.6
Pillow==12.2.0
PyYAML==6.0.3
PySide6==6.11.1
PyInstaller==6.21.0
hypothesis==6.156.6
```

## 快速開始

### 1. 建立環境並安裝套件

專案已預期使用根目錄下的 `env` 虛擬環境：

```powershell
cd <AOI_CVbased 專案目錄>
py -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.lock.txt
```

若 `env` 已存在，只需執行安裝指令。

### 2. 啟動 GUI

```powershell
.\env\Scripts\python.exe main.py --gui
```

### 3. 執行 CLI 單張檢測

```powershell
.\env\Scripts\python.exe main.py `
  --image C:\path\to\image.png `
  --recipe recipes\PRODUCT_A_FRAME_900_AOI_01.yaml `
  --output outputs
```

CLI 會將摘要輸出為 JSON。最終結果為 `PASS` 時結束碼是 `0`，為 `NG` 時是 `2`；配方、影像或執行階段錯誤則會回傳其他非零結束碼。

除錯與日誌選項：

```powershell
.\env\Scripts\python.exe main.py `
  --image C:\path\to\image.png `
  --recipe recipes\PRODUCT_A_FRAME_900_AOI_01.yaml `
  --output outputs `
  --debug `
  --log-level DEBUG `
  --log-dir outputs\logs
```

也可透過環境變數設定日誌：

```powershell
$env:AOI_LOG_LEVEL = 'DEBUG'
$env:AOI_LOG_DIR = 'outputs\logs'
```

## 專案結構

```text
AOI_CVbased/
|-- main.py                         # CLI／GUI 入口
|-- gui_launcher.py                 # 打包用 GUI 啟動器
|-- build_exe.ps1                   # PyInstaller 打包腳本
|-- VisionFlow AOI.spec             # PyInstaller 設定
|-- requirements.txt
|-- requirements.in / requirements.lock.txt
|-- AGENT.md                        # Codex／維護者工作規範
|-- Todo.md                         # 唯一專案工作清單
|-- .github/workflows/              # Windows CI 與 RTX 3090 runtime workflow
|-- core/
|   |-- pipeline.py                 # 檢測流程協調
|   |-- recipe_manager.py           # 配方載入與驗證
|   |-- recipe_builder.py           # GUI 配方建立
|   |-- image_loader.py             # 影像載入
|   |-- tiler.py                    # 四種切圖策略
|   |-- detector_manager.py         # Detector registry／factory
|   |-- preprocess_plan.py          # CPU／CUDA 共用前處理描述與 executor
|   |-- gpu_runtime.py              # CUDA DLL bridge、能力偵測與 fallback
|   |-- gpu_session.py              # batch／monitor 共用 GPU runtime/context
|   |-- aggregator.py               # Tile 與整張影像 PASS／NG 彙總
|   |-- result_mapper.py            # 區域座標映射至原圖座標
|   |-- result_compactor.py         # 長時間工作使用的結果壓縮
|   |-- reporter.py                 # PNG、CSV、JSON 報告
|   |-- performance.py              # 效能與 GPU 傳輸觀測
|   |-- batch_dashboard.py          # 批次與監控統計模型
|   |-- batch_processor.py          # 平行批次處理
|   `-- monitor_processor.py        # 資料夾監控
|-- detectors/
|   |-- base_detector.py            # Detector 共用介面
|   |-- detector_401.py
|   |-- detector_401_1.py
|   |-- detector_401_2.py
|   `-- detector_900.py
|-- gui/
|   |-- main_window.py
|   |-- workers.py                  # Qt 背景工作執行緒
|   |-- image_viewer.py
|   |-- screens/                    # Run、Results、Designer、Monitor、Dashboard
|   `-- widgets/                    # 共用 GUI 元件
|-- gpu/
|   |-- include/                    # 公開 C ABI 與內部 CUDA headers
|   |-- visionflow_cuda.cu          # CUDA kernels 與 DLL exports
|   |-- test_cuda_api.cu            # C++ smoke test
|   |-- preflight_cuda_build.py     # ABI/source/build manifest 靜態檢查
|   |-- production_manifest.example.yaml # 五份配方 PASS／NG 驗收清單範例
|   |-- validate_cuda_dll.py        # CPU／GPU 比對工具
|   `-- build_cuda_dll.ps1          # CUDA 編譯入口
|-- cuda_practice/                  # 獨立 CUDA 學習與裝置檢查範例
|-- design_handoff_aoi_gui/         # GUI 設計交接參考，不是 runtime dependency
|-- recipes/                        # 範例 YAML 配方
|-- tests/                          # 自動化測試
|-- outputs/                        # 正式執行輸出
`-- outputs_validation/             # 本機驗證輸出，不納入版本控制
```

## 系統流程

```text
影像 + YAML 配方
       |
       v
RecipeManager -> ImageLoader -> Tiler
                                  |
                                  v
                         DetectorManager
                                  |
                         逐 Tile 執行 Detector
                                  |
                   bbox_local -> bbox_global
                                  |
                                  v
                       Aggregator -> Reporter
                                  |
              Overlay / NG Tiles / CSV / JSON / Logs
```

GUI 不會複製另一套檢測邏輯，而是由 Qt worker 執行相同的 `AOIPipeline`，因此單張、批次及監控模式能維持一致的配方語意與輸出格式。

## YAML 配方

配方至少包含下列區段：

- `recipe_name`、`product_id`、`machine_id`、`version`
- `tile`：切圖方式及參數
- `decision`：整張影像的判定規則
- `detectors`：啟用的 Detector 與參數
- `output`：輸出開關
- `gpu`：可選的 CUDA 設定

最小範例：

```yaml
recipe_name: "PRODUCT_A_CIRCLE_401_1_AOI_01"
product_id: "PRODUCT_A"
machine_id: "AOI_01"
version: "0.1.0"

gpu:
  mode: auto  # auto=可回退、cpu=完全不載入 CUDA、cuda=CUDA 必須成功
  tiling: false
  display: false
  dll_path: "gpu/visionflow_cuda.dll"
  fallback_to_cpu: true
  queue_depth: 8  # batch/monitor throughput queue；單張 GUI 固定低延遲 depth=1

tile:
  mode: "grid"
  width: 512
  height: 512
  overlap_x: 64
  overlap_y: 64

decision:
  mode: "all_detectors_must_pass"
  important_detectors:
    - "401-1"
  max_ng_count: 0

detectors:
  "401-1":
    enabled: true
    use_gpu: false
    display_name: "401-1 adaptive circle contour detector"
    params:
      threshold_method: "adaptive_mean"
      max_value: 255
      invert: false
      blur_size: 45
      adaptive_block_size: 33
      adaptive_c: -2.0
      roi_inset_px: 100
      contour_mode: "list"
      morph_operation: "none"
      morph_kernel: 3
      morph_iterations: 1
      process_scale: 1.0
      min_area: 100
      max_area: 1000
      min_circularity: 0.70
      min_fill_ratio: 0.55
      max_fill_ratio: 1.20

output:
  save_overlay: true
  save_ng_tiles: true
  save_csv: true
  save_matrix_csv: true
  save_json: true
```

`decision.max_ng_count` 控制整張影像可容許的 NG Tile 數量。目前判定邏輯為：`ng_count <= max_ng_count` 時 `PASS`，否則為 `NG`。

## 切圖策略

每個 Tile 都會記錄 `tile_id`、位置、寬高、列／欄與模式專屬 metadata。Detector 在 Tile 區域內工作，`core/result_mapper.py` 再將 `bbox_local` 映射回原圖的 `bbox_global`。

### 固定網格 `grid`

```yaml
tile:
  mode: "grid"
  width: 512
  height: 512
  overlap_x: 64
  overlap_y: 64
```

適合均勻產品、全畫面掃描或不需要定位基準的檢測。

### 模板定位網格

```yaml
tile:
  mode: "grid"
  template_path: "path/to/template.png"
  search_x: 0
  search_y: 0
  search_w: 1200
  search_h: 1200
  offset_x: 10
  offset_y: 20
  rows: 8
  cols: 12
  roi_w: 100
  roi_h: 100
  gap_x: 12
  gap_y: 10
  match_threshold: 0.8
```

先在搜尋區域找出模板錨點，再依偏移、列欄數、ROI 大小及間距產生規則網格，適合有小幅位置漂移的重複工件。

### 輪廓切圖 `contour`

```yaml
tile:
  mode: "contour"
  threshold:
    method: "adaptive_mean"
    max_value: 255
    invert: false
    adaptive_block_size: 31
    adaptive_c: 5.0
    blur_size: 3
  shapes:
    enabled_shapes: ["rectangle", "circle", "polygon"]
    min_area: 100
    max_area: 0
    min_circularity: 0.75
    polygon_min_vertices: 3
    polygon_max_vertices: 99
    approx_epsilon_ratio: 0.02
    subpixel_enabled: true
    subpixel_window: 5
    crop_padding: 0
```

適合依可見零件輪廓擷取 ROI，或重複工件並非整齊排列的情境。

### 模板比對切圖 `pattern_match`

```yaml
tile:
  mode: "pattern_match"
  pattern_match:
    template_path: "path/to/template.png"
    match_threshold: 0.8
    max_count: 999
    nms_threshold: 0.3
    crop_padding: 0
    sort_row_tolerance: 20
    max_candidates: 20000
```

找出多個模板匹配位置，經局部峰值與 NMS 過濾後，由上而下、由左而右排序，適合重複視覺結構。

## Detector

所有 Detector 都繼承 `BaseDetector`，並輸出統一格式，包含 Detector ID、PASS／NG、分數、缺陷類型、區域座標、面積及 metadata。如此 Reporter、Aggregator 與 GUI 不需要知道個別演算法細節。

### `401`：負極旋轉矩形檢測

- 檔案：`detectors/detector_401.py`
- 用途：透過自適應閾值、形態學與旋轉矩形擬合偵測負極矩形 NG 區域。
- 主要參數：`roi_inset_px`、`blur_size`、`morph_operation`、`morph_kernel`、`morph_iterations`、`adaptive_block_size`、`adaptive_c`、`binary_inv`、`min_area`、`max_area`。
- 缺陷類型：`401_negative_rect_detected_ng`
- 範例配方：`recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml`

### `401-1`：自適應圓形輪廓檢測

- 檔案：`detectors/detector_401_1.py`
- 用途：以面積、圓度與填充比篩選圓形 NG 區域。
- 主要參數：`blur_size`、`adaptive_block_size`、`adaptive_c`、`roi_inset_px`、`process_scale`、`min_area`、`max_area`、`min_circularity`、`min_fill_ratio`、`max_fill_ratio`。
- 缺陷類型：`401_1_circle_detected_ng`
- 範例配方：`recipes/PRODUCT_A_CIRCLE_401_1_AOI_01.yaml`

### `401-2`：自適應白像素比例檢測

- 檔案：`detectors/detector_401_2.py`
- 用途：計算輪廓範圍內的白像素比例，超過 `white_pixel_ratio_threshold` 時判定 NG。
- 主要參數：`blur_size`、`adaptive_block_size`、`adaptive_c`、`roi_inset_px`、`min_area`、`max_area`、`white_pixel_ratio_threshold`。
- 預設白像素比例門檻：`0.625`
- 缺陷類型：`401_2_white_pixel_ratio_ng`
- 範例配方：`recipes/PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml`

### `900`：雙框間距檢測

- 檔案：`detectors/detector_900.py`
- 用途：找出外框與內框，檢查左、上、右、下四個邊距。
- 流程：外框全域閾值、內框自適應閾值、候選框尺寸過濾、內外框配對及最大邊距判定。
- 主要參數：`outer_threshold`、內外框目標寬高與容差、`inner_adaptive_block_size`、`inner_adaptive_c`、`max_edge_gap`、`roi_inset_px`。
- 缺陷類型：`900_frame_spacing_ng`
- 範例配方：`recipes/PRODUCT_A_FRAME_900_AOI_01.yaml`

Detector 900 的 NG Tile 會額外繪出內外框候選、被拒絕候選、間距輔助線與失敗原因，方便調整配方。

## GUI 使用方式

主視窗標題為 `VisionFlow AOI`，包含下列畫面：

- **Run**：載入影像與配方、執行單張或資料夾批次檢測、查看最近紀錄。
- **Monitor**：監控資料夾並逐張處理新加入且已穩定的影像。
- **Recipe Designer**：設定配方 metadata、切圖方式、Detector 開關與參數，並預覽 Tile。
- **Results**：查看最終結果、缺陷表格、縮圖及輸出路徑。
- **Batch Dashboard**：查看批次總量、PASS／NG、缺陷統計與 Tile 散佈圖。

### 操作模式

- **OP**：產線導向的限制模式，主要顯示監控工作流程。
- **Engineer**：工程調機模式，隱藏部分進階 Detector 參數。
- **Admin**：完整的配方與 Detector 參數權限。

這些模式是避免誤操作的 UI 分流，並不是帳號驗證或安全邊界。

### 單張檢測

1. 載入影像。
2. 載入配方。
3. 在設定抽屜確認輸出項目。
4. 執行檢測。
5. 查看 PASS／NG、Tile、NG 及缺陷數量與耗時。
6. 檢查 Overlay、缺陷表格與輸出檔案。

### 批次資料夾

1. 載入配方並選擇影像資料夾。
2. 選擇是否遞迴掃描子資料夾。
3. 啟動批次檢測。
4. 結果寫入 `outputs\batch\<timestamp>\`。
5. 在 Batch Dashboard 檢查整批摘要。

Worker 預設最多使用 4 個執行緒，並受 CPU 數與影像數限制；可用 `AOI_BATCH_WORKERS` 環境變數覆寫。記憶體內結果會壓縮以降低長時間執行的負擔，完整資料仍保留在 JSON 報告。

### 資料夾監控

1. 載入配方並選擇監控資料夾。
2. 可選擇處理後影像的移動資料夾。
3. 啟動監控；既有檔案會視為已看過。
4. 新影像通過檔案大小與修改時間的穩定檢查後，依序執行檢測。
5. 結果顯示在監控表格與散佈圖。

監控預設每秒輪詢一次，需連續通過 2 次穩定檢查；若設定移動資料夾，會保留子資料夾結構並處理同名衝突。

## 輸出內容

`core/reporter.py` 依配方寫入：

```text
outputs/
|-- overlay/
|-- ng_tiles/
|-- csv/
|-- matrix_csv/
|-- json/
`-- logs/
```

### Overlay PNG

- OK Tile 使用綠框、NG Tile 使用紅框。
- 缺陷框會繪製在原始影像座標。
- 若 metadata 包含圓形資訊，會同時繪製圓與 bbox。

### NG Tiles

- 只保存 NG Tile 裁切影像。
- 缺陷框以 Tile 區域座標繪製。
- Detector 900 額外提供內外框及邊距除錯標記。
- 每張 PNG 旁會產生同名 JSON dataset sidecar，記錄 source/effective recipe hash、build commit、detector 有效參數、局部／全域座標，以及 `pending` 人工複判欄位。

### 缺陷 CSV

包含影像、配方、機台、產品、最終結果、Detector、缺陷類型、全域／區域 bbox、Tile ID、分數與面積。檔案使用帶 BOM 的 UTF-8（`utf-8-sig`），方便 Excel 直接開啟。

### 矩陣 CSV

將具列欄資訊的 Tile NG 狀態轉成矩陣，欄位為 `c1`、`c2` 等，NG 儲存格以勾號標示，適合對照產品的實體排列。

### JSON

JSON 是最完整的追溯格式，包含影像與配方 metadata、最終結果、耗時、統計、輸出路徑、Tile、Detector、缺陷、區域／全域座標及 Detector 專屬 metadata。`provenance` 同時保存原始 YAML SHA-256、套用 runtime overrides 後的 canonical SHA-256、有效 detector params，以及 Git／PyInstaller build commit 與 dirty 狀態。

### 日誌

- CLI 預設：`<output>\logs\aoi.log`
- GUI 預設：`outputs\logs\aoi.log`

日誌採輪替檔案，涵蓋 Pipeline、Reporter、批次、監控、GUI workers 與主程式。

## 可選 CUDA 加速

`gpu/visionflow_cuda.dll` 是可選後端。未啟用 GPU 時不會載入 DLL；啟用但 DLL、CUDA 裝置或運算不可用時，會依 `fallback_to_cpu` 回退整個 Detector 至 CPU，或明確回報錯誤。系統不會把失敗前的部分 GPU 中間結果和 CPU 後續流程混用。

```yaml
gpu:
  mode: auto
  tiling: false
  display: false
  dll_path: "gpu/visionflow_cuda.dll"
  fallback_to_cpu: true
  queue_depth: 8

detectors:
  "401-2":
    enabled: true
    use_gpu: true
```

前處理由 backend-neutral `PreprocessPlan` 描述：

- `CpuPreprocessExecutor` 定義 OpenCV 正確性語意。
- `CudaPreprocessExecutor` 依 DLL 能力優先選擇 versioned generic native plan，再選相容的 fused adapter、舊版通用 primitive 或 CPU fallback。
- Generic native linear plan 支援 Gray、兩軸不放大的單通道 Resize(area)、Gaussian、Threshold、Adaptive Mean 與 Morphology；Resize 放大或混合軸縮放仍明確 fallback。整份 plan capability 通過後只做一次 H2D、連續 kernels 與一次必要 D2H。
- Generic native DAG plan 支援拓撲排序的分支與多輸出；Detector 900 共用一次 device gray，單次上傳後只下載 outer/inner masks。
- Detector 401-2 已有一次呼叫完成灰階、Gaussian 與 Adaptive Mean 的 persistent context 相容路徑。
- Persistent context 現在持有 non-blocking CUDA stream、grow-only scratch 與 morphology ping-pong buffers；plan 內的中間結果不回傳 CPU。
- Batch 與 monitor 會透過 `GpuExecutionSession` 跨多張影像共用同一個 `GpuRuntime`/CUDA context，結束工作後才統一釋放。
- 舊版 DLL 缺少新 exports 時仍保留既有路徑或 CPU fallback。
- GPU mode 統一為 `auto`、`cpu`、`cuda`：`auto` 依設定嘗試並可回退，`cpu` 不載入 CUDA，`cuda` 禁止隱性 CPU fallback；執行結果與 GUI 顯示的是實際 backend。

目前 CUDA 原始碼包含 separable Gaussian、constant weights、64-bit integral Adaptive Mean Threshold、persistent context 與 grow-only buffers。這些功能仍需在目標 RTX 3090（`sm_86`）完成正式編譯、五份配方等價、效能、VRAM 與壓力驗收後，才能視為 production-ready 或預設啟用。

### RTX 3090 編譯與驗證

安裝 NVIDIA Driver、CUDA Toolkit、Visual Studio 2022 C++ Build Tools 與 Windows SDK 後，在 x64 Native Tools PowerShell 執行：

```powershell
.\gpu\build_cuda_dll.ps1 -Architecture sm_86
```

執行 C++ smoke、structured primitive matrix 與 benchmark：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests
```

加入真實影像與配方進行 AOI CPU／GPU 比對：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests `
  -Image C:\AOI_TEST\sample.png `
  -Recipe .\recipes\PRODUCT_A_AOI_01.yaml
```

CUDA 詳細架構及操作請參考 [`gpu/README.md`](gpu/README.md)，完整實機驗收矩陣請參考 [`Todo.md`](Todo.md)。

## 獨立匯出工具

```powershell
.\env\Scripts\python.exe export_scatter_plots.py
.\env\Scripts\python.exe export_matrix_summary.py
```

- `export_scatter_plots.py`：從 JSON／CSV 報告匯出散佈圖摘要。
- `export_matrix_summary.py`：整合多個矩陣 CSV 為彙總報表。

兩者獨立於主 Pipeline，讓後處理工具可自行演進。

## 建立 Windows 執行檔

```powershell
.\build_exe.ps1
```

輸出位置：

```text
dist\VisionFlow AOI\VisionFlow AOI.exe
```

發佈時必須複製或壓縮整個 `dist\VisionFlow AOI` 資料夾，不能只取出 `.exe`，因為執行檔需要相鄰的 `_internal` runtime 目錄。

打包後可用非互動 smoke 模式驗證 bundled recipes 與 MainWindow 啟動；成功時 exit code 為 `0`：

```powershell
Start-Process -FilePath '.\dist\VisionFlow AOI\VisionFlow AOI.exe' -ArgumentList '--smoke-test' -WindowStyle Hidden -Wait -PassThru
```

此 smoke mode 會先驗證 bundle 內的配方與 Qt 視窗，再於打包程式內執行小型完整 Pipeline 矩陣：CPU-only、缺少 CUDA DLL 且允許 CPU fallback，以及缺少 CUDA DLL 的 strict CUDA。結束碼 `0` 代表 fallback 結果與 CPU 一致，且 strict CUDA 已如預期明確失敗。

`build_exe.ps1` 使用受版控的 `VisionFlow AOI.spec`，不會在每次建置時覆寫 CUDA DLL 的條件式收錄規則；建置時會將 commit/dirty provenance 嵌入 bundle。

發行檔命名格式：

```text
VisionFlow-AOI-vX.Y.Z-windows-x64.zip
```

## 驗證

所有 Python 指令應使用專案虛擬環境。修改完成後的基本驗證：

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
.\env\Scripts\python.exe -m compileall main.py gui_launcher.py core detectors gui gpu
.\env\Scripts\python.exe gpu\preflight_cuda_build.py
git diff --check
```

`.github/workflows/windows-ci.yml` 會在一般 Windows runner 執行上述 Python 驗證、合成影像 CLI smoke 與 GUI offscreen smoke；`.github/workflows/rtx3090-validation.yml` 只在受信任的 `self-hosted, Windows, X64, gpu, rtx3090` runner 執行 CUDA 編譯、原生 ABI smoke、CPU/GPU 等價、benchmark、壓測與可選 Nsight capture。RTX benchmark 第一次成功會建立 Actions cache baseline，後續任一 GPU P95 退化超過 15% 即失敗；hosted heartbeat 超過 48 小時沒有成功紀錄也會失敗。`weekly-packaging.yml` 每週以 Python 3.13 與 lock 重建 EXE 並跑 packaged smoke。RTX workflow 沒有 runner 接單或仍在 queued，不代表 CUDA runtime 已通過。

GUI offscreen smoke：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\env\Scripts\python.exe -c "from pathlib import Path; from PySide6.QtWidgets import QApplication; from gui.main_window import MainWindow; app=QApplication([]); w=MainWindow(); w.recipe_panel.load_recipe(Path('recipes/PRODUCT_A_AOI_01.yaml')); print(w.windowTitle(), w.recipe_panel.detector_list.count())"
```

正式發行前還應完成：

- 已知 PASS 與 NG 影像的 CLI 檢測。
- Overlay、NG Tiles、CSV、矩陣 CSV 與 JSON 檢查。
- 至少兩張影像的批次處理。
- 監控模式新檔案處理與移動。
- 打包版啟動及單張配方執行。
- 有／無 NVIDIA GPU 電腦的啟動與 fallback 測試。
- RTX 3090 的 CPU／GPU 等價、效能及長時間穩定性測試。

## 目前限制

- Detector 目前以傳統 CV 規則為主，效果高度依賴光源、治具穩定度及配方門檻。
- 尚未建立正式且具預期 PASS／NG 標籤的驗證資料集，因此不可將範例結果直接視為量產良率證明。
- GUI 模式僅是工作流程限制，不是資安權限系統。
- `--debug` 已存在，但尚未為所有 Detector 提供完整的中間影像輸出。
- GPU 預設啟用前仍需完成 `Todo.md` 中的 RTX 3090 驗收門檻。

## 延伸開發

- 新增 Detector 時，實作應放在 `detectors/` 並透過 `DetectorManager` 註冊。
- 可重用的前處理應使用或擴充 `PreprocessPlan` typed operators，不要為每個 Detector 建立獨立 CUDA workflow。
- Reporter、Aggregator、GUI 與 Detector 之間應維持統一結果格式。
- 所有 GPU 功能都必須保留 CPU 等價語意及可測試的 fallback。
- 未來可加入 YOLO、RT-DETR 或 segmentation plugin，但仍應沿用既有 Detector 輸出格式、GPU 排程與 VRAM 管理原則。

初次閱讀程式碼時，建議依序查看：

1. `core/pipeline.py`：完整檢測協調流程。
2. `core/tiler.py`：ROI 與 Tile 產生方式。
3. `detectors/`：各檢測演算法。
4. `core/preprocess_plan.py` 與 `core/gpu_runtime.py`：CPU／CUDA 前處理架構。
5. `core/reporter.py`：追溯輸出。
6. `gui/main_window.py` 與 `gui/screens/`：桌面應用程式。
