# VisionFlow AOI 統一開發清單

本文件是專案唯一的工作清單，涵蓋 CPU、GPU、CUDA、Detector、GUI、打包、CI 與實機驗收。完成程式修改時必須同步更新對應 checkbox；不得再建立分散的 CPU/GPU Todo 文件。

## 開發原則

- CPU 路徑是正確性基準，也是無 NVIDIA GPU、DLL 載入失敗、CUDA error 或顯存不足時的 fallback。
- GPU 最佳化不得改變 recipe 語意、PASS/NG、座標與輸出格式；允許差異必須先定義容差並加入測試。
- 不追求所有工作 GPU 化。YAML、彙總、少量 contour 幾何、GUI 控制、CSV/JSON、PNG 編碼與磁碟 I/O 預設留在 CPU。
- Detector 不得各自建立一套 CUDA workflow。Detector 只宣告 backend-neutral `PreprocessPlan`，由 CPU/CUDA executor 執行共用 operators。
- GPU 路徑應盡量一次 upload、連續執行多個 operators、最後只 download 必要 mask 或統計值。
- 新功能必須保持 OOP 邊界、CPU-only 可啟動、舊 DLL 相容與完整 detector CPU fallback。
- 只有 RTX 3090 實測通過數值等價、穩定性與端到端效能門檻的功能，才能預設啟用 GPU。

## 目前狀態摘要

- [x] CUDA DLL 已在 RTX 3090 編譯，並在另一台電腦確認可載入及顯示 CUDA active。
- [x] 已確認 CUDA active 不代表整條 AOI pipeline 都在 GPU；首次跨機測試端到端沒有加速。
- [x] 已有 CPU-only、缺 DLL fallback、GPU 呼叫統計及 detector 整體 CPU 重跑機制。
- [x] Gaussian 已改 separable kernels；Adaptive Mean 已改 64-bit integral image。
- [x] 已有 persistent context、grow-only buffers 與 401-2 fused preprocessing 原型。
- [x] 已建立通用 `PreprocessPlan`、typed operators、CPU/CUDA executors，401-2 已完成第一階段遷移。
- [ ] 目前開發機缺少可用的 `nvcc`/CMake，新增 CUDA 原始碼仍需在 RTX 3090 重新編譯與實測。
- [ ] 尚未完成固定 production 測試集、五個 recipes 全流程等價、長時間壓測與可信的 CPU/GPU benchmark。

## P0：正確性、CPU 基準與觀測能力

### Pipeline 與 profiler

- [x] 記錄 recipe setup、image load、tiling、各 detector、aggregation、reporting 與 end-to-end host wall time。
- [x] Reporter 分別記錄 overlay、NG tiles、CSV、matrix CSV 與 JSON 耗時。
- [x] 記錄 DLL load、同步呼叫、lock wait、估算 H2D/D2H bytes、round trips 與各 primitive 呼叫統計。
- [x] 加入 GUI 顯示、QImage/QPixmap 轉換與使用者實際等待時間計時。
- [x] DLL 加入 CUDA event，拆分 context、allocation、H2D、device copy、kernel、synchronize、D2H 與 free；RTX 數值驗證仍列在實機驗收清單。
- [x] benchmark JSON 保存 CPU、GPU、RAM、Driver、recipe、影像資訊與 commit hash；Toolkit 另由 runner environment artifact 保存。
- [ ] 在 RTX 3090 固定 production 測試集執行並建立可重現 baseline。（workflow_dispatch 已支援可選 production manifest；待真實樣本與 runner）
- [x] benchmark 分開記錄 cold、warm-up 次數、純檢測與既有 pipeline/report 端到端數據。
- [x] benchmark 記錄平均、median、P95、process CPU%、GPU utilization、VRAM、溫度與功耗快照。

### CPU 與 fallback 正確性

- [x] 缺少 DLL 時，CPU fallback 與純 CPU 的 PASS/NG、tiles、defects、bbox 與 metadata 完整一致。
- [x] fused GPU 呼叫失敗時不採用部分結果，整個 detector 重新從 CPU preprocess 開始執行。
- [x] 建立固定 random seed 合成測例：BGR、gray、全黑、全白、棋盤格與邊界像素。
- [ ] 補入固定真實 AOI 影像測例；manifest schema、路徑/標籤/coverage 驗證已完成，待取得可追蹤的生產樣本後執行。
- [x] 覆蓋奇數尺寸、極小圖、4K、non-contiguous stride、1/3 channels 與不同 ROI 尺寸。
- [ ] 五個 production recipes 各準備至少一張 PASS 與一張 NG 樣本；`gpu/production_manifest.example.yaml` 已固定所需 10 個 case，影像待提供。
- [ ] 實機注入 kernel error、CUDA 初始化失敗與 OOM，確認 fallback 後無 stale pointer 或錯誤中間結果。（loader tests 已覆蓋 ABI mismatch、無 device、context init failure；fake execution/OOM recovery 已完成，實機注入待 RTX）
- [x] `fallback_to_cpu: false` 且 CUDA DLL 不可用時必須明確失敗，不可回報假的 GPU success。

## P1：共用 Preprocess Plan 架構

### Python/OOP execution layer

- [x] 建立 immutable `PreprocessPlan`。
- [x] 建立 typed operators：Gray、Resize、Gaussian、Threshold、AdaptiveMean、Morphology。
- [x] 建立 `CpuPreprocessExecutor`，OpenCV 結果作為 fallback 與等價基準。
- [x] 建立 `CudaPreprocessExecutor`，支援 stateless primitives 與既有 401-2 fused compatibility adapter。
- [x] `BaseDetector.execute_preprocess_plan()` 統一選擇 executor。
- [x] 401-2 改為宣告 Gray → Gaussian → AdaptiveMean，不再直接呼叫 CUDA export。
- [x] CUDA 尚不能保持 `INTER_AREA` 語意時拒絕執行 Resize(area)，不可靜默改用 nearest-neighbor。
- [x] 將 plan 建立移出每次 tile 熱路徑，依 detector params、dtype 與 shape cache immutable plan，並以 bounded LRU 避免無限成長。
- [x] 加入 versioned operator/plan signature、輸入輸出 uint8 型別、channel、shape、順序與 operator 參數 validation。
- [x] 加入 versioned capability report，清楚列出 plan 為何走 fused、primitive、CPU 或 fallback，並寫入 detector execution metadata。
- [x] capability preflight 判定整份 plan/DAG 不支援 CUDA 時，在任何 primitive 執行前直接走完整 CPU fallback；關閉 fallback 時明確失敗。

### Generic native plan ABI

- [x] 定義 versioned C structs：operator kind、input node、參數、output node；不得包含 detector ID/name。
- [x] 新增 optional `vf_plan_create/execute/destroy` exports；ABI v1 primitives 與 401-2 adapter 保持相容。
- [x] plan create 階段驗證 operators、channel、shape、參數與輸出；execute 階段只處理資料。
- [x] 將 Gray、Gaussian、AdaptiveMean、Threshold、Morphology 接入通用 native executor。
- [x] 通用 native plan 達成一次 H2D、連續 kernels、最後一次必要 D2H。
- [x] 加入 plan capability query；任一 operator 不支援時整份 plan CPU fallback，避免反覆 CPU/GPU 傳輸。
- [ ] 實作與 OpenCV 等價的 `INTER_AREA` resize 後，才開放 CUDA Resize(area)。（native linear source 已加入兩軸不放大的單通道 `VF_PLAN_RESIZE_AREA`、動態 output shape 與一次 H2D/D2H routing；CPU 模擬 structured downscale 與 OpenCV 完全一致，實際 CUDA 像素容差仍待 RTX 編譯驗收）
- [x] Python/CPU plan 擴充 topologically ordered DAG/multi-output，支援一份 gray 產生多張 masks。
- [x] CUDA/native plan 擴充 DAG/multi-output，讓 device gray 直接產生多張 masks。

## P2：CUDA kernels 與資源生命週期

### 已完成的核心 kernels

- [x] Gaussian 使用 horizontal/vertical separable kernels 與 float 中間 buffer。
- [x] Gaussian weights 使用 constant memory。
- [x] Adaptive Mean 使用 replicate-border 64-bit integral image，視窗查詢為 O(1)。
- [x] Integral image 使用 row scan、transpose、第二次 row scan，並檢查 allocation overflow。
- [x] 驗證工具已加入 Gaussian、Adaptive Mean、401-2 fused 與 4K benchmark 案例。
- [ ] Gaussian 加入 shared-memory tile/halo，實測 kernel 45 收益與限制。
- [x] CUDA event 分別量測 Adaptive Mean integral/kernel、Gaussian passes 與 threshold kernel；待 RTX runner 回收實測數值。

### Persistent context 與 buffers

- [x] 保留 ABI v1 host-pointer primitives，使用 optional export probe 相容舊 DLL。
- [x] 新增 `vf_context_create/destroy/stats`。
- [x] context 擁有 grow-only uint8、float Gaussian 與 64-bit integral buffers。
- [x] 相同或較小尺寸的 401-2 fused 呼叫不再重複 `cudaMalloc/cudaFree`。
- [x] `GpuRuntime` 提供 `close()`、context manager、destructor 與 `RLock` 序列化。
- [x] 將 CUDA stream、morphology ping-pong 與所有 plan scratch 納入同一 context。
- [x] monitor/batch 跨多張影像重用同一個長生命週期 `GpuRuntime`/context。
- [ ] 測試尺寸增減、channel 切換、參數改變、CUDA error/OOM 後的重用與釋放。（validator 已覆蓋 shape grow/shrink、1/3 channel、參數切換與 warm allocation plateau；fake DLL 已覆蓋 execution error recovery、ROI batch OOM 降批，source contract 固定 allocation-before-free；真實 CUDA error/OOM 仍待 RTX）
- [ ] 評估 `cudaMallocAsync`/memory pool；只有相容且實測有收益時採用。

### Morphology

- [ ] 量測 detector 401 多 iterations 的 morphology 占比。（native morphology CUDA event 與 close iterations 1/2/4/8 benchmark 已完成；實際占比待 RTX runner）
- [ ] 評估矩形 kernel 的 horizontal/vertical separable min/max filter。
- [x] 多 iterations 使用 device ping-pong buffers，中間不得回傳 CPU。
- [ ] 小 kernel/少 iterations 建立 CPU/GPU crossover 規則。（validator 已輸出各 iterations 含傳輸 CPU/GPU median/P95/speedup；production threshold 待 RTX 數據）

## P3：Detector 遷移與 CPU/GPU 邊界

- [x] 401-1 遷移到 cached 共用 plan：Gray → Resize(area) → Gaussian → AdaptiveMean → Morphology；CUDA 無法保持 area 語意時整個 detector CPU fallback。
- [x] 401 遷移到 cached 共用 plan，保留 BGR Gaussian → Morphology → Gray → AdaptiveMean、threshold 與 contour 語意。
- [x] 401-2 preprocessing 已遷移到共用 plan，並保留 fused/legacy/CPU 路徑。
- [x] 900 遷移成 cached CPU DAG plan，共用一次 gray 產生 outer global 與 inner adaptive masks。
- [x] 900 DAG 接上 CUDA/native executor，共用 device gray 並只下載必要 masks。
- [x] 401/401-1/401-2 的 `findContours` 與少量幾何分析暫留 CPU，只下載 binary mask。
- [x] 401-2 contour mask 改為局部 bbox mask，避免每個 contour 配置整張 ROI mask。
- [ ] 評估 401-2 white-pixel reduction 移至 GPU，只下載統計值與必要 mask。（已拆出 `white_ratio_analysis` profiler；CPU bbox-local counting 改用 OpenCV countNonZero/bitwise_and，512² synthetic median 0.0343→0.0151 ms；GPU 搬移待 RTX/production 佔比證明）
- [x] 評估 connected components；bbox 雖可一致，但 pixel area、孔洞 contour 數與既有排序語意不等價，且固定 seed 4K CPU benchmark 無收益，因此不取代 contours。
- [ ] 全部 detector 遷移並通過 RTX 3090 驗收後，才評估移除 detector-specific compatibility adapter。

## P4：Tiling、ROI、Batch 與跨圖片重用

- [x] 偵測重複 GPU crop round trips，記錄傳輸量並輸出負優化警告。
- [x] production/benchmark 在 device tiling 改善前預設關閉 GPU crop。
- [x] 原圖一次 upload，以 device offset/view 表示 grid ROI，不再每 tile 上傳完整原圖。
- [x] detector 可直接消費 device ROI；只有 CPU contour、GUI、debug 或存檔時才下載。
- [x] 新增 batch ROI API，以座標陣列產生連續 device buffers。
- [x] 新增 `run_batch(images/rois)` 或等價 detector batch 介面；CPU 預設實作可逐張執行。
- [x] 依影像尺寸與可用 VRAM 自動選 batch size，配置失敗時自動縮小批次且不留下 stale handle。
- [ ] RTX 3090 實機測試 8、16、32、64 ROI batch 的正確性、效能與 VRAM 平台。
- [x] 單張 GUI 採低延遲策略；資料夾、monitor、batch 採高吞吐策略。
- [x] 使用 bounded 單一 GPU queue，避免多個 CPU workers 同時搶 GPU 或無限制累積 VRAM。
- [ ] 評估 pinned host memory 與 CUDA streams，量測 upload/kernel/download 重疊收益。

## P5：CPU 與整體 Pipeline 最佳化

- [x] 分別量測 `findContours`、幾何分析、Python tile/detector 迴圈、progress callback、aggregation 與 reporter。
- [x] 降低 progress callback 頻率，避免每個小 primitive 更新 GUI。
- [x] 移除不必要的 detector `image.copy()` 與完整尺寸 temporary masks；必要的 non-contiguous CUDA/QImage 邊界 copy 保留。
- [x] 相同 tile 的 CPU detectors 共用一次 gray；GPU detectors 共用 resident source，避免各自重傳原圖。
- [ ] RTX profiler 證明有收益後，再加入跨 detector 的 device-gray／完整 preprocessing result cache。
- [ ] 對小圖、小 ROI、少 tiles 建立 CPU/GPU crossover benchmark；低於門檻自動選 CPU。（64²～1024² native 401-style matrix 與穩定 1.0x/1.5x threshold 報告已完成；production policy 待 RTX 數據）
- [x] Overlay、NG tiles、CSV/JSON 與純檢測計時分離；目前各 reporter 與 `detectors_total` 已獨立計時，是否背景化由實測決定。
- [x] Pattern matching 目前維持 CPU；只有 RTX profiler 證明為主要熱點後才另案 GPU 化，並要求模板常駐與 CPU 等價路徑。
- [x] PNG 編碼、YAML、彙總、logging 與 GUI 控制邏輯維持 CPU，除非量測證明需要改變。

## P6：GUI、設定與部署

- [x] Recipe 與 GUI 可設定 GPU，並顯示 DLL/device/fallback 狀態。
- [x] GPU mode 統一為清楚的 `auto`、`cpu`、`cuda` 語意，並相容未含 mode 的舊 recipe。
- [ ] production 預設 mode 仍需由 RTX 3090 實機驗收決定。
- [x] GUI worker 不在 UI thread 等待 CUDA；monitor 取消、錯誤與進度以 stop callback／Qt signals 保持可回應。
- [x] GUI 顯示實際 backend，不得因 recipe 勾選 GPU 就顯示 CUDA active。
- [x] PyInstaller 有 DLL 時條件式包含 `gpu/visionflow_cuda.dll`，無 DLL 時建立 CPU-compatible package 且 runtime 可 fallback。
- [ ] 有 GPU、無 GPU、DLL 缺少、DLL 版本不符、fallback 開/關各完成一次打包實機測試。（runtime tests 已覆蓋 missing/ABI mismatch/no-device/context-failure 與 fallback policy；無 NVIDIA/CUDA DLL 電腦已完成 CPU-compatible package build 與 5 recipes bundle，packaged smoke 進一步驗證 MainWindow、CPU-only pipeline、缺 DLL fallback 開啟時與 CPU 結果一致且 GPU call count=0、fallback 關閉/strict CUDA 明確失敗，EXE exit 0；有 GPU 與 packaged ABI mismatch 待實機）

## P7：CI、GitHub Actions 與發布

- [x] 一般 Windows runner 執行 unit tests、compileall、recipe/CLI/GUI smoke 與 CUDA headers/API 靜態檢查。
- [x] DLL 與 test EXE 使用明確 source manifest 分開編譯，不以 glob 無差別加入所有 `.cu`，並以 preflight 靜態驗證。
- [x] workflow 明確加入 `gpu/include/`，上傳 DLL、LIB、test EXE 與 build log artifacts。
- [x] CUDA runtime、CPU/GPU 等價、VRAM leak 與 benchmark 只在 GPU self-hosted runner 執行。
- [x] self-hosted runner 使用 `self-hosted`、`Windows`、`X64`、`gpu`、`rtx3090` labels。
- [x] 不允許不受信任的 fork PR 直接在可接觸本機資料的 self-hosted runner 執行。
- [x] GPU job 支援手動與 nightly；PR 至少完成 compile/static checks。
- [ ] 保存 benchmark JSON、Nsight report、Driver/Toolkit/GPU 與 commit hash，支援 commit 間比較。（JSON、環境與 commit 已完成；workflow 已加入可用時執行 nsys smoke capture 並記錄 skip/status，report 待 RTX runner）

## P8：產線安全、追溯與持續驗證

- [x] Detector 宣告共用參數 schema；recipe 載入嚴格拒絕未知 detector、未知參數、錯誤型別、越界值與非法 enum，GUI designer 使用同一份 schema 建立欄位。
- [x] Inspection 輸出保存原始 recipe SHA-256、套用 runtime override 後的 effective recipe SHA-256，以及 build commit/dirty provenance。
- [x] 每張 NG tile 旁產生 dataset metadata sidecar，包含 recipe provenance、detector/參數、局部與全域座標、來源影像及人工複判欄位。
- [x] 五份 production recipe 皆有可重現的合成 PASS/NG golden regression，斷言 final result、defect count、bbox 容差、area/confidence/metadata 與順序；四種 detector 各至少五個合成案例。
- [x] 建立 Windows 精確 dependency lock；hosted CI、RTX runner 與 PyInstaller build 使用同一份 lock，避免時間與機器造成版本漂移。
- [x] hosted CI 監測 RTX workflow 最近成功時間（超過 48 小時失敗）、benchmark 與 baseline 比較並 gate P95 退化，另有 weekly PyInstaller build + packaged smoke，Python 版本與部署版本一致。
- [x] Hypothesis 隨機產生影像與合法 PreprocessPlan，驗證 CPU executor 與直接 OpenCV reference；固定生成順序可供 RTX CPU/GPU fuzzing，recipe/designer schema 與 GPU ABI/metrics 已拆成可 headless 測試模組。

## RTX 3090 編譯與實機驗收

### 環境與編譯

- [ ] `nvidia-smi` 可看到 GPU，並記錄 Driver、CUDA compatibility 與 VRAM。
- [ ] 安裝 CUDA Toolkit、VS 2022 C++ Build Tools、Windows SDK；確認 `nvcc --version` 與 `where.exe cl`。
- [ ] 使用 x64 Native Tools PowerShell 執行 `gpu/build_cuda_dll.ps1 -Architecture sm_86`。
- [ ] 產生 `visionflow_cuda.dll`、`visionflow_cuda.lib` 與 `test_cuda_api.exe`，沒有 link/architecture 錯誤。
- [ ] `test_cuda_api.exe` 驗證 ABI、device、compute capability、grayscale、context 與 fused smoke。
- [ ] `dumpbin /exports` 檢查所有預期 `vf_` exports；`dumpbin /dependents` 無缺少依賴。

### Primitive、plan 與效能

- [ ] BGR→RGB、crop、threshold、morphology 與 CPU 完全一致。
- [ ] BGR→Gray、resize、Gaussian 與 Adaptive Mean 通過既定像素容差。
- [ ] Gaussian 覆蓋 kernel 3/5/15/25/45 與 structured/non-contiguous inputs。
- [ ] Adaptive Mean 覆蓋 block 3/11/35、正負與小數 C、invert 及邊界輸入。
- [ ] 401-2 fused 與 CPU plan 結果在容差內；相同尺寸連續執行 allocation count 不增加。
- [ ] 通用 native plan 完成後，逐 operator 與完整 plan 對 CPU executor 建立等價矩陣。
- [ ] 記錄 4K primitives、preprocessing plan、純檢測與端到端 CPU/GPU speedup。
- [ ] 連續執行三次完整驗證，沒有 CUDA error、崩潰或 VRAM 持續成長。

### Production recipes、GUI、打包與壓測

- [ ] `PRODUCT_A_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_NEGATIVE_401_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_FRAME_900_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] 比較 tiles、PASS/NG、defect count、bbox、area、confidence、metadata 與 fallback log。
- [ ] GUI 的 recipe 儲存/載入、viewer backend、status、overlay、輸出與 fallback 正確。
- [ ] 打包版在有 NVIDIA GPU 與無 NVIDIA GPU 電腦均完成驗證。（目前無 GPU 電腦已完成 CPU-compatible package build 與 bundled recipe/MainWindow smoke；有 GPU 電腦待驗收）
- [ ] warm-up 5 張後測 10、100、1000 張；VRAM 穩定、GUI 可回應、無 crash/error。（validator/workflow 已加入 checkpoints、allocation/VRAM/median/P95；待 RTX 執行）

## 未來 AI Detector

- [ ] 導入模型時比較 PyTorch CUDA、ONNX Runtime CUDA 與 TensorRT 的部署及效能。
- [ ] 模型/session 只載入一次並常駐 GPU；支援 batch inference 與固定輸入尺寸。
- [ ] 優先驗證 FP16；INT8 必須完成校正與精度驗收後才能啟用。
- [ ] AI 與傳統 CV 共用 GPU scheduler、VRAM budget、warm-up、metrics 與 fallback policy。
- [ ] 避免 GUI、monitor、batch worker 各自載入一份大型模型。

## 最終驗收門檻

- [x] CPU-only 是完整受支援模式，沒有 CUDA/NVIDIA GPU 仍可啟動 GUI、CLI、batch 與 monitor。
- [ ] 五個 production recipes 通過 CPU/GPU 等價規則，沒有未解釋的 fallback。
- [x] 每個 GPU plan 原則上每張輸入最多一次 upload 與一次必要 download；resident ROI plan 額外 H2D 為零。
- [x] native plan/context 預留並重用 operator buffers，相同 shape warm-up 後不再逐 operator `cudaMalloc/cudaFree`。
- [ ] 連續 1000 張後 VRAM 位於穩定平台，沒有資源洩漏或程序崩潰。
- [ ] GPU 純檢測 median 與 P95 在目標資料集均優於 CPU；目標加速門檻為至少 1.5 倍。
- [x] 未達 RTX 效能門檻的 production recipe/operator 保持 CPU、GPU 預設關閉。
- [ ] 加速不得犧牲 GUI 回應、打包啟動、結果追溯、錯誤訊息或 CPU fallback。

## 完成紀錄

- [x] 2026-07-13：建立 `cuda_practice/`、RTX 3090 `sm_86` 練習與編譯說明。
- [x] 2026-07-14：加入 recipe/detector/GUI GPU 開關、CUDA 狀態與安全 CPU fallback。
- [x] 2026-07-14：建立 CUDA DLL C ABI、ctypes bridge、build script、C++ smoke 與 Python 驗證工具。
- [x] 2026-07-14：完成 M0 第一批 profiler、Reporter 分項計時、傳輸統計、crop 警告及容差修正。
- [x] 2026-07-14：完成 CPU-only/缺 DLL fallback 等價回歸。
- [x] 2026-07-14：完成 M1 原始碼：separable Gaussian、constant weights、64-bit integral Adaptive Mean 與 structured tests。
- [x] 2026-07-14：完成 M2 第一個垂直切片：persistent context、grow-only buffers、context stats 與 401-2 fused preprocessing。
- [x] 2026-07-14：建立通用 `PreprocessPlan`、typed operators、CPU/CUDA executors，並遷移 401-2。
- [x] 2026-07-14：將 CPU、GPU、CUDA、GUI、CI、打包與 RTX 3090 驗收清單合併為唯一 `Todo.md`。
- [x] 2026-07-14：更新 `AGENT.md`，統一 Todo 紀律、模組責任、PreprocessPlan/CPU fallback 架構、驗證矩陣、安全 staging 與 commit/push 規範。
- [x] 2026-07-14：更新 `aoi-verify-push`，並新增 `aoi-detector-development`、`aoi-cuda-validate`、`aoi-release` skills；四者皆完成 metadata 與官方 validator 檢查。
- [x] 2026-07-15：整理 2026-07-09 至 2026-07-15 Git 紀錄、GPU/CUDA 進度與待驗收項目，完成本週流水帳報告。
- [x] 2026-07-15：更新並完整中文化根目錄 `README.md`，同步目前 CLI、GUI、配方、Detector、輸出、CUDA fallback、打包與驗證方式。
- [x] 2026-07-17：加入 GUI 預覽影像載入、色彩轉換、QImage/QPixmap、scene 顯示與使用者實際等待時間量測，並輸出至日誌及 viewer backend tooltip。
- [x] 2026-07-17：補齊固定 seed 合成影像、奇數/極小/4K、non-contiguous、1/3 channels、ROI 尺寸 CPU 測試，並驗證關閉 fallback 時缺少 CUDA DLL 會明確失敗；真實照片與 GPU 實機項目保留待辦。
- [x] 2026-07-17：新增 per-detector bounded LRU `PreprocessPlanCache`，依 shape、dtype 與參數 signature 重用 immutable plan；401-2 已移出逐 tile plan 建立熱路徑並加入 cache/失效測試。
- [x] 2026-07-17：加入 versioned operator/plan signature、tensor spec 推導及輸入輸出 dtype/channel/shape/order/參數驗證，CPU 與 CUDA executor 共用相同契約並以 fake runtime 覆蓋錯誤輸出。
- [x] 2026-07-17：加入 preprocess capability report，記錄 requested/selected backend、fused/primitive/CPU/fallback route、原因、plan signature 與不支援項目，並帶入 detector execution metadata。
- [x] 2026-07-17：將 401-1 遷移到 cached shared plan（Gray/Resize area/Gaussian/AdaptiveMean/Morphology），保留 process scale、ROI、contour 與 metadata 語意，area 不支援時維持 full-detector CPU fallback。
- [x] 2026-07-17：將 401 遷移到 cached shared plan，保留 BGR Gaussian/Morphology 後轉 Gray/AdaptiveMean 的既有逐像素順序，以及 ROI、contour、座標、排序與 metadata 語意。
- [x] 2026-07-17：新增 CPU DAG/multi-output plan/executor，900 改以 cached DAG 共用一次 Gray 產生 outer Threshold 與 inner AdaptiveMean masks；CUDA DAG/device gray 另列待辦。
- [x] 2026-07-17：401-2 contour white-ratio 改用局部 bbox mask，避免逐 contour 配置整張 ROI；CPU 測試確認逐像素統計、排序、ROI offset 與 metadata 均與舊 full-ROI 演算法一致。
- [x] 2026-07-17：完成 connected components CPU 評估；合成測試證明 pixel area 與孔洞/list contour 語意不等價，固定 seed 4K/350 blobs benchmark 的 findContours LIST median 3.562 ms、connectedComponentsWithStats 8.063 ms，因此 401/401-1/401-2 維持 CPU contours。
- [x] 2026-07-17：加入 CUDA build preflight 與 SHA-256 manifest，靜態核對 17 個 ABI v1 header/source/runtime/smoke exports；DLL、LIB、test EXE 改在 staging 成功編譯並通過 dumpbin exports/dependencies 後才發布，避免 stale artifacts。
- [x] 2026-07-17：修正 CUDA capability preflight routing；unsupported linear/DAG plan 不再先執行部分 GPU primitive，並讓 `fallback_to_cpu: false` 對 runtime/semantic failure 維持嚴格失敗。
- [x] 2026-07-17：完成 generic native linear plan 原始碼與 OOP routing：versioned detector-neutral structs、optional query/create/execute/destroy、compiled-plan cache、persistent buffers、Gray/Gaussian/Threshold/AdaptiveMean/Morphology 單次 H2D/D2H execution，並同步 Python bridge、fake-DLL lifecycle、C++ smoke、validator、preflight 與文件；RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：persistent context 納入 non-blocking CUDA stream、plan scratch 與 morphology device ping-pong；新增 `GpuExecutionSession` 讓 batch/monitor 跨影像共用同一 runtime/context，並以 pipeline、batch、monitor 與 CUDA source contract 測試驗證生命週期；RTX 3090 runtime 驗證仍保留待辦。
- [x] 2026-07-17：新增 detector-neutral native DAG/multi-output ABI、compiled-plan cache 與 CUDA executor；900 以一次 root H2D 共用 device gray，僅下載 outer/inner masks 並同步一次，已覆蓋 descriptor、fake-DLL lifecycle、detector routing、C++ smoke、validator 與 source contract；RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：新增 detector `run_batch(images/rois)` CPU 預設契約與 manager 介面；GpuRuntime 採 bounded queue 加單一序列化 execution，單張 pipeline 使用 latency depth=1，batch/monitor 使用可設定 throughput depth，production recipes 持續預設關閉負優化 GPU crop。
- [x] 2026-07-17：新增 Windows CPU/static CI 與受信任 RTX 3090 self-hosted manual/nightly workflow；PR 執行 tests、compileall、recipe/CLI/GUI smoke、CUDA contract，GPU job 使用專屬 labels 並上傳 DLL/LIB/EXE/build log、環境及含 commit 的 benchmark JSON；Nsight capture 保留實機待辦。
- [x] 2026-07-17：新增 context-owned resident image 與 linear/DAG device ROI ABI；grid pipeline 每張原圖只 upload 一次，以可組合子 ROI 對應 tile 與 detector inset，ROI plan 僅 D2D staging 並下載必要輸出；已覆蓋 generation/bounds、零額外 H2D、pipeline 單次 upload、C++ smoke、validator 與 source contract，RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：新增 native ROI coordinate batch opaque API，以單一 3D gather kernel 產生連續 device buffers；Python OOP handle 支援 download/context cleanup，依 `cudaMemGetInfo`、ROI 工作集與 8/16/32/64 candidates 自動選批次，配置失敗逐級降批且無 stale handle；validator 已準備四種批次實測，RTX 3090 數據仍保留待辦。
- [x] 2026-07-17：細分 detector preprocess/findContours/geometry、Python tile loop overhead、progress callback、aggregation、純檢測與各 reporter 計時；相同 percent 的 progress callback 去重，移除四個 detector 無必要 input copy，並以測試固定 profiler schema 與 callback 行為。
- [x] 2026-07-17：新增 tile-scope CPU preprocess cache，401-1/401-2/900 共用一次 Gray；稽核並測試五種 GUI worker 均先 moveToThread 再執行、無 UI wait、monitor stop/error/progress 使用 callback/signals，以及 PyInstaller CUDA DLL 條件式收錄與 CPU-only build path。
- [x] 2026-07-17：擴充 RTX validator benchmark schema，分離 cold 與 warm-up、average/median/P95/process CPU%，並記錄 nvidia-smi utilization/VRAM/溫度/功耗/Driver、CPU/RAM/Python、recipe/影像與 commit；workflow 明確 warm-up 5 次，實際 baseline 數據待 RTX runner。
- [x] 2026-07-17：在無 nvcc/CUDA DLL/GPU 環境實跑 CLI（含 outputs）、GUI offscreen、單圖 batch 與 monitor 均成功；確認 production recipes 的 tiling/display/use_gpu 預設全關閉，並以 source/runtime tests 固定 native plan 單次傳輸與 warm-up buffer reuse。
- [x] 2026-07-17：首次手動 dispatch RTX 3090 workflow run `29574501971`；workflow active 且 request 成功，但持續 queued、updated_at 未變，確認目前 self-hosted RTX runner 尚未上線接單。
- [x] 2026-07-17：統一 GPU `auto/cpu/cuda` policy：auto 可安全 fallback、cpu 完全不要求/載入 CUDA、cuda 強制成功且禁止 fallback；recipe 驗證、pipeline、長生命週期 session、GUI preview/tiling worker 與設計器均共用同一語意，GUI/history 顯示實際 backend。
- [x] 2026-07-17：新增 `VfCudaTimingsV1` 與 `vf_context_last_timings`，persistent plan 以 CUDA events 拆分 H2D/D2D、kernel、D2H、Gaussian、Adaptive Mean、threshold 與 device total，host clock 補 context/allocation/synchronize/free；Python metrics、C++ smoke、preflight 與 source/runtime tests 已同步，數值正確性待 RTX runner 驗證。
- [x] 2026-07-17：RTX validator 新增 persistent native plan 累積壓測 checkpoints，workflow 固定 warm-up 5 後跑 10/100/1000 次並保存 allocation count、VRAM、telemetry、average/median/P95 與 CUDA metrics；fake DLL 測試確認 warm-up 後不再配置，且一次 execution error 後可安全重用同一 plan handle。
- [x] 2026-07-17：RTX validator 新增 64²、128²、256²、512²、1024² 的 401-style native plan CPU/GPU crossover matrix，包含 cold/warm-up/median/P95、含傳輸 speedup、穩定 1.0x/1.5x 門檻候選；只輸出證據、不在 RTX 驗收前改 production routing。
- [x] 2026-07-17：新增 production acceptance manifest 與 validator 入口，強制五份 production recipes 各具 PASS/NG、唯一 case id、有效檔案與標籤，逐案執行 CPU/GPU 完整 pipeline 等價並核對 expected final；example 已列出 10 個待提供的真實樣本位置。
- [x] 2026-07-17：`VfCudaTimingsV1` 新增 morphology CUDA event 分項，linear/DAG native plan 均量測完整 morphology passes；RTX validator 加入 detector-401-style close iterations 1/2/4/8 的 CPU/GPU cold/warm/median/P95、含傳輸 speedup 與 morphology/kernel 占比，separable kernel 與 routing threshold 仍待實機數據決策。
- [x] 2026-07-17：新增 persistent context reuse matrix，依序覆蓋 BGR shape grow、gray channel 切換、BGR shrink 與 plan parameter 改變，第二輪要求 allocation count 不再增加；source contract 證明 grow-only reserve 先成功配置 replacement 才釋放舊 pointer，因此單次 OOM 不會破壞既有 buffer，真實 CUDA error/OOM 注入仍待 RTX。
- [x] 2026-07-17：補齊 CUDA loader failure matrix，實跑 ABI mismatch、零 CUDA device 與 persistent context create failure；context failure 現在一致傳遞到 fused、native linear、native DAG capability reason，避免 fallback metadata 誤報成缺少 generic ABI。
- [x] 2026-07-17：RTX workflow 新增可選 `production_manifest` dispatch input，可直接執行五 recipes PASS/NG acceptance；新增 Nsight Systems smoke capture，runner 有 `nsys` 時產生 `.nsys-rep`，否則保存明確 skip status，兩者均納入 artifacts。
- [x] 2026-07-17：401-2 profiler 將 contour white-pixel mask/count 從 geometry 拆成 `white_ratio_analysis`；bbox-local 統計由 NumPy boolean temporaries 改成 OpenCV `bitwise_and`/`countNonZero`，保持 count/ratio/order/metadata 等價，512² synthetic microbenchmark median 由 0.0343 ms 降至 0.0151 ms；是否移至 GPU 留待 RTX production 占比。
- [x] 2026-07-17：完成 native linear `VF_PLAN_RESIZE_AREA` source routing：descriptor 固定 area target、query 拒絕放大/混合軸語意、compiled plan 追蹤 output shape，401-1 下採樣維持單次 H2D/D2H；同步 Python encoding、C++ smoke、RTX validator、舊 DLL fallback 與 fake handle/OOP lifecycle tests，真實 CUDA 等價待 RTX runner。
- [x] 2026-07-17：修正 `build_exe.ps1` 每次覆寫受版控 spec、導致 CUDA 條件式收錄規則遺失的缺陷；改由固定 `VisionFlow AOI.spec` 建置，新增 packaged `--smoke-test` 從 PyInstaller bundle 載入 recipe 並建立 MainWindow。CPU-compatible package 在目前無 GPU 電腦實跑 exit 0、5 recipes、無 CUDA DLL，validation ZIP 103,993,603 bytes、SHA-256 `5E4E833AEA184A7889F2911B56AB22DCFAD3F2E1A6E82D46D60C5C431A4C134F`。
- [x] 2026-07-17：擴充 packaged `--smoke-test` 為缺 DLL fallback policy 全 pipeline 矩陣；CPU-only 與 auto fallback 的 PASS/NG、tiles、defects、bbox、metadata 一致且 GPU call count=0，strict CUDA 明確回報 DLL 不存在；重建 CPU-compatible EXE（5,550,515 bytes、5 recipes、無 CUDA DLL）後實際 exit 0。validation ZIP 103,996,491 bytes、SHA-256 `7477496D9DC5FD47CA99752235D451A132C9C5BC0279F237760FD308471271AD`；Windows CI 通過，RTX workflow 因 repository 無 self-hosted runner 排隊中。
- [x] 2026-07-17：依目前 codebase 稽核並更新 `README.md` 與 `AGENT.md`，同步 Windows／RTX CI、shared preprocess plan、GPU session、CUDA preflight、打包 fallback smoke、專案模組地圖與實際驗證命令；未變更 runtime 行為或 RTX 實機驗收狀態。
- [x] 2026-07-17：修正 Windows CLI smoke 的 exit code 判斷，明確接受 PASS=0 與 NG=2，並讓未捕捉例外等其他 exit code 正確使 CI 失敗。
- [x] 2026-07-17：完成 P8 產線安全與持續驗證：strict detector schema/GUI 共用、recipe/build SHA-256/commit provenance、NG dataset sidecar、五配方與每 detector 至少五個合成 golden cases、Python 3.13 Windows lock、RTX 48h heartbeat/P95 15% gate/weekly package smoke、100-case Hypothesis preprocess fuzz，並拆分 GPU ABI 與 metrics；本機 CPU-compatible PyInstaller build 及 packaged smoke exit 0。
