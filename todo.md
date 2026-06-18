# AOI CV 檢測架構 TODO

> 目標：建立一套 Recipe Based 的 AOI CV Framework。  
> 核心流程：原圖切圖 → 多 Detector 平行檢測 → 結果彙整 → 原圖拼回 → 報表輸出。  
> 設計原則：不做全域前處理，每個 Detector 自己做自己的 preprocess / detect / parameter。

---

## 0. 專案基本方向

- [ ] 採用 CV 為主，AI detector 保留為未來 plugin 擴充
- [ ] 使用 Python + OpenCV 作為影像處理核心
- [ ] 使用 PySide6 製作 GUI
- [ ] 支援大圖切圖檢測
- [ ] 支援多 Detector 模組化架構
- [ ] 支援 Recipe 儲存 / 載入
- [ ] 支援工程師模式與 OP 模式
- [ ] 所有結果都要能回寫到原圖座標

---

## 1. 建議專案結構

```text
aoi_cv_system/
├─ main.py
├─ todo.md
├─ README.md
├─ requirements.txt
├─ recipes/
│  ├─ PRODUCT_A_AOI_01.yaml
│  └─ PRODUCT_A_AOI_02.yaml
├─ core/
│  ├─ image_loader.py
│  ├─ tiler.py
│  ├─ pipeline.py
│  ├─ recipe_manager.py
│  ├─ detector_manager.py
│  ├─ aggregator.py
│  ├─ result_mapper.py
│  └─ reporter.py
├─ detectors/
│  ├─ base_detector.py
│  ├─ detector_999.py
│  ├─ detector_102.py
│  ├─ detector_305.py
│  ├─ detector_777.py
│  └─ detector_888.py
├─ gui/
│  ├─ main_window.py
│  ├─ image_viewer.py
│  ├─ recipe_panel.py
│  ├─ detector_param_panel.py
│  └─ result_panel.py
└─ outputs/
   ├─ overlay/
   ├─ ng_tiles/
   ├─ csv/
   └─ json/
```

---

## 2. MVP 第一階段：核心流程先跑通

目標：先不做完整 GUI，先用 command line 跑通整個 AOI pipeline。

### 2.1 Image Loader

- [ ] 支援讀取 jpg / png / bmp / tif
- [ ] 保留原圖尺寸資訊
- [ ] 檢查圖片是否成功讀取
- [ ] 大圖讀取失敗時要輸出明確錯誤訊息
- [ ] 統一轉成 OpenCV 使用格式
- [ ] 未來預留 BigTIFF / 超大圖分塊讀取

### 2.2 Tiler 切圖模組

- [ ] 支援 tile width / tile height
- [ ] 支援 overlap_x / overlap_y
- [ ] 每個 tile 都要記錄：
  - [ ] tile_id
  - [ ] x
  - [ ] y
  - [ ] width
  - [ ] height
  - [ ] row
  - [ ] col
- [ ] 邊界不足一整塊時要能處理
- [ ] 支援 debug 模式輸出 tile 小圖
- [ ] 正式模式只保留 NG tile，不全部輸出

### 2.3 Detector 統一介面

- [ ] 建立 `BaseDetector`
- [ ] 每個 detector 都要包含：
  - [ ] detector_id
  - [ ] detector_name
  - [ ] display_name
  - [ ] default_params
  - [ ] preprocess()
  - [ ] detect()
  - [ ] run()
- [ ] 所有 detector 輸出格式統一

建議統一輸出格式：

```python
{
    "detector_id": "999",
    "display_name": "黑點異物檢測",
    "pass": False,
    "score": 0.92,
    "defects": [
        {
            "type": "black_blob",
            "bbox_local": [x, y, w, h],
            "area": 120,
            "confidence": 0.92,
            "metadata": {}
        }
    ]
}
```

### 2.4 Aggregator 結果彙整

- [ ] 收集同一個 tile 的所有 detector 結果
- [ ] 支援 all pass 模式
- [ ] 支援任一重要 detector NG 直接 NG
- [ ] 預留 score rule 模式
- [ ] 輸出 tile-level PASS / NG 結果
- [ ] 統計 NG 數量
- [ ] 統計各 detector 觸發次數

### 2.5 Mapper 原圖座標轉換

- [ ] 將 bbox_local 轉成 bbox_global
- [ ] 公式：
  - [ ] global_x = tile_x + local_x
  - [ ] global_y = tile_y + local_y
- [ ] 所有 NG defect 都要保留原圖座標
- [ ] 支援畫回原圖 overlay
- [ ] 支援輸出 NG 位置表

### 2.6 Reporter 報表輸出

- [ ] 輸出原圖 overlay
- [ ] 輸出 NG tile
- [ ] 輸出 CSV
- [ ] 輸出 JSON
- [ ] 報表內容至少包含：
  - [ ] image_name
  - [ ] recipe_name
  - [ ] machine_id
  - [ ] product_id
  - [ ] final_result
  - [ ] detector_id
  - [ ] defect_type
  - [ ] bbox_global
  - [ ] bbox_local
  - [ ] tile_id
  - [ ] score
  - [ ] area

---

## 3. 第一批 Detector

先做簡單可控的 CV detector，不要一開始塞太多。

### 3.1 detector_999：黑點 / 異物檢測

- [ ] detector_id = "999"
- [ ] 功能：抓黑點、異物、局部 blob
- [ ] 參數：
  - [ ] threshold
  - [ ] min_area
  - [ ] max_area
  - [ ] blur_size
  - [ ] invert
  - [ ] clahe_enabled
- [ ] preprocess 由 detector 內部自行處理
- [ ] detect 使用 threshold + contour / connected components
- [ ] 輸出 defect bbox / area / score

### 3.2 detector_102：線狀刮傷檢測

- [ ] detector_id = "102"
- [ ] 功能：抓線狀瑕疵、刮傷
- [ ] 參數：
  - [ ] canny_low
  - [ ] canny_high
  - [ ] min_length
  - [ ] max_width
  - [ ] morphology_kernel
- [ ] preprocess 可使用 blur / CLAHE / Sobel
- [ ] detect 可使用 Canny + morphology + contour
- [ ] 輸出線狀 defect bbox / length / score

### 3.3 detector_305：亮暗異常檢測

- [ ] detector_id = "305"
- [ ] 功能：抓局部過亮、過暗、光源不均
- [ ] 參數：
  - [ ] mean_min
  - [ ] mean_max
  - [ ] std_max
  - [ ] percentile_low
  - [ ] percentile_high
- [ ] 使用 tile 區域統計判定
- [ ] 可支援 cell-based 區域分割

### 3.4 detector_777：Pattern Match 檢測

- [ ] detector_id = "777"
- [ ] 功能：固定 pattern / 缺件 / 位置偏移檢測
- [ ] 參數：
  - [ ] template_path
  - [ ] match_threshold
  - [ ] max_count
  - [ ] min_count
  - [ ] nms_threshold
- [ ] 支援 template matching
- [ ] 支援數量上下限判定
- [ ] 支援把 match 結果畫回原圖

### 3.5 detector_888：Texture / 紋理異常檢測

- [ ] detector_id = "888"
- [ ] 功能：抓布料紋理不均、皺摺、壓痕
- [ ] 參數：
  - [ ] laplacian_var_min
  - [ ] local_std_min
  - [ ] local_std_max
  - [ ] block_size
- [ ] 可先用 local std / Laplacian variance
- [ ] 未來再考慮 Gabor / LBP

---

## 4. Recipe 系統

目標：不同產品、不同機台可以有不同 recipe。

### 4.1 Recipe 格式

- [ ] 使用 YAML 儲存
- [ ] Recipe 包含：
  - [ ] recipe_name
  - [ ] product_id
  - [ ] machine_id
  - [ ] version
  - [ ] tile 設定
  - [ ] decision 設定
  - [ ] detector 啟用清單
  - [ ] detector 參數
  - [ ] output 設定

範例：

```yaml
recipe_name: "PRODUCT_A_AOI_01"
product_id: "PRODUCT_A"
machine_id: "AOI_01"
version: "0.1.0"

tile:
  width: 512
  height: 512
  overlap_x: 64
  overlap_y: 64

decision:
  mode: "all_detectors_must_pass"
  important_detectors:
    - "999"
    - "102"
  max_ng_count: 0

detectors:
  "999":
    enabled: true
    display_name: "黑點異物檢測"
    params:
      threshold: 45
      min_area: 20
      max_area: 5000
      blur_size: 3
      invert: false
      clahe_enabled: true

  "102":
    enabled: true
    display_name: "線狀刮傷檢測"
    params:
      canny_low: 30
      canny_high: 120
      min_length: 50
      max_width: 8

output:
  save_overlay: true
  save_ng_tiles: true
  save_csv: true
  save_json: true
```

### 4.2 Recipe Manager

- [ ] 載入 recipe
- [ ] 儲存 recipe
- [ ] 另存新 recipe
- [ ] 驗證 recipe 格式
- [ ] 檢查 detector_id 是否存在
- [ ] 缺少參數時補 default value
- [ ] 不允許 OP 模式直接覆蓋 recipe
- [ ] recipe 修改要記錄時間與版本

---

## 5. GUI 第二階段

目標：讓工程師可以調參，讓 OP 只需要選 recipe 開始檢測。

### 5.1 GUI 主畫面

- [ ] 左側：Recipe 清單
- [ ] 中間：圖片預覽 / overlay 顯示
- [ ] 右側：Detector 參數面板
- [ ] 下方：log / 結果表 / NG 統計
- [ ] 支援載入單張圖片
- [ ] 支援載入資料夾
- [ ] 支援開始檢測
- [ ] 支援停止檢測
- [ ] 支援只顯示 NG
- [ ] 支援點選 NG 後跳到對應位置

### 5.2 工程師模式

- [ ] 可啟用 / 停用 detector
- [ ] 可調 detector 參數
- [ ] 可測試目前 tile
- [ ] 可測試整張圖
- [ ] 可儲存 recipe
- [ ] 可另存 recipe
- [ ] 可還原 default params
- [ ] 可查看每個 detector 的 debug image
- [ ] 工程師模式需要密碼或權限切換

### 5.3 OP 模式

- [ ] 只能選 recipe
- [ ] 只能載入圖片 / 資料夾
- [ ] 只能開始檢測
- [ ] 只能查看 PASS / NG
- [ ] 不能修改 detector 參數
- [ ] 不能覆蓋 recipe
- [ ] 結果顯示要簡單明確：
  - [ ] PASS 綠色
  - [ ] NG 紅色
  - [ ] 顯示 NG 數量
  - [ ] 顯示輸出資料夾位置

### 5.4 Detector 參數面板

- [ ] 根據 detector default_params 自動產生 UI
- [ ] bool 類型使用 checkbox
- [ ] int / float 類型使用 spinbox
- [ ] string / path 類型使用 line edit + browse button
- [ ] 每個參數要顯示目前值
- [ ] 修改後可立即套用
- [ ] 修改後標記 recipe dirty 狀態

---

## 6. Debug 與驗證

### 6.1 Debug 圖

- [ ] 每個 detector 可選擇是否輸出 debug image
- [ ] debug image 包含：
  - [ ] 原始 tile
  - [ ] detector preprocess 後影像
  - [ ] threshold / edge 結果
  - [ ] defect overlay
- [ ] debug image 要依 detector_id 分資料夾

### 6.2 檢測驗證

- [ ] 建立 PASS 樣本資料夾
- [ ] 建立 NG 樣本資料夾
- [ ] 建立誤判樣本資料夾
- [ ] 建立漏檢樣本資料夾
- [ ] 每次修改 recipe 後可跑 validation
- [ ] 統計：
  - [ ] PASS 正確率
  - [ ] NG 抓出率
  - [ ] 誤判率
  - [ ] 漏檢率

---

## 7. 未來擴充

### 7.1 AI Detector Plugin

- [ ] 預留 YOLOX detector
- [ ] 預留 RT-DETR detector
- [ ] AI detector 一樣遵守 BaseDetector 輸出格式
- [ ] AI detector 一樣由 recipe 控制參數
- [ ] AI detector 可以和 CV detector 混合判定

### 7.2 多機台管理

- [ ] 同產品不同機台 recipe 分開
- [ ] 支援 recipe 複製
- [ ] 支援 recipe 版本管理
- [ ] 支援 machine_id 綁定
- [ ] 支援機台間公差微調紀錄

### 7.3 報表與追溯

- [ ] 每張圖輸出檢測紀錄
- [ ] 每批次輸出總表
- [ ] 支援批號 / 工單號 / OP ID
- [ ] 支援 NG 圖片追溯
- [ ] 支援 recipe version 追溯

---

## 8. 開發順序建議

### Phase 1：核心 CLI 版本

- [ ] 建立專案資料夾
- [ ] 完成 image_loader.py
- [ ] 完成 tiler.py
- [ ] 完成 base_detector.py
- [ ] 完成 detector_999.py
- [ ] 完成 detector_manager.py
- [ ] 完成 aggregator.py
- [ ] 完成 result_mapper.py
- [ ] 完成 reporter.py
- [ ] 用 main.py 跑完整流程

### Phase 2：Recipe 版本

- [ ] 完成 recipe yaml 格式
- [ ] 完成 recipe_manager.py
- [ ] pipeline 改成讀 recipe 執行
- [ ] 支援不同機台 recipe
- [ ] 支援 detector 參數由 recipe 控制

### Phase 3：GUI 工程師版

- [ ] 完成 main_window.py
- [ ] 完成 image_viewer.py
- [ ] 完成 recipe_panel.py
- [ ] 完成 detector_param_panel.py
- [ ] 可調 detector 參數
- [ ] 可測試單張圖
- [ ] 可儲存 recipe

### Phase 4：GUI OP 版

- [ ] 加入 OP 模式
- [ ] 限制 OP 權限
- [ ] 簡化操作流程
- [ ] 顯示 PASS / NG 結果
- [ ] 輸出報表

### Phase 5：Detector 擴充與驗證

- [ ] 加入 detector_102
- [ ] 加入 detector_305
- [ ] 加入 detector_777
- [ ] 加入 detector_888
- [ ] 建立 validation dataset
- [ ] 建立誤判 / 漏檢回饋流程

---

## 9. 優先做的最小可用版本

最小可用版本只需要先完成：

- [ ] 原圖讀取
- [ ] 切圖 + overlap
- [ ] detector_999 黑點異物檢測
- [ ] bbox local → global
- [ ] 原圖 overlay 畫框
- [ ] CSV / JSON 輸出
- [ ] YAML recipe 控制 detector_999 參數

這一版完成後，就可以開始拿現場圖片測誤判與漏檢。

---

## 10. 注意事項

- [ ] 不要把所有 preprocess 寫死在 pipeline
- [ ] 不要讓 OP 直接改 detector 參數
- [ ] 不要只輸出 tile 座標，一定要輸出原圖座標
- [ ] 切圖一定要有 overlap
- [ ] 正式檢測不要全部 tile 都存檔，避免硬碟爆掉
- [ ] 每個 detector 要能獨立 debug
- [ ] recipe 要能追溯版本
- [ ] 所有 detector 都要遵守統一輸出格式
- [ ] 後續 AI 模型也應該包成 detector plugin，不要破壞主架構
## Progress / Completed Items

- [x] Phase 1 CLI project structure
- [x] `requirements.txt`
- [x] `README.md`
- [x] `main.py` CLI entry point
- [x] `core/image_loader.py`: OpenCV image loading for jpg/png/bmp/tif/tiff
- [x] `core/tiler.py`: tile width/height, overlap, tile metadata
- [x] `detectors/base_detector.py`: shared detector API
- [x] `detectors/detector_999.py`: threshold/blob detector
- [x] `core/detector_manager.py`: detector registry and factory
- [x] `core/aggregator.py`: tile/final PASS-NG aggregation
- [x] `core/result_mapper.py`: local bbox to global bbox mapping
- [x] `core/reporter.py`: overlay, NG tile, CSV, JSON output
- [x] `core/recipe_manager.py`: YAML recipe loading and validation
- [x] `recipes/PRODUCT_A_AOI_01.yaml`: sample recipe and detector parameters
- [x] `detectors/detector_102.py`: scratch / thin line detector
- [x] `detectors/detector_305.py`: brightness / uniformity detector
- [x] `detectors/detector_777.py`: pattern match count detector
- [x] `detectors/detector_888.py`: texture / blur detector
- [x] `gui/main_window.py`: PySide6 main window
- [x] `gui/image_viewer.py`: image and overlay viewer
- [x] `gui/recipe_panel.py`: recipe metadata and detector list
- [x] `gui/detector_param_panel.py`: detector parameter display
- [x] `gui/result_panel.py`: result summary and defect table
- [x] `main.py --gui`: GUI launch path
- [x] GUI phase shell
- [x] `AGENT.md`: repository agent workflow instructions
- [x] Local Codex skill `aoi-verify-push`: validate, update todo, commit, and push workflow
- [x] `core/image_loader.py` and `gui/image_viewer.py`: PIL-first image loading with disabled Pillow pixel limit before OpenCV/preview conversion
- [x] GUI Chinese localization and background workers for image preview loading and inspection execution
- [x] Contour-based tile splitting with binary/adaptive/inverse thresholding, shape filters, and subpixel metadata
- [x] GUI recipe design tab with contour threshold controls, tile preview, and recipe YAML export
- [x] `recipes/PRODUCT_A_CONTOUR_AOI_01.yaml`: contour tiling recipe for GUI preview and per-tile detector execution
- [x] Pattern match tile mode with multi-match preview, top-to-bottom/left-to-right sorting, and Recipe wording in GUI
- [x] Detector000 binary contour area guard and pattern-match Recipe using detector 000
- [x] GUI Recipe designer admin-mode detector selection and editable detector parameters
- [x] Detector001 circle threshold NG detector and circle overlay preview with thicker frames
- [x] GUI progress bar for image loading, tile preview, inspection, aggregation, and report output steps
- [x] Pattern-match inspection overlay shows each matched unit as OK green frame or NG red frame
- [x] 2026-06-11 GUI redesign validation: rail/topbar/screen shell, run/results/designer screens, settings drawer, themed widgets, and design handoff files
- [x] 2026-06-11 GUI output toggles wired to reporter `save_*` options and validated with no-output pipeline smoke
- [x] OP mode
- [x] 2026-06-12 GUI tile preview QFont point-size warning fixed, app font stabilized, and preview matching bounded
- [x] 2026-06-12 Pattern Match preview crash fixed by painting preview pixmap without QLabel resize recursion
- [x] 2026-06-12 GUI run-page Pattern Match overlay uses match_bbox status frames for each matched unit
- [x] 2026-06-12 GUI run-page image viewer zoom/fit buttons restored with visible dark-toolbar icons
- [x] 2026-06-12 GUI batch folder inspection panel and batch data summary wired through OOP batch processor
- [x] 2026-06-12 GUI left-nav Batch Data Dashboard with OOP chart statistics model
- [x] 2026-06-12 Recipe designer saves template picture path into recipe assets, tiling config, and detector 777 params
- [x] 2026-06-18 Detector401-1 adaptive mean circular contour NG detector and recipe
- [x] 2026-06-18 Detector401-1 contour list alias and 100px inset ROI
- [x] 2026-06-18 Removed legacy test detectors and recipes, leaving Detector401-1
- [ ] debug image export per detector
- [x] editable recipe saving from GUI
- [ ] validation dataset
- [ ] AI detector plugin phase
