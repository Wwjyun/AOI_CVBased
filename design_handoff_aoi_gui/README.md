# Handoff: AOI 視覺檢測系統 GUI 重新設計（→ PySide6）

## Overview
本套設計是 `AOI_CVbased` 專案 PySide6 GUI（`gui/main_window.py` 等）的重新設計，目標是解決原介面「按鈕雜亂、層級不清」的問題。設計涵蓋三個畫面：**檢測執行**、**Recipe 設計**、**檢測結果**，外加 **設定抽屜** 與 **操作員 OP / 工程師模式切換**。

## About the Design Files
本資料夾內的 HTML/JSX/CSS 檔案是 **以 HTML 製作的設計參考（互動原型）**，展示預期的外觀與行為，**不是要直接拿去用的程式碼**。你的任務是：**在現有的 PySide6 codebase（`AOI_CVbased/gui/`）中重現這份設計**，沿用既有的 worker/thread 架構（`gui/workers.py`）、core pipeline 與 signal/slot 模式，只重寫 UI 層。

打開 `AOI Console.html`（需經 local server 或瀏覽器直接開啟，會載入 `app/` 內的 JSX）即可操作原型：載入影像 → 載入 Recipe → 開始檢測 → 查看結果。

## Fidelity
**High-fidelity（hifi）**：顏色、字級、間距、圓角、互動狀態皆為最終設計值，請依下方 Design Tokens 盡量精確重現（PySide6 以 QSS + custom widget 實作）。

---

## 整體結構（取代現有 MainWindow 佈局）

```
┌──┬────────────────────────────────────────────┐
│  │ Top bar (52px)                              │
│左│  畫面標題 │ [影像 chip] [Recipe chip] …    │
│側│           進度條(執行中)    [OP|工程師]     │
│導├────────────────────────────────────────────┤
│覽│                                             │
│56│  Screen content（padding 12px）             │
│px│                                             │
│  ├────────────────────────────────────────────┤
│  │ Status bar (26px, mono 字體)                │
└──┴────────────────────────────────────────────┘
```

對應現有程式的改動：
- **移除** 原本的 `QToolBar`（載入圖片/載入 Recipe/輸出目錄/開始檢測/進度條全部擠在一起）。
- **移除** 左側 `QTabWidget`（檢測執行/Recipe 設計分頁）→ 改為左側 56px 圖示導覽欄（垂直 QToolButton 列）。
- 「載入圖片」「載入 Recipe」→ 變成 top bar 上的 **狀態 chip**（顯示目前檔名，點擊開啟選擇對話框/抽屜）。
- 「開始檢測」→ 移到檢測執行畫面右側控制面板，成為**唯一的大主按鈕**。
- 「輸出目錄」與輸出選項 → 收進**設定抽屜**（左下角齒輪）。

### 左側導覽欄（rail, 56px 寬）
- 背景 `#ffffff`，右邊框 1px `#e1e6e8`。
- 頂部 logo 方塊 34×34px、圓角 8px、背景 accent 色、白字。
- 導覽按鈕 40×40px、圓角 6px、置中 icon（19px line icon）：
  - 檢測執行（play icon）、Recipe 設計（pencil icon，**OP 模式時隱藏**）、檢測結果（table icon）
  - active 狀態：背景 `#e3f3f1`、icon 色 `#0a6b62`、左側 3px 圓角 accent 指示條
  - hover：背景 `#eef1f2`；hover 時顯示右側 tooltip
- 底部：設定（齒輪）→ 開啟設定抽屜。

### Top bar（52px）
- 背景白、底框 1px `#e1e6e8`、左右 padding 16px、元素間距 12px。
- 左→右：畫面標題（14px/600）、1px 分隔線、影像 chip、Recipe chip、（執行中時）寬 180px 進度條＋百分比、彈性空間、模式切換 segmented control（操作員 OP｜工程師）。
- **Chip 規格**：高 30px、padding 0 11px、背景 `#f8fafa`、邊框 1px `#e1e6e8`、圓角 6px；內容 = icon(14px) + 標籤(12px 灰) + 檔名(mono 12px 深色)。未載入時邊框改虛線、文字「點擊載入」。檢測執行中 disable。

### Status bar（26px）
- mono 字體 11px、色 `#8a979e`；左側顯示最近一次狀態訊息（同原 `statusBar().showMessage`），右側 `AOI_01 · 操作員/工程師模式`。

---

## Screens

### 1. 檢測執行（主畫面）
左右佈局：左 = 影像檢視器（彈性填滿），右 = 318px 固定寬側欄，間距 12px。

**影像檢視器**（重構 `gui/image_viewer.py`）
- 自帶頂列（深色 `#1d2326`）：左側檔名（mono、白 60%）、右側縮放工具（縮小/放大/符合視窗 icon 按鈕）＋「缺陷 Overlay」切換按鈕（開啟時 accent 底白字）。
- 畫布區背景 `#171c1f`；支援滾輪縮放（以游標為中心）、拖曳平移、雙態 grab/grabbing 游標。
- 未載入時顯示置中 empty state（圖片 icon＋「尚未載入檢測影像」）。
- 缺陷框：絕對定位矩形，邊框 1.5px、依類型配色（blob `#ff5d52`、scratch `#ffb13d`、uniformity `#5db6ff`），可點擊選取；選取時邊框 2.5px＋外圈光暈＋左上角顯示 `#id type score` 標籤（深字、彩底、mono 11px）。
- 執行中顯示掃描線動畫（accent 色水平線上下掃）＋右上角「檢測中 n%」浮層。
- 底部狀態列（深色）：影像尺寸、zoom %、游標影像座標（mono 11px）。

**右側欄 — 工程師模式**
1. 「檢測控制」面板：
   - 大主按鈕「開始檢測」（高 40px、accent 底白字、play icon），影像或 Recipe 未載入時 disabled＋提示文字。
   - 執行中：按鈕變「檢測執行中…」＋進度條（5px、圓角、accent）＋階段訊息（mono）。階段訊息沿用 pipeline 階段：載入圖片→切圖→Detector 999→…→輸出 overlay/CSV/JSON。
   - 完成後顯示結果卡：PASS 綠底/NG 紅底色帶＋耗時、三欄統計（Tiles / NG Tiles / 缺陷）、底部「查看完整結果 →」連到結果頁。
2. 「Recipe」面板（重構 `gui/recipe_panel.py` + `detector_param_panel.py`）：
   - 名稱（mono 600）＋ badge 列（product / machine / version / tile mode）。
   - Detector 清單：每列 = 展開箭頭 + ID(mono 600) + 中文名 + 啟用/停用 badge；點擊展開唯讀參數表（取代原本獨立的 DetectorParamPanel）。
   - 右上「更換」→ 開 Recipe 選擇。

**右側欄 — 操作員 OP 模式**（新功能）
- 大狀態面板：44px mono 粗體顯示「待機 / n% / PASS / NG」（執行中 accent 色＋pulse、PASS 綠、NG 紅），下方一行摘要。
- 52px 高特大「開始檢測」按鈕。
- 「本批紀錄」面板：時間 / PASS-NG badge / 缺陷數 的簡表。
- OP 模式下：隱藏 Recipe 設計導覽、不顯示任何參數編輯。

### 2. Recipe 設計（重構 `gui/recipe_designer_panel.py`）
- 左欄 360px（可捲動）：
  1. 「Recipe 資訊」：名稱/產品/機台/版本（mono 輸入框）。
  2. 「切圖 Tiling」：面板標題右側 segmented control 切換 **Pattern Match / Grid / Contour**，下方表單隨模式切換（pattern match：Template 路徑+選擇鈕、匹配門檻、最大匹配數、NMS 門檻、裁切外擴、排序列容差；grid：寬/高/重疊；contour：最小面積/近似 ε）。數值欄一律附上下步進鈕（自製 stepper，非原生 QSpinBox 樣式）。
  3. 「切圖預覽」：影像縮圖，預覽成功後疊綠色 (`#39d98a`) 匹配框；下方狀態文字（執行中 spinner / 成功 accent 色 / 失敗紅色）。
- 右欄（彈性）：「Detector 選用與參數」雙欄面板 — 左 280px 清單（toggle 開關 + ID + 中文名 + 英文名，點擊選取高亮），右側為選取 detector 的**可編輯**參數表（bool→toggle、int/float→stepper、str→mono 輸入框，同原 `_make_param_widget` 的型別對應）。
- 底部動作列（獨立面板）：左「已啟用 n 個 detector」、右「預覽切圖」（次要鈕）＋「儲存 Recipe」（主鈕）。儲存後自動載入該 recipe（同原 `recipe_saved` → `load_recipe` 流程）。

### 3. 檢測結果（重構 `gui/result_panel.py`，從主畫面底部抽出為獨立頁）
- 無結果時：置中 empty state＋「前往檢測執行」按鈕。
- 摘要列（5 卡，間距 12px）：PASS/NG 大字卡（30px mono 800，PASS 綠底 `#e4f5ea`/NG 紅底 `#fcebea`）、Tiles、NG Tiles（紅字）、缺陷數（紅字）、耗時。
- 左（flex 3）「缺陷清單」表格：欄位 # / Tile / Detector / 類型（色點＋名稱）/ Global bbox / 面積 / 分數 / 檢視鈕。
  - 表頭 sticky、11px 大寫灰字、底色 `#f8fafa`；列 hover `#f0f9f8`、選取 `#e3f3f1`；數值欄 mono。
  - 面板標題右側 segmented filter：全部｜各 detector ID。
  - 「檢視」→ 切回檢測執行畫面並選取該缺陷（viewer 高亮）。
- 右（flex 1.2）：「NG Tiles」縮圖牆（104px 方形 crop、左上 #id 標籤、底部 tile·detector mono 條、點擊同「檢視」）＋「輸出檔案」清單（overlay/csv/json 路徑列）。

### 設定抽屜（新）
- 由右側滑入、寬 380px、白底、左框＋大陰影，背後 35% 暗遮罩，Esc/點遮罩關閉。
- 內容：「輸出」區（輸出目錄 mono 輸入＋瀏覽鈕；儲存 overlay / NG tiles / CSV / JSON 四個 toggle）＋「機台」區（Machine ID、Pipeline 版本，唯讀）。
- 載入影像/Recipe 的選擇器在原型中也是抽屜（檔案列表）；PySide6 實作可直接用 `QFileDialog`，但 chip 的顯示行為要保留。

---

## Interactions & Behavior
- **檢測流程**：開始檢測 → 鎖定 chip 與按鈕 → 進度依 pipeline 階段更新（top bar 進度條＋控制面板＋status bar 同步）→ 完成後 viewer 疊缺陷框、控制面板顯示結果卡、寫入本批紀錄。沿用既有 `InspectionWorker`/`QThread` 架構。
- **缺陷選取雙向同步**：viewer 點框 ↔ 結果表選列 ↔ NG tile 縮圖，三處共用同一個 selected id。
- **模式切換**即時生效；OP 模式下若停在 Recipe 設計頁則自動跳回檢測執行。
- 動畫：面板/缺陷框淡入 0.2s ease；toggle/按鈕 transition 0.12–0.15s；掃描線 1.6s linear loop。PySide6 可用 `QPropertyAnimation`，掃描線可省略或用 QTimer。
- Hover：按鈕/列 hover 設色見 tokens；focus ring = 2px accent outline。

## State Management
- `screen: run | designer | results`、`mode: op | eng`
- `image_path`、`recipe`（含 detectors 設定）、`running`、`run_pct`、`run_msg`
- `result {final, summary{tile_count, ng_count, defect_count}, defects[], dur}`
- `selected_defect_id`、`show_overlay`、`history[]`、`output_dir`、`output_opts`

## Design Tokens

### 顏色
| Token | 值 | 用途 |
|---|---|---|
| bg | `#f3f5f6` | 視窗背景 |
| surface | `#ffffff` | 面板/欄 |
| surface-2 | `#f8fafa` | chip、表頭、次要底 |
| surface-3 | `#eef1f2` | hover、segmented 底 |
| viewer-bg / viewer-bg-2 | `#171c1f` / `#1d2326` | 影像區 |
| border / border-strong | `#e1e6e8` / `#c9d1d4` | 框線 / 輸入框 |
| text / text-2 / text-3 | `#1b262c` / `#51616a` / `#8a979e` | 主/次/弱文字 |
| **accent** | `#0d9488`（hover `#0b7d73`、淡底 `#e3f3f1`、文字 `#0a6b62`） | 主按鈕、active、進度 |
| pass / pass-soft | `#1a9e54` / `#e4f5ea` | PASS |
| ng / ng-soft | `#d6453d` / `#fcebea` | NG |
| 缺陷框 | blob `#ff5d52`、scratch `#ffb13d`、uniformity `#5db6ff` | overlay |

### 字體
- UI：**IBM Plex Sans**，中文 fallback **Noto Sans TC**（Windows 可用 Microsoft JhengHei）。
- 數值/檔名/座標/ID：**IBM Plex Mono**。
- 字級：body 13px、small 12px、mono 12px、面板標題 12px/600 大寫 letter-spacing 0.04em、畫面標題 14px/600。

### 形狀與間距
- 圓角：4px（輸入框）/ 6px（按鈕、chip）/ 10px（面板）；badge 全圓。
- 控件高：按鈕 32px（大 40px、小 26px）、輸入框 30px、chip 30px、表格列 ~32px。
- 面板 padding 16px、畫面 padding 12px、面板間距 12px。
- 陰影：sm `0 1px 2px rgba(20,32,38,0.06)`、抽屜 `0 10px 32px rgba(20,32,38,0.18)`。

## Assets
- Icon 全部為 24px viewBox、stroke 1.7、round cap 的 line icon（見 `app/icons.jsx`），PySide6 可改用 [Lucide](https://lucide.dev) / qtawesome 同風格 icon。
- 原型中的 PCB 影像為 canvas 模擬圖，實際應顯示載入的檢測影像與 pipeline 輸出的 overlay。

## Files
- `AOI Console.html` — 入口（載入順序見檔內 script 標籤）
- `app/tokens.css` — design tokens（CSS 變數，含 compact 密度變體）
- `app/ui.css` — 元件樣式（rail/topbar/按鈕/chip/badge/segmented/面板/表單/表格/抽屜）
- `app/data.js` — 模擬資料（detector 定義、recipe、缺陷、pipeline 階段）
- `app/icons.jsx` / `app/components.jsx` — icon 與共用元件
- `app/viewer.jsx` — 影像檢視器（縮放/平移/缺陷框/掃描動畫）
- `app/screen-run.jsx` / `app/screen-designer.jsx` / `app/screen-results.jsx` — 三個畫面
- `app/main.jsx` — shell（rail、topbar、抽屜、狀態）
- `app/tweaks-panel.jsx` — 原型專用的調參面板，**不需移植**
