# VisionFlow CUDA DLL

## Preprocessing architecture

Detectors must describe preprocessing with the backend-neutral operators in `core/preprocess_plan.py`. The CPU executor defines OpenCV fallback semantics; the CUDA executor may use reusable primitives or a compatible fused adapter. New detectors should compose existing operators instead of adding detector-named DLL exports. The planned native direction is a versioned generic plan ABI with persistent context buffers and one upload/download per supported plan.

`vf_preprocess_401_2_u8` remains an additive ABI v1 compatibility adapter. It is not the template for future detector APIs.

`visionflow_cuda.dll` 是 AOI 的可選 CUDA backend。Recipe 未勾選 GPU 時不載入它；勾選但 DLL/裝置不可用時，依 `fallback_to_cpu` 決定回退 CPU 或明確失敗。

目前 Gaussian blur 使用 horizontal/vertical separable kernels 與 constant weights；Adaptive Mean Threshold 使用 replicate-border 64-bit integral image。公開 C ABI 維持 v1，因此 Python bridge 與既有打包版介面不需修改，但更新原始碼後必須重新編譯 DLL。

新版 DLL 另外提供可選的 persistent context exports。Detector 401-2 會在 capability 可用時，以一次同步呼叫完成 BGR/gray、Gaussian 與 Integral Adaptive Threshold；context buffers 只在容量不足時成長。Python bridge 若載入舊 DLL，會自動保留原本 stateless primitive 路徑。

## 檔案

```text
gpu/
├── include/
│   ├── visionflow_cuda.h            # 公開、穩定的 C ABI
│   ├── visionflow_cuda_errors.h     # 錯誤碼
│   └── visionflow_cuda_internal.cuh # .cu 內部 CUDA helper
├── visionflow_cuda.cu               # 正式 DLL kernels 與 exports
├── test_cuda_api.cu                 # C++ ABI/device smoke
├── validate_cuda_dll.py             # OpenCV 與 AOI CPU/GPU 比對
└── build_cuda_dll.ps1               # Windows 編譯與測試入口
```

## RTX 3090 編譯

安裝 NVIDIA Driver、CUDA Toolkit、Visual Studio 2022 C++ Build Tools 後，在 x64 Native Tools PowerShell 執行：

```powershell
.\gpu\build_cuda_dll.ps1 -Architecture sm_86
```

輸出：

- `gpu/visionflow_cuda.dll`
- `gpu/visionflow_cuda.lib`
- `gpu/test_cuda_api.exe`

build script 明確使用 static CUDA runtime；`nvcuda.dll` 仍由 NVIDIA Driver 提供。

## 編譯並測試

測試 ABI、structured primitive matrix，以及 4K grayscale、Gaussian k45、Adaptive Mean b35 的 CPU/GPU benchmark：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests
```

再加一張真實影像與 recipe，會同時執行 CPU/GPU AOI 結果比對：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests `
  -Image C:\AOI_TEST\sample.png `
  -Recipe .\recipes\PRODUCT_A_AOI_01.yaml
```

完整環境、五個 recipe、GUI、打包與壓力測試清單見 [`cuda-Todo.md`](../cuda-Todo.md)。
