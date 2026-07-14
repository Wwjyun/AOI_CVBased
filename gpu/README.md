# VisionFlow CUDA DLL

`visionflow_cuda.dll` 是 AOI 的可選 CUDA backend。Recipe 未勾選 GPU 時不載入它；勾選但 DLL/裝置不可用時，依 `fallback_to_cpu` 決定回退 CPU 或明確失敗。

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

只測 ABI、所有 primitive 與 4K grayscale benchmark：

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
