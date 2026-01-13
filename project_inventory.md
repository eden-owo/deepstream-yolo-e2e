# 專案盤點與環境記錄

## 專案名稱: DeepStream & YOLO E2E
**位置:** `\\wsl.localhost\Ubuntu-22.04\home\user\deepstream-yolo-e2e-rtspsrc`

### 1. 專案解析
此專案是一個基於 NVIDIA DeepStream SDK 的端到端 (End-to-End) YOLO 偵測與分割模型實作項目。其主要特點包括：
- **高效能**: 將 NMS (Non-Maximum Suppression) 直接整合進模型中，提升推論效率。
- **靈活性**: 支援動態 Batch Size 與輸入尺寸。
- **影片來源**: 支援多種來源，包含 YouTube 影片串流與 RTSP 來源。
- **支援模型**: YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOv7 等。

#### 核心技術：端到端 (End-to-End, E2E) 的意義
在影片分析與 AI 推論的領域中，「基於 NVIDIA DeepStream SDK 的端到端」具有兩層核心意義：

1. **全流程的自動化 (DeepStream SDK 的角色)**
   所謂的「端到端」是指從影片輸入到報表/影像輸出的整個流程都在一個連續的硬體加速管道 (Pipeline) 中完成：
   - **端 A (輸入)**: 讀取 RTSP 攝影機、本地影片檔、甚至是 YouTube 串流。
   - **中間過程 (硬體加速)**:
     - **硬體解碼**: 使用 GPU 的 NVDEC 直接解碼影片。
     - **影像預處理**: 縮放、格式轉換 (例如從 YUV 轉成模型需要的 RGB)。
     - **AI 推論**: 使用 TensorRT 核心進行物件偵測 (YOLO)。
     - **追蹤 (Tracking)**: 對偵測到的物件進行 ID 編號，追蹤移動路徑。
     - **繪圖 (OSD)**: 將方框、文字直接畫在影像上。
   - **端 B (輸出)**: 將結果推播到螢幕、轉成 RTSP 串流、或將資料存入資料庫。

2. **模型層級的簡化 (此專案 YOLO E2E 的關鍵)**
   在傳統做法中，YOLO 模型只負責產出原始數據，後續需要 CPU 或額外的程式碼來做後處理 (NMS、座標解析)。而在本專案的 E2E 實做中：
   - **NMS 直接整合進模型**: 透過 TensorRT 插件 (如 EfficientNMS) 將 NMS 邏輯直接寫入 ONNX 檔中。
   - **模型產出的就是最終結果**: DeepStream 拿到的是直接可用的「座標、類別、信心值」，不需額外轉碼。

#### 為什麼這很重要？ (優點)
- **極致效能 (GPU-Only)**: 運算全在 GPU 完成，消除 CPU/GPU 轉導延遲。
- **開發簡單**: 更換 ONNX 檔案即可支援不同 YOLO 版本，無需修改後處理代碼。
- **高度適應性**: 模型支援動態解析度 (Dynamic Shape)。

**總結來說**: 本專案是以「最快、最精簡」的方式處理影像，實現極高的推論張數 (FPS)。


### 2. 環境環境 (Docker / 虛擬環境)
專案主要使用 **Docker** 環境進行開發與部署，依賴 NVIDIA 官方提供的 DeepStream 映像檔。

#### 使用鏡像 (Base Images)
推薦使用以下官方鏡像 (視 DeepStream 版本而定)：
- `nvcr.io/nvidia/deepstream:7.1-triton-multiarch`
- `nvcr.io/nvidia/deepstream:7.0-triton-multiarch`
- `nvcr.io/nvidia/deepstream:6.4-triton-multiarch`
- `nvcr.io/nvidia/deepstream:6.3-triton-multiarch`

#### Docker 執行指令 (WSL2 範例)
```bash
xhost +
docker run \
        -it \
        --privileged \
        --rm \
        --net=host \
        --ipc=host \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e CUDA_CACHE_DISABLE=0 \
        --device /dev/snd \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v `pwd`:/apps/deepstream-yolo-e2e \
        -w /apps/deepstream-yolo-e2e \
        nvcr.io/nvidia/deepstream:7.1-triton-multiarch
```

### 3. 環境安裝方法 (Installation Steps)
專案內建了自動化安裝指令碼與編譯腳本，主要步驟如下：

#### A. 一鍵安裝腳本 (`one_hit_install.sh`)
此腳本負責 container 內部的環境配置：
1. **Python App 安裝**: 根據 DeepStream 版本安裝對應的 `deepstream_python_apps`。
2. **編譯解析函數**: 執行 `scripts/compile_nvdsinfer_yolo.sh`。
3. **套件補丁**: 執行 `TensorRTPlugin/patch_libnvinfer.sh` 進行庫檔案修補。
4. **系統套件**: 安裝 `ffmpeg`。
5. **Python 套件**: 安裝 `yt-dlp` 與 `prettytable`。

#### B. 關鍵腳本位置
- **主安裝程式**: `[one_hit_install.sh](file:///\\wsl.localhost\Ubuntu-22.04\home\user\deepstream-yolo-e2e-rtspsrc\one_hit_install.sh)`
- **YOLO 編譯**: `[scripts/compile_nvdsinfer_yolo.sh](file:///\\wsl.localhost\Ubuntu-22.04\home\user\deepstream-yolo-e2e-rtspsrc\scripts\compile_nvdsinfer_yolo.sh)`
- **庫修補**: `[TensorRTPlugin/patch_libnvinfer.sh](file:///\\wsl.localhost\Ubuntu-22.04\home\user\deepstream-yolo-e2e-rtspsrc\TensorRTPlugin\patch_libnvinfer.sh)`

### 4. 依賴項盤點
- **關鍵子模組 (Git Submodules)**:
  - `nvdsinfer_yolo`: YOLO 解析函數。
  - `yolo_e2e`: 端到端模型匯出工具。
- **Python 依賴**:
  - `yt-dlp` (用於 YouTube 整合)
  - `prettytable`
  - `pyds` (NVIDIA DeepStream Python Bindings)
