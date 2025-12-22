#!/bin/bash
# =============================================================================
# CosyVoice3 Docker 容器啟動腳本
# 負責：檢查並下載預訓練模型
# =============================================================================

set -e

# 激活 conda 環境
source /opt/conda/etc/profile.d/conda.sh
conda activate cosyvoice

echo "========================================"
echo "  CosyVoice3 容器初始化"
echo "========================================"

MODEL_DIR="/workspace/CosyVoice/pretrained_models/CosyVoice3-0.5B"
MODEL_SOURCE="${MODEL_SOURCE:-huggingface}"

# 必要檔案列表
REQUIRED_FILES=(
    "llm.pt"
    "flow.pt"
    "hifigan.pt"
    "speech_tokenizer_v3.onnx"
    "campplus.onnx"
)

# 檢查預訓練模型
check_model() {
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MODEL_DIR/$file" ]; then
            echo "[WARN] 缺少: $file"
            return 1  # 有檔案缺失，返回失敗
        fi
    done
    return 0  # 所有檔案都存在，返回成功
}

# 下載模型
download_model() {
    echo "[INFO] 下載預訓練模型..."
    mkdir -p "$MODEL_DIR"

    if [ "$MODEL_SOURCE" = "modelscope" ]; then
        echo "[INFO] 使用 ModelScope 下載..."
        python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_DIR')
print('下載完成')
"
    else
        echo "[INFO] 使用 HuggingFace 下載..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_DIR')
print('下載完成')
"
    fi

    if [ $? -eq 0 ]; then
        echo "[OK] 模型下載完成"
    else
        echo "[ERROR] 模型下載失敗"
        exit 1
    fi
}

# 主流程
if check_model; then
    echo "[OK] 預訓練模型已就緒"
else
    echo "[WARN] 預訓練模型不完整，開始下載..."
    download_model
fi

# 修復權限（讓主機用戶可訪問）
echo "[INFO] 修復目錄權限..."
chown -R 1000:1000 /workspace/CosyVoice/pretrained_models || true
chown -R 1000:1000 /workspace/CosyVoice/exp || true
chown -R 1000:1000 /workspace/CosyVoice/tensorboard || true
chown -R 1000:1000 /workspace/CosyVoice/data || true

echo ""
echo "========================================"
echo "  CosyVoice3 環境就緒"
echo "========================================"
echo ""

echo ""
echo "使用方式："
echo "  ./main.sh status    # 檢查狀態"
echo "  ./main.sh prepare   # 準備資料"
echo "  ./main.sh train     # 開始訓練"
echo ""

# 執行傳入的命令（讓輸出進 docker logs）
exec "$@"
