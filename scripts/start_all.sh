#!/bin/bash
# =============================================================================
# CosyVoice3 服務啟動腳本
# 啟動 API Server + Gradio WebUI，所有輸出進 docker logs
# =============================================================================

echo "=========================================="
echo "  CosyVoice3 服務啟動"
echo "=========================================="
echo ""

cd /workspace/CosyVoice

# 啟動 API Server (新的 FastAPI 前端)
echo ">> [系統] 正在啟動 API Server (Port 7857)..."
python3 api.py --host 0.0.0.0 --port 7857 &
API_PID=$!
echo ">> [系統] API PID: $API_PID"

sleep 3

# 啟動 Gradio WebUI (原本的介面)
echo ">> [系統] 正在啟動 Gradio WebUI (Port 7858)..."
python3 webui.py --port 7858 &
WEBUI_PID=$!
echo ">> [系統] WebUI PID: $WEBUI_PID"

echo ""
echo "=========================================="
echo "  服務已啟動"
echo "  API:   http://localhost:7857"
echo "  WebUI: http://localhost:7858"
echo "=========================================="
echo ""

# 捕捉訊號以優雅關閉
trap "echo '>> [系統] 收到停止訊號，正在關閉服務...'; kill $API_PID $WEBUI_PID 2>/dev/null; exit" SIGINT SIGTERM

# 等待子進程
wait
