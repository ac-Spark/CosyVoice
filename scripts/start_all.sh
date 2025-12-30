#!/bin/bash
# =============================================================================
# CosyVoice3 服務啟動腳本
# 僅啟動 API Server，所有輸出進 docker logs
# =============================================================================

echo "=========================================="
echo "  CosyVoice3 服務啟動"
echo "=========================================="
echo ""

cd /workspace/CosyVoice

# 啟動 API Server (FastAPI)
echo ">> [系統] 正在啟動 API Server (Port 7857)..."
python3 api.py --host 0.0.0.0 --port 7857 &
API_PID=$!
echo ">> [系統] API PID: $API_PID"

echo ""
echo "=========================================="
echo "  服務已啟動"
echo "  API: http://localhost:7857"
echo "=========================================="
echo ""

# 捕捉訊號以優雅關閉
trap "echo '>> [系統] 收到停止訊號，正在關閉服務...'; kill $API_PID 2>/dev/null; exit" SIGINT SIGTERM

# 等待子進程
wait
