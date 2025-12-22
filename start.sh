#!/bin/bash
# =============================================================================
# CosyVoice3 環境啟動腳本
# 負責：建立目錄、啟動 Docker 服務
# 模型下載由 docker-entrypoint.sh 在容器內處理
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# 建立必要目錄
# -----------------------------------------------------------------------------
create_directories() {
    log_info "建立必要目錄..."

    mkdir -p pretrained_models/CosyVoice3-0.5B
    mkdir -p data
    mkdir -p exp
    mkdir -p tensorboard

    log_success "目錄建立完成"
}

# -----------------------------------------------------------------------------
# 啟動 Docker 服務
# -----------------------------------------------------------------------------
start_docker() {
    log_info "啟動 Docker 服務..."

    # 檢查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安裝"
        exit 1
    fi

    # 檢查是否需要建立 image
    if ! docker images | grep -q "cosyvoice3"; then
        log_info "首次執行，建立 Docker image..."
        docker compose build
    fi

    # 啟動服務
    docker compose up -d

    log_success "Docker 服務已啟動"
}

# -----------------------------------------------------------------------------
# 顯示使用說明
# -----------------------------------------------------------------------------
show_usage() {
    echo ""
    echo -e "${BOLD}CosyVoice3 環境啟動完成${NC}"
    echo ""
    echo "=============================================="
    echo ""
    echo "進入訓練容器："
    echo "  docker compose exec cosyvoice3-train bash"
    echo ""
    echo "使用主控台（推薦）："
    echo "  ./run.sh help           # 顯示所有命令"
    echo "  ./run.sh status         # 檢查狀態"
    echo "  ./run.sh prepare        # 準備資料"
    echo "  ./run.sh extract        # 提取特徵"
    echo "  ./run.sh train          # 開始訓練"
    echo ""
    echo "TensorBoard："
    echo "  ./run.sh tensorboard    # 啟動（http://localhost:6007）"
    echo ""
    echo "停止服務："
    echo "  docker compose down"
    echo ""
    echo "=============================================="
}

# -----------------------------------------------------------------------------
# 主程式
# -----------------------------------------------------------------------------
main() {
    echo ""
    echo -e "${BOLD}CosyVoice3 環境啟動腳本${NC}"
    echo ""

    # 檢查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安裝"
        exit 1
    fi

    create_directories
    start_docker
    
    log_info "等待容器初始化（下載模型）..."
    sleep 3
    
    show_usage
}

# 執行
main "$@"
