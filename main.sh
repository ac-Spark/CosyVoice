#!/bin/bash
# =============================================================================
# CosyVoice3 訓練 Wrapper
# 簡化容器內執行官方訓練腳本
# =============================================================================

# 移除 set -e，因為很多命令（如 pkill）可能返回非零但不代表錯誤
# set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

CONTAINER_NAME="cosyvoice3-train"
OFFICIAL_SCRIPT="/workspace/CosyVoice/examples/libritts/cosyvoice3/run.sh"

# Conda 激活命令
CONDA_ACTIVATE="source /opt/conda/etc/profile.d/conda.sh && conda activate cosyvoice"

# -----------------------------------------------------------------------------
# 檢查容器
# -----------------------------------------------------------------------------
check_container() {
    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        echo -e "${RED}[ERROR]${NC} 容器未運行，請先執行: ./start.sh"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# 啟動 TensorBoard
# -----------------------------------------------------------------------------
start_tensorboard() {
    echo -e "${CYAN}[INFO]${NC} 啟動 TensorBoard..."
    
    # 先停止舊的
    docker exec "$CONTAINER_NAME" bash -c "pkill -9 -f tensorboard || true" > /dev/null 2>&1
    sleep 1
    
    # 啟動 TensorBoard（輸出到 docker logs）
    docker exec "$CONTAINER_NAME" bash -c "
        source /opt/conda/etc/profile.d/conda.sh && conda activate cosyvoice &&
        cd /workspace/CosyVoice/examples/libritts/cosyvoice3 &&
        tensorboard --logdir tensorboard --host 0.0.0.0 --port 6007 2>&1 | tee /tmp/tensorboard.log > /proc/1/fd/1 &
        disown
    " > /dev/null 2>&1
    
    sleep 2
    
    # 檢查是否成功啟動
    if docker exec "$CONTAINER_NAME" pgrep -f "tensorboard" > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} TensorBoard 已啟動: http://localhost:6007"
        return 0
    else
        echo -e "${YELLOW}[WARN]${NC} TensorBoard 啟動失敗"
        docker exec "$CONTAINER_NAME" cat /tmp/tensorboard.log 2>&1 | tail -5
        return 1
    fi
}

# -----------------------------------------------------------------------------
# 顯示說明
# -----------------------------------------------------------------------------
show_help() {
    echo ""
    echo -e "${BOLD}==================================${NC}"
    echo -e "${BOLD}   CosyVoice3 訓練管理工具${NC}"
    echo -e "${BOLD}==================================${NC}"
    echo ""
    echo -e "${BOLD}📋 快速開始${NC}"
    echo -e "  1. docker compose up -d              # 啟動容器"
    echo -e "  2. ./main.sh preprocess              # 資料前處理"
    echo -e "  3. ./main.sh train                   # 開始訓練"
    echo -e "  4. ./main.sh webui                   # 測試模型"
    echo ""
    echo -e "${BOLD}📊 資料處理${NC}"
    echo -e "  ${GREEN}preprocess${NC}     完整前處理 (prepare → embedding → token → parquet)"
    echo -e "  ${GREEN}prepare${NC}        準備資料列表 (wav.scp/text/utt2spk)"
    echo -e "  ${GREEN}embedding${NC}      提取說話人特徵"
    echo -e "  ${GREEN}token${NC}          提取語音 token"
    echo -e "  ${GREEN}parquet${NC}        轉換為 Parquet 格式"
    echo ""
    echo -e "${BOLD}🚀 訓練${NC}"
    echo -e "  ${GREEN}train${NC}          開始訓練 (背景執行，可關閉終端)"
    echo -e "  ${GREEN}resume${NC}         從最新 checkpoint 繼續訓練"
    echo ""
    echo -e "${BOLD}🎯 推理與測試${NC}"
    echo -e "  ${GREEN}webui${NC}          啟動 Gradio WebUI → http://localhost:7858"
    echo -e "                 支援模型：CosyVoice3-0.5B (預訓練) / MyFinetuned (你的模型)"
    echo -e "  ${GREEN}infer${NC}          進入 Python 推理環境"
    echo ""
    echo -e "${BOLD}🔧 工具${NC}"
    echo -e "  ${GREEN}shell${NC}          進入容器 bash"
    echo -e "  ${GREEN}deploy${NC}         部署訓練好的模型 (創建推理目錄)"
    echo ""
    echo -e "${BOLD}💡 提示${NC}"
    echo -e "  • 訓練/WebUI 都在背景執行，關閉終端不會中斷"
    echo -e "  • 訓練 checkpoint 自動保存在 ${CYAN}exp/cosyvoice3/llm/torch_ddp/${NC}"
    echo -e "  • WebUI 會自動清理 checkpoint 中的訓練 keys，可直接使用"
    echo -e "  • 查看訓練 log: ${CYAN}docker logs -f cosyvoice3-train${NC}"
    echo ""
}

# -----------------------------------------------------------------------------
# 主程式
# -----------------------------------------------------------------------------
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    local command=$1

    case "$command" in
        # 進入容器
        shell)
            check_container
            echo -e "${CYAN}[INFO]${NC} 進入容器..."
            docker exec -it "$CONTAINER_NAME" bash
            ;;

        # 查看日誌

        # 查看訓練 log

        # 查看 WebUI log

        # 查看 API log

        # API Server

        # 查看所有 log

        # 幫助
        help|--help|-h)
            show_help
            ;;

        # 下載數據

        # 準備資料
        prepare)
            check_container
            echo -e "${CYAN}[INFO]${NC} 準備資料 (Stage 0)"
            docker exec -it "$CONTAINER_NAME" bash -c "
                $CONDA_ACTIVATE
                cd /workspace/CosyVoice/examples/libritts/cosyvoice3
                sed -i 's|^stage=.*|stage=0|' run.sh
                sed -i 's|^stop_stage=.*|stop_stage=0|' run.sh
                sed -i 's|^data_dir=.*|data_dir=/workspace/CosyVoice/data|' run.sh
                ./run.sh
            "
            ;;

        # 提取 Embedding
        embedding)
            check_container
            echo -e "${CYAN}[INFO]${NC} 提取 Speaker Embedding (Stage 1)"
            docker exec -it "$CONTAINER_NAME" bash -c "
                $CONDA_ACTIVATE
                cd /workspace/CosyVoice/examples/libritts/cosyvoice3
                sed -i 's|^stage=.*|stage=1|' run.sh
                sed -i 's|^stop_stage=.*|stop_stage=1|' run.sh
                sed -i 's|^data_dir=.*|data_dir=/workspace/CosyVoice/data|' run.sh
                ./run.sh
            "
            ;;

        # 提取 Token
        token)
            check_container
            echo -e "${CYAN}[INFO]${NC} 提取 Speech Token (Stage 2)"
            docker exec -it "$CONTAINER_NAME" bash -c "
                $CONDA_ACTIVATE
                cd /workspace/CosyVoice/examples/libritts/cosyvoice3
                sed -i 's|^stage=.*|stage=2|' run.sh
                sed -i 's|^stop_stage=.*|stop_stage=2|' run.sh
                sed -i 's|^data_dir=.*|data_dir=/workspace/CosyVoice/data|' run.sh
                ./run.sh
            "
            ;;

        # 轉 Parquet
        parquet)
            check_container
            echo -e "${CYAN}[INFO]${NC} 轉換為 Parquet 格式 (Stage 3)"
            docker exec -it "$CONTAINER_NAME" bash -c "
                $CONDA_ACTIVATE
                cd /workspace/CosyVoice/examples/libritts/cosyvoice3
                sed -i 's|^stage=.*|stage=3|' run.sh
                sed -i 's|^stop_stage=.*|stop_stage=3|' run.sh
                sed -i 's|^data_dir=.*|data_dir=/workspace/CosyVoice/data|' run.sh
                ./run.sh
            "
            ;;

        # 完整前處理
        preprocess)
            check_container
            echo -e "${CYAN}[INFO]${NC} 執行完整資料前處理 (Stage 0-3)"
            docker exec -it "$CONTAINER_NAME" bash -c "
                $CONDA_ACTIVATE
                cd /workspace/CosyVoice/examples/libritts/cosyvoice3
                sed -i 's|^stage=.*|stage=0|' run.sh
                sed -i 's|^stop_stage=.*|stop_stage=3|' run.sh
                sed -i 's|^data_dir=.*|data_dir=/workspace/CosyVoice/data|' run.sh
                ./run.sh
            "
            ;;

        # 訓練
        train)
            check_container
            start_tensorboard
            echo -e "${CYAN}[INFO]${NC} 背景訓練啟動中..."
            docker exec "$CONTAINER_NAME" bash -c "
                $CONDA_ACTIVATE
                cd /workspace/CosyVoice/examples/libritts/cosyvoice3
                sed -i 's|^stage=.*|stage=5|' run.sh
                sed -i 's|^stop_stage=.*|stop_stage=5|' run.sh
                sed -i 's|^data_dir=.*|data_dir=/workspace/CosyVoice/data|' run.sh
                > /tmp/train.log
                nohup bash -c './run.sh 2>&1 | tee -a /tmp/train.log > /proc/1/fd/1' &
                echo \$! > /tmp/train.pid
            "
            echo -e "${GREEN}[OK]${NC} 訓練已在背景執行"
            echo -e "${YELLOW}[提示]${NC} Ctrl+C 或關閉終端不會中斷訓練"
            echo -e "${CYAN}[INFO]${NC} 正在顯示 log (Ctrl+C 停止顯示)..."
            echo ""
            docker exec "$CONTAINER_NAME" tail -f /tmp/train.log
            ;;

        # 接續訓練（訓練全部三種模型：llm, flow, hifigan）
        resume)
            check_container

            # 顯示各模型 checkpoint 狀態
            echo -e "${CYAN}[INFO]${NC} 檢查各模型 checkpoint 狀態..."
            docker exec "$CONTAINER_NAME" bash -c "
                cd /workspace/CosyVoice/examples/libritts/cosyvoice3
                for model in llm flow hifigan; do
                    ckpt=\$(ls -t exp/cosyvoice3/\$model/torch_ddp/epoch_*_whole.pt 2>/dev/null | head -1)
                    if [ -n \"\$ckpt\" ]; then
                        echo \"  \$model: \$ckpt\"
                    else
                        echo \"  \$model: init.pt (從頭訓練)\"
                    fi
                done
            "

            # 啟動訓練（包含 TensorBoard）
            echo -e "${CYAN}[INFO]${NC} 準備訓練環境並啟動 TensorBoard..."
            
            # 創建啟動腳本（訓練全部三種模型：llm, flow, hifigan）
            docker exec "$CONTAINER_NAME" bash -c "cat > /tmp/start_training.sh << 'TRAINING_SCRIPT'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cosyvoice
cd /workspace/CosyVoice/examples/libritts/cosyvoice3

# 建立資料列表
> data/train.data.list
cat data/train/parquet/data.list >> data/train.data.list
> data/dev.data.list
cat data/test/parquet/data.list >> data/dev.data.list

# 啟動 TensorBoard
pkill -9 -f tensorboard || true
nohup tensorboard --logdir tensorboard --host 0.0.0.0 --port 6007 </dev/null >/tmp/tensorboard.log 2>&1 &

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Starting training for all models\" | tee /tmp/train.log > /proc/1/fd/1
echo \"TensorBoard: http://localhost:6007\" | tee -a /tmp/train.log > /proc/1/fd/1

# 依序訓練三種模型
for model in llm flow hifigan; do
    echo \"\" | tee -a /tmp/train.log > /proc/1/fd/1
    echo \"========================================\" | tee -a /tmp/train.log > /proc/1/fd/1
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Training \$model model\" | tee -a /tmp/train.log > /proc/1/fd/1
    echo \"========================================\" | tee -a /tmp/train.log > /proc/1/fd/1

    # 尋找最新 checkpoint
    CKPT=\$(ls -t exp/cosyvoice3/\$model/torch_ddp/epoch_*_whole.pt 2>/dev/null | head -1)
    if [ -z \"\$CKPT\" ]; then
        CKPT=\"exp/cosyvoice3/\$model/torch_ddp/init.pt\"
    fi
    echo \"Using checkpoint: \$CKPT\" | tee -a /tmp/train.log > /proc/1/fd/1

    torchrun --nnodes=1 --nproc_per_node=2 \
        --rdzv_id=1986 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
        cosyvoice/bin/train.py \
        --train_engine torch_ddp \
        --config conf/cosyvoice3.yaml \
        --train_data data/train.data.list \
        --cv_data data/dev.data.list \
        --qwen_pretrain_path ../../../pretrained_models/CosyVoice3-0.5B/CosyVoice-BlankEN \
        --model \$model \
        --checkpoint \$CKPT \
        --model_dir \$(pwd)/exp/cosyvoice3/\$model/torch_ddp \
        --tensorboard_dir \$(pwd)/tensorboard/cosyvoice3/\$model/torch_ddp \
        --ddp.dist_backend nccl \
        --num_workers 2 \
        --prefetch 100 \
        --pin_memory \
        --use_amp \
        --deepspeed_config ./conf/ds_stage2.json \
        --deepspeed.save_states model+optimizer 2>&1 | tee -a /tmp/train.log > /proc/1/fd/1

    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Finished training \$model\" | tee -a /tmp/train.log > /proc/1/fd/1
done

echo \"\" | tee -a /tmp/train.log > /proc/1/fd/1
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] All training completed!\" | tee -a /tmp/train.log > /proc/1/fd/1
TRAINING_SCRIPT
chmod +x /tmp/start_training.sh"
            
            # 使用 nohup 在背景執行腳本（輸出到 docker logs）
            docker exec "$CONTAINER_NAME" bash -c "
                nohup /tmp/start_training.sh </dev/null >/proc/1/fd/1 2>&1 &
                echo \$! > /tmp/training.pid
            "
            
            sleep 3
            echo -e "${GREEN}[OK]${NC} 訓練已在背景啟動"
            
            # 檢查是否成功啟動
            if docker exec "$CONTAINER_NAME" pgrep -f "torchrun.*train.py" > /dev/null 2>&1; then
                echo -e "${GREEN}[OK]${NC} 訓練進程運行中"
            else
                echo -e "${YELLOW}[WARN]${NC} 訓練尚未完全啟動，請稍候..."
            fi
            
            if docker exec "$CONTAINER_NAME" pgrep -f "tensorboard" > /dev/null 2>&1; then
                echo -e "${GREEN}[OK]${NC} TensorBoard: http://localhost:6007"
            else
                echo -e "${YELLOW}[WARN]${NC} TensorBoard 尚未啟動"
            fi
            
            echo ""
            echo -e "${YELLOW}[提示]${NC} 使用 ${GREEN}./main.sh logs-train${NC} 查看訓練 log"
            echo -e "${YELLOW}[提示]${NC} Ctrl+C 或關閉終端不會中斷訓練"
            echo ""
            echo -e "${CYAN}[INFO]${NC} 正在顯示即時 log (Ctrl+C 停止顯示，不影響訓練)..."
            echo ""
            sleep 2
            
            # 捕獲中斷信號，正常退出
            trap 'echo ""; echo -e "${GREEN}[INFO]${NC} Log 顯示已停止，訓練繼續在背景運行"; exit 0' INT TERM
            docker exec "$CONTAINER_NAME" tail -f /tmp/train.log 2>/dev/null || true
            ;;

        # 推理環境
        infer)
            check_container
            echo -e "${CYAN}[INFO]${NC} 啟動推理環境..."
            docker exec -it "$CONTAINER_NAME" bash -c "
                $CONDA_ACTIVATE
                cd /workspace/CosyVoice
                python3 -c \"
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# 載入模型（使用訓練後的 checkpoint）
print('載入模型中...')
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice3-0.5B')
print('模型載入完成！')
print()
print('使用方式：')
print('  # Zero-shot 語音合成')
print('  for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav)):')
print('      torchaudio.save(f\\\"output_{i}.wav\\\", j[\\\"tts_speech\\\"], cosyvoice.sample_rate)')
print()
print('變數：cosyvoice, torchaudio')
\"
                python3 -i -c \"
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice3-0.5B')
\"
            "
            ;;

        # TensorBoard
        tensorboard)
            check_container
            start_tensorboard
            ;;

        # WebUI
        webui)
            check_container
            echo -e "${CYAN}[INFO]${NC} 啟動 Gradio WebUI (背景模式)..."

            # 先停止舊的進程
            docker exec "$CONTAINER_NAME" bash -c "pkill -9 -f 'webui.py' || true" > /dev/null 2>&1

            # 等待 port 釋放
            for i in {1..10}; do
                if docker exec "$CONTAINER_NAME" python3 -c "import socket; s=socket.socket(); s.bind(('0.0.0.0', 7858)); s.close()" 2>/dev/null; then
                    break
                fi
                sleep 1
            done

            # 創建啟動腳本
            docker exec "$CONTAINER_NAME" bash -c "cat > /tmp/start_webui.sh << 'WEBUI_SCRIPT'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cosyvoice
cd /workspace/CosyVoice
python3 webui.py --model_dir pretrained_models/CosyVoice3-0.5B --port 7858 2>&1 | tee /tmp/webui.log > /proc/1/fd/1
WEBUI_SCRIPT
chmod +x /tmp/start_webui.sh"
            
            # 使用 nohup 在背景執行
            docker exec "$CONTAINER_NAME" bash -c "
                nohup /tmp/start_webui.sh </dev/null >/dev/null 2>&1 &
                echo \$! > /tmp/webui.pid
            "

            sleep 3
            echo -e "${GREEN}[OK]${NC} WebUI 已在背景啟動"
            echo -e "${CYAN}[INFO]${NC} 模型載入中（約需 30-60 秒）..."
            echo ""
            echo -e "${YELLOW}[提示]${NC} WebUI: http://localhost:7858"
            echo -e "${YELLOW}[提示]${NC} 使用 ${GREEN}./main.sh logs-webui${NC} 查看即時 log"
            echo -e "${YELLOW}[提示]${NC} 使用 ${GREEN}docker exec cosyvoice3-train pgrep -f webui${NC} 檢查進程"
            ;;

        # 模型平均

        # 導出模型

        # 部署訓練好的模型
        deploy)
            check_container
            echo -e "${CYAN}[INFO]${NC} 部署訓練好的模型..."

            # 建立新的模型目錄
            DEPLOY_DIR="pretrained_models/CosyVoice3-0.5B-Finetuned"
            docker exec "$CONTAINER_NAME" bash -c "
                cd /workspace/CosyVoice

                # 建立目錄
                mkdir -p $DEPLOY_DIR

                # 複製基礎檔案（tokenizer, config 等）
                cp pretrained_models/CosyVoice3-0.5B/*.yaml $DEPLOY_DIR/ 2>/dev/null || true
                cp pretrained_models/CosyVoice3-0.5B/*.onnx $DEPLOY_DIR/ 2>/dev/null || true
                cp -r pretrained_models/CosyVoice3-0.5B/CosyVoice-BlankEN $DEPLOY_DIR/ 2>/dev/null || true

                echo '處理模型檔案...'

                # 清理 checkpoint 中的 epoch/step key 的 Python 腳本
                cat > /tmp/clean_checkpoint.py << 'PYSCRIPT'
import torch
import sys

ckpt_path = sys.argv[1]
output_path = sys.argv[2]

ckpt = torch.load(ckpt_path, map_location='cpu')

# 移除訓練用的 metadata
keys_to_remove = ['epoch', 'step', 'optimizer', 'scheduler', 'scaler']
for key in keys_to_remove:
    if key in ckpt:
        print(f'  移除 key: {key}')
        del ckpt[key]

torch.save(ckpt, output_path)
print(f'  已儲存: {output_path}')
PYSCRIPT

                source /opt/conda/etc/profile.d/conda.sh && conda activate cosyvoice

                # LLM - 使用訓練好的或原始的
                LLM_CKPT=\$(ls -t examples/libritts/cosyvoice3/exp/cosyvoice3/llm/torch_ddp/epoch_*_whole.pt 2>/dev/null | head -1)
                if [ -n \"\$LLM_CKPT\" ]; then
                    echo \"  LLM: \$LLM_CKPT\"
                    python3 /tmp/clean_checkpoint.py \"\$LLM_CKPT\" $DEPLOY_DIR/llm.pt
                else
                    echo \"  LLM: 使用原始模型\"
                    cp pretrained_models/CosyVoice3-0.5B/llm.pt $DEPLOY_DIR/
                fi

                # Flow - 使用訓練好的或原始的
                FLOW_CKPT=\$(ls -t examples/libritts/cosyvoice3/exp/cosyvoice3/flow/torch_ddp/epoch_*_whole.pt 2>/dev/null | head -1)
                if [ -n \"\$FLOW_CKPT\" ]; then
                    echo \"  Flow: \$FLOW_CKPT\"
                    python3 /tmp/clean_checkpoint.py \"\$FLOW_CKPT\" $DEPLOY_DIR/flow.pt
                else
                    echo \"  Flow: 使用原始模型\"
                    cp pretrained_models/CosyVoice3-0.5B/flow.pt $DEPLOY_DIR/
                fi

                # HiFiGAN/HiFT - 使用訓練好的或原始的
                HIFI_CKPT=\$(ls -t examples/libritts/cosyvoice3/exp/cosyvoice3/hifigan/torch_ddp/epoch_*_whole.pt 2>/dev/null | head -1)
                if [ -n \"\$HIFI_CKPT\" ]; then
                    echo \"  HiFT: \$HIFI_CKPT\"
                    python3 /tmp/clean_checkpoint.py \"\$HIFI_CKPT\" $DEPLOY_DIR/hift.pt
                else
                    echo \"  HiFT: 使用原始模型\"
                    cp pretrained_models/CosyVoice3-0.5B/hift.pt $DEPLOY_DIR/
                fi

                echo ''
                echo '部署完成！'
                ls -la $DEPLOY_DIR/*.pt
            "

            echo ""
            echo -e "${GREEN}[OK]${NC} 模型已部署到: $DEPLOY_DIR"
            echo -e "${YELLOW}[提示]${NC} 使用方式："
            echo -e "  Python: AutoModel(model_dir='$DEPLOY_DIR')"
            ;;

        # 使用微調模型的 WebUI

        *)
            echo -e "${RED}[ERROR]${NC} 未知命令: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 執行
main "$@"
