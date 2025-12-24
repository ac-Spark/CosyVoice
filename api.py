"""
CosyVoice3 API Server

FastAPI 服務，提供 RESTful API 進行語音合成。
"""
import argparse
import io
import os
import sys
import tempfile
import time
import random
from typing import Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed


# 全域變數
cosyvoice: Optional[object] = None
model_dir: str = "pretrained_models/CosyVoice3-0.5B-Finetuned-1222"
available_models: dict = {}
sft_spk: list = []


class ModelReloadRequest(BaseModel):
    """模型切換請求"""
    model_name: str


def scan_available_models():
    """掃描可用的模型目錄"""
    import glob
    models = {}

    # 掃描 pretrained_models 目錄
    for model_path in glob.glob('pretrained_models/*/llm.pt'):
        dir_name = os.path.dirname(model_path)
        display_name = os.path.basename(dir_name)
        models[display_name] = dir_name

    # 掃描訓練目錄中的 checkpoint（只顯示最新的）
    checkpoints = sorted(glob.glob('examples/libritts/cosyvoice3/exp/cosyvoice3/llm/torch_ddp/epoch_*_whole.pt'))
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        epoch = os.path.basename(latest_ckpt).replace('_whole.pt', '')
        display_name = f'訓練中-{epoch}'
        models[display_name] = latest_ckpt

    return models


def initialize_tts():
    """初始化 TTS 模型"""
    global cosyvoice, sft_spk, available_models, model_dir

    available_models = scan_available_models()

    # 檢查預設模型是否存在，若不存在則嘗試其他模型
    if not os.path.exists(os.path.join(model_dir, "llm.pt")):
        logging.warning(f"預設模型 {model_dir} 不存在")
        # 嘗試使用備用模型
        fallback_model = "pretrained_models/CosyVoice3-0.5B"
        if os.path.exists(os.path.join(fallback_model, "llm.pt")):
            model_dir = fallback_model
            logging.info(f"使用備用模型: {model_dir}")
        elif available_models:
            # 使用第一個可用模型
            first_model = list(available_models.values())[0]
            model_dir = first_model
            logging.info(f"使用第一個可用模型: {model_dir}")
        else:
            raise RuntimeError("找不到任何可用的模型")

    logging.info(f"正在初始化 CosyVoice3 引擎 (模型: {model_dir})...")
    try:
        cosyvoice = AutoModel(model_dir=model_dir)
        sft_spk = cosyvoice.list_available_spks()
        if len(sft_spk) == 0:
            sft_spk = ['']
        logging.info(f"TTS 引擎初始化完成，可用音色: {sft_spk}")
    except Exception as e:
        logging.error(f"初始化失敗: {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """伺服器啟動與關閉時的生命週期管理"""
    initialize_tts()
    yield


# 初始化 FastAPI
app = FastAPI(
    title="CosyVoice3 API",
    description="CosyVoice3 語音合成 API 服務",
    version="1.0.0",
    lifespan=lifespan
)

# 掛載靜態檔案
os.makedirs("static", exist_ok=True)
os.makedirs("outputs/api", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/asset", StaticFiles(directory="asset"), name="asset")


@app.get("/")
async def read_index():
    """首頁"""
    return FileResponse('static/index.html')


@app.get("/health")
async def health_check():
    """健康檢查"""
    return {"status": "ok", "model_loaded": cosyvoice is not None}


@app.get("/models")
def list_models():
    """列出所有可用的模型"""
    models = []
    for name, path in available_models.items():
        models.append({
            "name": name,
            "path": path,
            "type": "finetune" if "訓練中" in name or "Finetuned" in name else "base"
        })

    current = os.path.basename(model_dir) if cosyvoice is not None else "None"
    return {
        "models": models,
        "current_model": current,
        "speakers": sft_spk
    }


@app.post("/model/reload")
def reload_model(request: ModelReloadRequest):
    """動態切換模型"""
    global cosyvoice, sft_spk, model_dir

    target_name = request.model_name
    if target_name not in available_models:
        raise HTTPException(status_code=404, detail=f"模型 {target_name} 不存在")

    target_path = available_models[target_name]

    try:
        logging.info(f"正在切換至模型: {target_path}")

        # 釋放舊模型
        del cosyvoice
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 載入新模型
        cosyvoice = AutoModel(model_dir=target_path)
        sft_spk = cosyvoice.list_available_spks()
        if len(sft_spk) == 0:
            sft_spk = ['']
        model_dir = target_path

        return {
            "status": "success",
            "message": f"已切換至模型: {target_name}",
            "speakers": sft_spk
        }

    except Exception as e:
        logging.error(f"模型切換失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers")
def list_speakers():
    """列出當前模型的可用音色"""
    return {"speakers": sft_spk}


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    mode: str = Form("sft"),  # sft, zero_shot, cross_lingual, instruct
    speaker: str = Form(None),
    prompt_audio: UploadFile = File(None),
    prompt_audio_path: str = Form(None),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    seed: int = Form(-1),
    speed: float = Form(1.0),
    stream: bool = Form(False),
):
    """
    執行語音合成。

    Args:
        text: 要合成的文字
        mode: 推理模式 (sft/zero_shot/cross_lingual/instruct)
        speaker: 預訓練音色名稱 (sft/instruct 模式需要)
        prompt_audio: 上傳的參考音訊 (zero_shot/cross_lingual 模式需要)
        prompt_audio_path: 伺服器上的參考音訊路徑
        prompt_text: 參考音訊的文字內容 (zero_shot 模式需要)
        instruct_text: 指令文字 (instruct 模式需要)
        seed: 隨機種子 (-1 表示隨機)
        speed: 語速調整 (0.5-2.0)
        stream: 是否使用流式推理
    """
    if not cosyvoice:
        raise HTTPException(status_code=503, detail="TTS 引擎尚未初始化")

    # 設定隨機種子
    if seed == -1:
        actual_seed = random.randint(1, 100000000)
    else:
        actual_seed = seed
    set_all_random_seed(actual_seed)
    logging.info(f"使用 Seed: {actual_seed}")

    # 處理參考音訊
    temp_audio_file = None
    target_prompt_path = None

    if mode in ["zero_shot", "cross_lingual"]:
        if prompt_audio:
            suffix = os.path.splitext(prompt_audio.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await prompt_audio.read()
                tmp.write(content)
                target_prompt_path = tmp.name
                temp_audio_file = tmp.name
        elif prompt_audio_path:
            if not os.path.exists(prompt_audio_path):
                raise HTTPException(status_code=404, detail=f"參考音訊 {prompt_audio_path} 不存在")
            target_prompt_path = prompt_audio_path
        else:
            raise HTTPException(status_code=400, detail="zero_shot/cross_lingual 模式需要提供參考音訊")

    # 驗證參數
    if mode == "sft":
        if not speaker or speaker not in sft_spk:
            if sft_spk:
                speaker = sft_spk[0]
            else:
                raise HTTPException(status_code=400, detail="沒有可用的預訓練音色")
    elif mode == "zero_shot":
        if not prompt_text:
            raise HTTPException(status_code=400, detail="zero_shot 模式需要提供 prompt_text")
    elif mode == "instruct":
        if not instruct_text:
            raise HTTPException(status_code=400, detail="instruct 模式需要提供 instruct_text")
        if not speaker or speaker not in sft_spk:
            if sft_spk:
                speaker = sft_spk[0]

    try:
        # 準備輸出
        output_dir = "outputs/api"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"tts_{int(time.time())}_{os.urandom(2).hex()}.wav"
        output_path = os.path.join(output_dir, output_filename)

        logging.info(f"收到請求: text='{text[:30]}...', mode={mode}, speaker={speaker}")

        # 執行推理
        audio_data = None

        if mode == "sft":
            for result in cosyvoice.inference_sft(text, speaker, stream=stream, speed=speed):
                audio_data = result['tts_speech'].numpy().flatten()

        elif mode == "zero_shot":
            for result in cosyvoice.inference_zero_shot(text, prompt_text, target_prompt_path, stream=stream, speed=speed):
                audio_data = result['tts_speech'].numpy().flatten()

        elif mode == "cross_lingual":
            for result in cosyvoice.inference_cross_lingual(text, target_prompt_path, stream=stream, speed=speed):
                audio_data = result['tts_speech'].numpy().flatten()

        elif mode == "instruct":
            for result in cosyvoice.inference_instruct(text, speaker, instruct_text, stream=stream, speed=speed):
                audio_data = result['tts_speech'].numpy().flatten()
        else:
            raise HTTPException(status_code=400, detail=f"不支援的推理模式: {mode}")

        if audio_data is None:
            raise RuntimeError("音訊生成失敗")

        # 儲存音訊
        import torchaudio
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        torchaudio.save(output_path, audio_tensor, cosyvoice.sample_rate)

        # 回傳音訊串流
        def iterfile():
            with open(output_path, mode="rb") as f:
                yield from f

        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}",
                "X-Seed": str(actual_seed)
            }
        )

    except Exception as e:
        logging.error(f"TTS 錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice3 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="監聽位址")
    parser.add_argument("--port", type=int, default=7857, help="監聽埠號")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice3-0.5B-Finetuned-1222", help="模型目錄")

    args = parser.parse_args()
    model_dir = args.model_dir

    logging.info(f"啟動 API Server 於 {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
