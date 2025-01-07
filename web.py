from flask import Flask, request, jsonify
import whisperx
import gc
import tempfile
import os

app = Flask(__name__)

device = "cuda"
compute_type = "float16"  # 改成 "int8" 减少显存占用 (可能降低精度)

# 加载全局模型
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
model_a, metadata = None, None


# 加载全局对齐模型
def load_align_model(language_code):
    global model_a, metadata
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )


# 加载全局扬声器识别管道
diarize_model = whisperx.DiarizationPipeline(
    model_name="pyannote/speaker-diarization-3.1",
    # use_auth_token="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",
    device=device,
)

# 确保 tmp 目录存在
tmp_dir = os.path.join(os.getcwd(), "tmp")
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    # 构建临时文件路径
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".wav", dir=tmp_dir
    ) as temp_audio_file:
        audio_file_path = temp_audio_file.name
        audio_file.save(audio_file_path)

    try:
        # 分批用原始whisper转录
        audio = whisperx.load_audio(audio_file_path)
        result = model.transcribe(audio, batch_size=16)
        language = result["language"]
        print(result["segments"])
        # 对齐whisper输出
        if model_a is None or metadata is None or metadata["language"] != language:
            load_align_model(language_code=language)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        # 指定扬声器标签
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        # 段落现在已经带上了speakerid

        # 提取所需字段
        segments = []
        for segment in result["segments"]:
            segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": segment["speaker"],
                    "text": segment["text"],
                }
            )

        # 构建响应
        response = {
            "language": language,
            "segments": segments,
            "word_segments": result["word_segments"],
        }

        # 释放显存
        # del model
        # del model_a
        # gc.collect()
        # torch.cuda.empty_cache()

        # 删除临时文件
        os.remove(audio_file_path)

        return jsonify(response), 200

    except Exception as e:
        # 删除临时文件
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe_basic", methods=["POST"])
def transcribe_audio_basic():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    # 构建临时文件路径
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".wav", dir=tmp_dir
    ) as temp_audio_file:
        audio_file_path = temp_audio_file.name
        audio_file.save(audio_file_path)

    try:
        # 分批用原始whisper转录
        audio = whisperx.load_audio(audio_file_path)
        result = model.transcribe(audio, batch_size=16)
        language = result["language"]
        print(result["segments"])  # 对齐前

        # 提取所需字段
        segments = []
        for segment in result["segments"]:
            segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                }
            )

        # 构建响应
        response = {"language": language, "segments": segments}

        # 删除临时文件
        os.remove(audio_file_path)

        return jsonify(response), 200

    except Exception as e:
        # 删除临时文件
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
