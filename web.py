import io
from flask import Flask, request, jsonify
import numpy as np
import whisperx

app = Flask(__name__)

device = "cuda"
compute_type = "float16"  # 改成 "int8" 减少显存占用 (可能降低精度)

# 加载全局模型
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

#不需要对齐与获取对话人，先注释掉
# model_a, metadata = None, None


# # 加载全局对齐模型
# def load_align_model(language_code):
#     global model_a, metadata
#     model_a, metadata = whisperx.load_align_model(
#         language_code=language_code, device=device
#     )


# # 加载全局扬声器识别管道
# diarize_model = whisperx.DiarizationPipeline(
#     model_name="pyannote/speaker-diarization-3.1",
#     # use_auth_token="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",
#     device=device,
# )


# # 语音转文字+对齐+获取对话人
# @app.route("/transcribe", methods=["POST"])
# def transcribe_audio():
#     if "audio" not in request.files:
#         return jsonify({"error": "No audio file provided"}), 400

#     audio_file = request.files["audio"]

#     try:
#         # 使用 io.BytesIO 直接加载音频文件
#         audio_file_bytes = io.BytesIO(audio_file.read())
#         audio = whisperx.audio.load_audio(audio_file_bytes)
#         result = model.transcribe(audio, batch_size=16)
#         language = result["language"]

#         # 对齐whisper输出
#         if model_a is None or metadata is None or metadata["language"] != language:
#             load_align_model(language_code=language)
#         result = whisperx.align(
#             result["segments"],
#             model_a,
#             metadata,
#             audio,
#             device,
#             return_char_alignments=False,
#         )

#         # 指定扬声器标签
#         diarize_segments = diarize_model(audio)
        
#         # 让段落带上speakerid
#         result = whisperx.assign_word_speakers(diarize_segments, result)

#         # 提取所需字段
#         segments = []
#         for segment in result["segments"]:
#             segments.append(
#                 {
#                     "start": segment["start"],
#                     "end": segment["end"],
#                     "speaker": segment["speaker"],
#                     "text": segment["text"],
#                 }
#             )

#         # 构建响应
#         response = {
#             "language": language,
#             "segments": segments,
#             "word_segments": result["word_segments"],
#         }

#         return jsonify(response), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# 语音转文字
@app.route("/asr", methods=["POST"])
def transcribe_audio_basic():
    content_type = request.headers.get("Content-Type")

    #如果是application/octet-stream，则处理原始 PCM 数据
    if content_type == "application/octet-stream":

        audio_data = request.get_data()

        if not audio_data:
            return jsonify({"error": "未提供音频数据"}), 500

        # 将 PCM 数据转换为 NumPy 数组
        np_pcm = np.frombuffer(audio_data, np.int16).flatten().astype(np.float32) / 32768.0
        audio = np_pcm
    
    #否则直接处理音频文件
    else:
        if "audio" not in request.files:
            return jsonify({"error": "未提供音频文件"}), 500

        audio_file = request.files["audio"]
        audio_file_bytes = io.BytesIO(audio_file.read())
        audio = whisperx.audio.load_audio(audio_file_bytes)

    try:
        # 调用 stt 获取转录结果
        result = model.transcribe(audio, batch_size=16)
        language = result["language"]

        # 合并所有 segment 中的 text
        full_text = "".join(segment["text"] for segment in result["segments"])

        response = {"language": language, "text": full_text}

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
