import subprocess
import requests
import io

# 定义音频文件路径
audio_file_path = "/disk6/dly/bobby-whisperX/tmp/chunwang.wav"  # 替换为你的音频文件路径
pcm_file_path = "/disk6/dly/bobby-whisperX/tmp/audio.pcm"  # 临时保存 PCM 文件的路径

# 使用 ffmpeg 将音频文件转换为 16 位单声道 16000 Hz 的 PCM 数据
def convert_to_pcm(audio_file_path, pcm_file_path):
    command = [
        "ffmpeg",
        "-i", audio_file_path,
        "-f", "s16le",
        "-ac", "1",
        "-ar", "16000",
        pcm_file_path
    ]
    subprocess.run(command, check=True)

# 读取 PCM 文件并发送到后端
def send_pcm_to_asr(pcm_file_path):
    with open(pcm_file_path, "rb") as f:
        audio_data = f.read()

    url = "http://183.131.7.9:5000/asr"
    headers = {"Content-Type": "application/octet-stream"}

    response = requests.post(url, headers=headers, data=audio_data)

    if response.status_code == 200:
        result = response.json()
        print("Transcription Result:")
        print(result)
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

# 主函数
def main():
    # 转换音频文件为 PCM 格式
    convert_to_pcm(audio_file_path, pcm_file_path)
    print(f"Converted audio to PCM: {pcm_file_path}")

    # 发送 PCM 数据到 ASR 接口
    send_pcm_to_asr(pcm_file_path)

if __name__ == "__main__":
    main()