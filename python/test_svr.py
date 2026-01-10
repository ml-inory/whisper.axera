import requests


def transcribe_audio(
    server_url: str,
    wav_path: str,
    model_type: str = "tiny",
    model_path: str = "../models/models-ax650",
    language: str = "zh",
    task: str = "transcribe",
):
    url = f"{server_url.rstrip('/')}/asr"

    files = {
        "wav": open(wav_path, "rb"),
    }

    data = {
        "model_type": model_type,
        "model_path": model_path,
        "language": language,
        "task": task,
    }

    print(f"Sending request to: {url}")

    response = requests.post(url, files=files, data=data)
    if response.status_code != 200:
        print("❌ Error:", response.text)
        return None

    result = response.json()
    print("服务器返回结果：")
    print(result)

    return result


if __name__ == "__main__":
    # 你的服务器地址
    SERVER = "http://127.0.0.1:8000"

    # 本地 wav 文件路径
    WAV = "../demo.wav"

    transcribe_audio(SERVER, WAV)
