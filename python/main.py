import argparse
import os
from whisper import Whisper


def get_args():
    parser = argparse.ArgumentParser(
        prog="whisper",
        description="Run Whisper on input audio file"
    )
    parser.add_argument("--wav", "-w", type=str, required=True, help="Input audio file")
    parser.add_argument("--model_type", "-t", type=str, choices=["tiny", "base", "small", "large", "large-v3", "turbo"], required=True, help="model type, only support tiny, base and small currently")
    parser.add_argument("--model_path", "-p", type=str, required=False, default="../models/models-ax650", help="model path for *.axmodel, tokens.txt, positional_embedding.bin")
    parser.add_argument("--language", "-l", type=str, required=False, default="zh", help="Target language, support en, zh, ja, and others. See languages.py for more options.")
    parser.add_argument("--task", type=str, required=False, choices=["translate", "transcribe"], default="transcribe")
    return parser.parse_args()


def print_args(args):
    print(f"wav: {args.wav}")
    print(f"model_type: {args.model_type}")
    print(f"model_path: {args.model_path}")
    print(f"language: {args.language}")
    print(f"task: {args.task}")


def main():
    args = get_args()
    print_args(args)

    # Check wav existence
    wav_path = args.wav
    assert os.path.exists(wav_path), f"{wav_path} NOT exist"

    model = Whisper(args.model_type, args.model_path, args.language, args.task)
    
    print("\n预测结果:")
    print(model.run(wav_path))

if __name__ == "__main__":
    main()
