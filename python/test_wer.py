import argparse
import os
import logging
import re
from whisper import Whisper


def setup_logging():
    """配置日志系统，同时输出到控制台和文件"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "test_wer.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class AIShellDataset:
    def __init__(self, gt_path: str):
        """
        初始化数据集
        
        Args:
            json_path: voice.json文件的路径
        """
        self.gt_path = gt_path
        self.dataset_dir = os.path.dirname(gt_path)
        self.voice_dir = os.path.join(self.dataset_dir, "aishell_S0764")
        
        # 检查必要文件和文件夹是否存在
        assert os.path.exists(gt_path), f"gt文件不存在: {gt_path}"
        assert os.path.exists(self.voice_dir), f"aishell_S0764文件夹不存在: {self.voice_dir}"
        
        # 加载数据
        self.data = []
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                audio_path, gt = line.split(" ")
                audio_path = os.path.join(self.voice_dir, audio_path + ".wav")
                self.data.append({"audio_path": audio_path, "gt": gt})

        # 使用logging而不是print
        logger = logging.getLogger()
        logger.info(f"加载了 {len(self.data)} 条数据")
    
    def __iter__(self):
        """返回迭代器"""
        self.index = 0
        return self
    
    def __next__(self):
        """返回下一个数据项"""
        if self.index >= len(self.data):
            raise StopIteration
        
        item = self.data[self.index]
        audio_path = item["audio_path"]
        ground_truth = item["gt"]
        
        self.index += 1
        return audio_path, ground_truth
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    

class CommonVoiceDataset:
    """Common Voice数据集解析器"""
    
    def __init__(self, tsv_path: str):
        """
        初始化数据集
        
        Args:
            json_path: voice.json文件的路径
        """
        self.tsv_path = tsv_path
        self.dataset_dir = os.path.dirname(tsv_path)
        self.voice_dir = os.path.join(self.dataset_dir, "clips")
        
        # 检查必要文件和文件夹是否存在
        assert os.path.exists(tsv_path), f"{tsv_path}文件不存在: {tsv_path}"
        assert os.path.exists(self.voice_dir), f"voice文件夹不存在: {self.voice_dir}"
        
        # 加载JSON数据
        self.data = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                line = line.strip()
                splits = line.split("\t")
                audio_path = splits[1]
                gt = splits[2]
                audio_path = os.path.join(self.voice_dir, audio_path)
                self.data.append({"audio_path": audio_path, "gt": gt})
        
        # 使用logging而不是print
        logger = logging.getLogger()
        logger.info(f"加载了 {len(self.data)} 条数据")
    
    def __iter__(self):
        """返回迭代器"""
        self.index = 0
        return self
    
    def __next__(self):
        """返回下一个数据项"""
        if self.index >= len(self.data):
            raise StopIteration
        
        item = self.data[self.index]
        audio_path = item["audio_path"]
        ground_truth = item["gt"]
        
        self.index += 1
        return audio_path, ground_truth
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

def get_args():
    parser = argparse.ArgumentParser(
        prog="whisper",
        description="Test WER on dataset"
    )
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["aishell", "common_voice"], help="Test dataset")
    parser.add_argument("--gt_path", "-g", type=str, required=True, help="Test dataset ground truth file")
    parser.add_argument("--max_num", type=int, default=-1, required=False, help="Maximum test data num")
    parser.add_argument("--model_type", "-t", type=str, choices=["tiny", "base", "small", "large", "large-v3", "turbo"], required=True, help="model type, only support tiny, base and small currently")
    parser.add_argument("--model_path", "-p", type=str, required=False, default="../models/models-ax650", help="model path for *.axmodel, tokens.txt, positional_embedding.bin")
    parser.add_argument("--language", "-l", type=str, required=False, default="zh", help="Target language, support en, zh, ja, and others. See languages.py for more options.")
    return parser.parse_args()


def print_args(args):
    logger = logging.getLogger()
    logger.info(f"dataset: {args.dataset}")
    logger.info(f"gt_path: {args.gt_path}")
    logger.info(f"max_num: {args.max_num}")
    logger.info(f"model_type: {args.model_type}")
    logger.info(f"model_path: {args.model_path}")
    logger.info(f"language: {args.language}")


def min_distance(word1: str, word2: str) -> int:
 
    row = len(word1) + 1
    column = len(word2) + 1
 
    cache = [ [0]*column for i in range(row) ]
 
    for i in range(row):
        for j in range(column):
 
            if i ==0 and j ==0:
                cache[i][j] = 0
            elif i == 0 and j!=0:
                cache[i][j] = j
            elif j == 0 and i!=0:
                cache[i][j] = i
            else:
                if word1[i-1] == word2[j-1]:
                    cache[i][j] = cache[i-1][j-1]
                else:
                    replace = cache[i-1][j-1] + 1
                    insert = cache[i][j-1] + 1
                    remove = cache[i-1][j] + 1
 
                    cache[i][j] = min(replace, insert, remove)
 
    return cache[row-1][column-1]


def remove_punctuation(text):
    # 定义正则表达式模式，匹配所有标点符号
    # 这个模式包括常见的标点符号和中文标点
    pattern = r'[^\w\s]|_'
    
    # 使用sub方法将所有匹配的标点符号替换为空字符串
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text


def main():
    # 设置日志系统
    logger = setup_logging()

    args = get_args()
    print_args(args)

    dataset_type = args.dataset.lower()
    if dataset_type == "aishell":
        dataset = AIShellDataset(args.gt_path)
    elif dataset_type == "common_voice":
        dataset = CommonVoiceDataset(args.gt_path)
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    max_num = args.max_num

    # Load model
    model = Whisper(args.model_type, args.model_path, args.language, "transcribe")

    # Iterate over dataset
    references = []
    hyp = []
    all_character_error_num = 0
    all_character_num = 0
    wer_file = open("wer.txt", "w")
    max_data_num = max_num if max_num > 0 else len(dataset)
    for n, (audio_path, reference) in enumerate(dataset):
        hypothesis = model.run(audio_path)

        hypothesis = remove_punctuation(hypothesis)
        reference = remove_punctuation(reference)

        character_error_num = min_distance(reference, hypothesis)
        character_num = len(reference)
        character_error_rate = character_error_num / character_num * 100

        all_character_error_num += character_error_num
        all_character_num += character_num

        hyp.append(hypothesis)
        references.append(reference)
        
        line_content = f"({n+1}/{max_data_num}) {os.path.basename(audio_path)}  gt: {reference}  predict: {hypothesis}  WER: {character_error_rate}%"
        wer_file.write(line_content + "\n")
        logger.info(line_content)

        if n + 1 >= max_data_num:
            break

    total_character_error_rate = all_character_error_num / all_character_num * 100

    logger.info(f"Total WER: {total_character_error_rate}%")
    wer_file.write(f"Total WER: {total_character_error_rate}%")
    wer_file.close()

if __name__ == "__main__":
    main()
