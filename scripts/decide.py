"""python scripts/decide.py --ckpt out/decision/decision.pth --input examples/choice.json"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.inference import DecisionPredictor


def main():
    parser = argparse.ArgumentParser(description='输入 state / question / primitive，输出决策概率 JSON')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--tokenizer', default='model')
    parser.add_argument('--input', required=True, help='JSON 文件；- 表示标准输入')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--calibration', help='与权重和词表配套的 T.json；仅用于相应校准域')
    parser.add_argument('--max_state_tokens', type=int, default=512)
    parser.add_argument('--chunk', type=int, default=16)
    args = parser.parse_args()
    try:
        if args.input == '-':
            request = json.load(sys.stdin)
        else:
            with open(args.input, encoding='utf-8-sig') as f:
                request = json.load(f)
        predictor = DecisionPredictor(args.ckpt, args.tokenizer, args.device, args.calibration)
        print(json.dumps(predictor.predict(request, args.max_state_tokens, args.chunk), ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
