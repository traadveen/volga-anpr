"""Reproducible, weight-free probes of original deskew and CRNN geometry."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/legacy_probes.json'))
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from torch import Tensor, nn

    torch.manual_seed(42)
    torch.set_num_threads(4)
    ns = {'cv2': cv2, 'np': np}
    extract(ROOT / 'run.py', {'_deskew_estimate', '_rotate'}, ns)
    base = np.full((160, 600), 255, np.uint8)
    for y in [35, 65, 95, 125]:
        cv2.line(base, (20, y), (580, y), 0, 2)
    deskew = []
    for rotation in [-10, 10]:
        tilted = ns['_rotate'](base, rotation)
        angle = ns['_deskew_estimate'](tilted)
        corrected = ns['_rotate'](tilted, -angle)
        deskew.append(dict(input_rotation=rotation, estimated_line_angle=angle,
                           after_legacy_deskew=ns['_deskew_estimate'](corrected)))
    scope = {'torch': torch, 'nn': nn, 'Tensor': Tensor}
    extract(ROOT / 'ocr/crnn.py', {'CRNN', 'create_crnn_model'}, scope)
    model = scope['create_crnn_model']('0123456789ABCEHKMOPTXY').eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 1, 32, 160))
        try:
            model(torch.zeros(1, 1, 48, 160))
            height_error = None
        except RuntimeError as error:
            height_error = str(error)
        narrow = model(torch.zeros(1, 1, 32, 32))
        # 8 target symbols require at least 8 frames; the network returns only 7.
        loss = nn.CTCLoss(blank=22, zero_infinity=True)(
            narrow.log_softmax(2), torch.tensor([10, 0, 0, 1, 11, 12, 7, 7]),
            torch.tensor([len(narrow)]), torch.tensor([8]))
    result = dict(seed=42, torch=torch.__version__, opencv=cv2.__version__, deskew=deskew,
                  crnn=dict(parameters=sum(p.numel() for p in model.parameters()),
                            default_output=list(logits.shape), height48_error=height_error,
                            width32_frames=len(narrow), impossible_ctc_loss=float(loss)),
                  meaning='Controlled software probes, not model accuracy or an ablation result')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
