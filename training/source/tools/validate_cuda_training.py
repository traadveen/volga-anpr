"""Exercise the real trainer on CUDA without modifying CPU artifacts."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ocr.crnn import DEFAULT_ALPHABET, CRNNRecognizer, create_crnn_model
from training.crnn_v2 import TrainConfig, OCRDatasetV2, seed_everything, train
from dataset.schema import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='models/phase_d/cuda_smoke')
    args = parser.parse_args()
    output = ROOT / 'reports/phase_d/cuda_validation.json'
    baseline = ROOT / 'models/ocr_crnn_nomeroff_clean.pt'
    original_hash = sha256(baseline)
    result = {'status': 'FAILED', 'executable': sys.executable,
              'torch_version': str(torch.__version__), 'torch_cuda_version': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(), 'cpu_environment_modified': False}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; large dataset/training gate remains closed')
        seed_everything(42)
        torch.set_num_threads(4)
        props = torch.cuda.get_device_properties(0)
        result.update({'GPU': props.name, 'VRAM_bytes': props.total_memory,
                       'compute_capability': [props.major, props.minor],
                       'compiled_architectures': torch.cuda.get_arch_list(), 'training_device': 'cuda'})
        result['driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
        tensor = torch.randn(128, 128, device='cuda')
        assert torch.isfinite(tensor @ tensor.T).all()
        model = create_crnn_model(DEFAULT_ALPHABET).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = next(model.parameters()).detach().clone()
        logits = model(torch.randn(4, 1, 32, 160, device='cuda'))
        loss = logits.square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert not torch.equal(before, next(model.parameters()))
        torch.cuda.synchronize()
        result['tensor_forward_backward_optimizer'] = {
            'passed': True, 'logits_shape': list(logits.shape), 'finite_loss': float(loss.detach())}
        del model, optimizer, tensor, before, logits, loss
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config = replace(TrainConfig.load(ROOT / 'configs/crnn_smoke.yaml'),
                         output_dir=args.run_dir, epochs=2,
                         samples_per_type=18, batch_size=18, device='cuda',
                         num_workers=0, max_training_seconds=240.)
        started = time.perf_counter()
        first = train(config, stop_after_epoch=1)
        resumed = train(config, resume=Path(config.output_dir) / 'last.pt')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        best_path = Path(config.output_dir) / 'best.pt'
        recognizer = CRNNRecognizer(str(best_path), device='cuda')
        data = OCRDatasetV2(config, config.validation_split)
        crop = data.input_for(0).crop
        prediction = recognizer.readtext(crop, detail=1)
        torch.cuda.synchronize()
        evidence = resumed['resume']
        checks = {
            'cuda_forward_backward_optimizer': True,
            'dataloader_all_types': set(first['train_types']) == {'type1', 'type1a', 'type1b'},
            'saved_last_best': best_path.exists() and (Path(config.output_dir)/'last.pt').exists(),
            'checkpoint_inference_load': bool(prediction),
            'resume_epoch': evidence['loaded_epoch'] == 1 and resumed['last_epoch'] == 2,
            'resume_optimizer': evidence['optimizer_step_after'] > evidence['optimizer_step_before'] > 0,
            'resume_scheduler': evidence['scheduler_after']['last_epoch'] == 2,
            'resume_rng': evidence['rng_restored'],
            'original_weights_intact': sha256(baseline) == original_hash,
            'isolated_environment': 'include-system-site-packages = false' in
                (Path(sys.prefix)/'pyvenv.cfg').read_text().lower(),
        }
        assert all(checks.values()), checks
        seconds = first['training_time_seconds'] + resumed['training_time_seconds']
        result.update({'status': 'PASS', 'checks': checks, 'batch_size': config.batch_size,
                       'training_samples': 54, 'validation_samples': len(data), 'epochs': 2,
                       'training_seconds': seconds, 'whole_check_seconds': elapsed,
                       'images_per_sec_including_validation_and_checkpoints': 108 / seconds,
                       'GPU_memory_usage': {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                            'peak_reserved_bytes': torch.cuda.max_memory_reserved()},
                       'first_epoch': first, 'resumed': resumed,
                       'inference_probe': prediction, 'baseline_sha256': original_hash})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({k: result[k] for k in ('status', 'cuda_available', 'torch_version')}, indent=2))


if __name__ == '__main__':
    main()
