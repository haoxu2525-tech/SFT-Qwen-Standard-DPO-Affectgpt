"""Run the existing inference and scoring commands after a deployed DPO job."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-status', type=Path, required=True)
    p.add_argument('--media-status', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--inference-python', required=True)
    p.add_argument('--scoring-python', required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    args = p.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.log_dir / 'evaluation-status.json'

    def status(stage, **details):
        value = dict(stage=stage, pid=os.getpid(),
                     utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        temp.replace(state_path)
        print(json.dumps(value), flush=True)

    def wait_for(path, stage):
        status(stage)
        while True:
            if path.exists():
                value = json.loads(path.read_text(encoding='utf-8'))
                if value['stage'] == 'completed':
                    return
                if value['stage'] == 'failed':
                    raise RuntimeError(f'{path.name}: {value.get("error", "job failed")}')
            time.sleep(30)

    try:
        wait_for(args.training_status, 'waiting_for_training_and_export')
        wait_for(args.media_status, 'waiting_for_original_evaluation_media')
        pipeline = Path(__file__).resolve().with_name('run_pipeline.py')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        for stage, python in [('infer', args.inference_python), ('evaluate', args.scoring_python)]:
            command = [python, '-u', str(pipeline), stage, '--runtime', str(args.runtime),
                       '--dataset-root', str(args.dataset_root)]
            if stage == 'infer':
                command += ['--checkpoints', str(args.checkpoints), '--checkpoint-epoch', '1']
            log_path = args.log_dir / (stage + '.log')
            with log_path.open('x', encoding='utf-8') as log:
                proc = subprocess.Popen(command, cwd=pipeline.parent, env=env,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                status(stage, child_pid=proc.pid, log=str(log_path))
                code = proc.wait()
                if code:
                    raise RuntimeError(f'{stage} exited {code}; see {log_path}')
        status('completed', runtime=str(args.runtime))
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
