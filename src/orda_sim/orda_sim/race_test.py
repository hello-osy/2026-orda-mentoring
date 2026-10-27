"""Start a fresh simulator, answer and independent completion evaluator."""
from datetime import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time

from .common import project_path
from .simulator import stop_process


def run(config):
    settings = config['race_test']
    output = project_path(config, settings['output']) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True)
    answer = project_path(config, 'src/orda_sim/orda_sim/answer.py')
    (output / 'answer_snapshot.txt').write_text(answer.read_text())
    provenance = {'answer_sha256': hashlib.sha256(answer.read_bytes()).hexdigest()}
    weights = project_path(config, config['answer']['detector']['weights'])
    if config['answer']['detector']['enabled'] and weights.is_file():
        provenance['weights_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    processes = {}
    handles = []

    def start(action):
        handle = (output / (action + '.log')).open('w')
        handles.append(handle)
        processes[action] = subprocess.Popen(
            [sys.executable, '-m', 'orda_sim.entry', action, config['_config_path']],
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_message(action, message):
        deadline = time.monotonic() + settings['startup_timeout']
        while time.monotonic() < deadline:
            if processes[action].poll() is not None:
                raise RuntimeError(f'{action} 시작 실패: {output / (action + ".log")}')
            if message in (output / (action + '.log')).read_text(errors='replace'):
                return
            time.sleep(0.1)
        raise RuntimeError(f'{action} 시작 시간 초과: {output}')

    outcome = {'passed': False, 'logs': str(output)}
    try:
        start('simulator')
        wait_message('simulator', 'Gazebo 시작:')
        start('evaluate')
        wait_message('evaluate', '평가 시작:')
        start('answer')
        print(f'완주 시험 시작 · 로그: {output}', flush=True)
        deadline = time.monotonic() + settings['wall_timeout']
        while processes['evaluate'].poll() is None:
            for action in ['simulator', 'answer']:
                if processes[action].poll() is not None:
                    raise RuntimeError(f'{action} 중단: {output / (action + ".log")}')
            if time.monotonic() > deadline:
                raise RuntimeError('완주 시험 실행 시간 초과')
            time.sleep(0.2)
        outcome['passed'] = processes['evaluate'].returncode == 0
        outcome['evaluation_log'] = str(output / 'evaluate.log')
    except KeyboardInterrupt:
        outcome['reason'] = 'interrupted'
    except Exception as error:
        outcome['reason'] = str(error)
    finally:
        # Stop the driver first. Simulator shutdown then cleans its own children.
        for action in ['answer', 'evaluate', 'simulator']:
            if action in processes:
                stop_process(processes[action])
        for handle in handles:
            handle.close()
        # Keep renderer/bridge evidence before the next simulator overwrites it.
        for name in ['gazebo', 'bridge']:
            log = project_path(config, config['paths']['logs']) / (name + '.log')
            if log.is_file():
                shutil.copyfile(log, output / (name + '.log'))
        outcome['process_exit_codes'] = {name: process.returncode for name, process in processes.items()}
        (output / 'result.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2))
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
    if not outcome['passed']:
        raise SystemExit(1)
