#!/usr/bin/env python3
"""Run a delay/noise sweep for the virtual gimbal experiment."""
import argparse, pathlib, re, subprocess, tempfile

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='config/rune_gimbal_virtual.yaml')
    p.add_argument('--frames', type=int, default=1000)
    p.add_argument('--delays', nargs='*', type=float, default=[0.0, .01, .02, .05])
    p.add_argument('--noises', nargs='*', type=float, default=[0.0, .1, .2, .5])
    p.add_argument('--exe', default='./build/rune_aim')
    a = p.parse_args(); base = pathlib.Path(a.config).read_text()
    print('delay_s noise_deg init_ok aim_ok mean_deg max_deg')
    for delay in a.delays:
        for noise in a.noises:
            text = re.sub(r'transform_delay:\s*[^\s]+', f'transform_delay: {delay}', base)
            text = re.sub(r'transform_noise:\s*[^\s]+', f'transform_noise: {noise}', text)
            with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
                f.write(text); f.flush()
                out = subprocess.run([a.exe, '-c', f.name, '--no-display', '--max-frames', str(a.frames)], text=True, capture_output=True).stdout
            vals = {}
            for line in re.findall(r'^\s*(?:frames|init_ok|aim_ok|predict err).*$', out, re.M):
                key, val = line.strip().split(':', 1)
                key = key.strip()
                if key == 'predict err':
                    m = re.search(r'mean=.*\(([-+0-9.]+) deg\).*max=.*\(([-+0-9.]+) deg\)', val)
                    if m: vals['mean'], vals['max'] = m.groups()
                else: vals[key] = val.split(':', 1)[-1].strip()
            print(delay, noise, vals.get('init_ok','?'), vals.get('aim_ok','?'), vals.get('mean','?'), vals.get('max','?'))
if __name__ == '__main__': main()
