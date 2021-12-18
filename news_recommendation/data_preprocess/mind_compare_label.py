import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--removed',
    type=str,
    required=True,
)
parser.add_argument(
    '--full',
    type=str,
    required=True,
)
args = parser.parse_args()

with open(args.removed, 'r') as f:
    removed = f.read().strip().split('\n')

with open(args.full, 'r') as f:
    full = f.read().strip().split('\n')

assert len(removed) == len(full)

for r, f in tqdm(zip(removed, full)):
    r_begin, r_end = r.split('\t')[:-1], r.split('\t')[-1]
    f_begin, f_end = f.split('\t')[:-1], f.split('\t')[-1]
    assert r_begin == f_begin
    f_end = ' '.join([x.split('-')[0] for x in f_end.split(' ')])
    assert r_end == f_end
