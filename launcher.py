import os
import sys

SCRIPT = './main.py'
METRICS_SCRIPT = './metrics_summary.py'


def print_help():
    print(f'Wrong numer of arguments.')
    print(f'Expected: python launcher.py <directory>')
    print(f'\t e.g: python launcher.py ./test_1')


if len(sys.argv) < 2:
    print_help()
    exit()

DIRECTORY = sys.argv[1]

if not os.path.exists(DIRECTORY):
    raise ValueError(f"Not found source directory {DIRECTORY}")

EXEC_CLASSIFY = [
    f'-dir={DIRECTORY}/AA_618_A59 -task=classify -cluster=UBC8 -model=all',
    f'-dir={DIRECTORY}/AA_618_A59 -task=classify -cluster=UBC17_a -model=all',
    f'-dir={DIRECTORY}/AA_618_A59 -task=classify -cluster=UBC17_b -model=all',
    f'-dir={DIRECTORY}/AA_635_A45 -task=classify -cluster=UBC106 -model=all',
    f'-dir={DIRECTORY}/AA_635_A45 -task=classify -cluster=UBC186 -model=all',
    f'-dir={DIRECTORY}/AA_635_A45 -task=classify -cluster=UBC570 -model=all',
    f'-dir={DIRECTORY}/AA_661_A118 -task=classify -cluster=UBC1004 -model=all',
    f'-dir={DIRECTORY}/AA_661_A118 -task=classify -cluster=UBC1015 -model=all',
    f'-dir={DIRECTORY}/AA_661_A118 -task=classify -cluster=UBC1565 -model=all'
]


for arg in EXEC_CLASSIFY:
    print (f'{arg}')
    items = arg.split(' ')
    sys.argv.clear()
    sys.argv.append(SCRIPT)
    for i in items:
        sys.argv.append(i)
    with open(SCRIPT) as f:
        exec(f.read())

# Launch metrics
print('Metrics summary')
sys.argv.clear()
sys.argv.append(METRICS_SCRIPT)
sys.argv.append(DIRECTORY)
with open(METRICS_SCRIPT) as f:
    exec(f.read())

print('Done')
