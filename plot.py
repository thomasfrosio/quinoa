import numpy as np
from matplotlib import pyplot as plt
from typing import List, Iterator
import sys, glob, os

def check_batch_size_is_constant(data: List[List[str|float|int]]) -> int:
    it = iter(data)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('Invalid batched data. Varying batch size')
    return the_len


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def add_arange(lines: Iterator[str]):
    label = next(lines).strip().split('=', 1)
    x = next(lines).strip().split('=', 1)
    y = next(lines).strip().split('=', 1)
    assert(label[0] == 'label' and x[0] == 'x' and y[0] == 'y')

    data = []
    for batch in y[1].split(';'):
        data.append([float(i) for i in batch.split(',')])
    length = check_batch_size_is_constant(data)

    arange = x[1].split(',')
    start = float(arange[0])
    step = float(arange[1])
    stop = start + step * length
    # print(f'start={start}, step={step}, len={length}, stop={stop}')
    assert(len(arange) == 2)
    x = np.linspace(start, stop, length, endpoint=False)

    for batch in data:
        plt.plot(x, batch, label=label[1], alpha=0.75, linestyle='dotted', linewidth=0.75, markersize=5, marker='.')


def add_linspace(lines: Iterator[str], style):
    label = next(lines).strip().split('=', 1)
    x = next(lines).strip().split('=', 1)
    y = next(lines).strip().split('=', 1)
    assert(label[0] == 'label' and x[0] == 'x' and y[0] == 'y')

    data = []
    for batch in y[1].split(';'):
        data.append([float(i) for i in batch.split(',')])
    length = check_batch_size_is_constant(data)

    linspace = x[1].split(',')
    assert(len(linspace) == 3)
    x = np.linspace(float(linspace[0]), float(linspace[1]), length, endpoint=bool(linspace[2]))

    offset = 0
    for batch in data:
        plt.plot(x, np.array(batch) + offset, label=label[1], alpha=0.8, linestyle=style, linewidth=2)
        offset += 0.05
        # plt.show()


def add_scatter(lines: Iterator[str]):
    label = next(lines).strip().split('=', 1)
    x = next(lines).strip().split('=', 1)
    y = next(lines).strip().split('=', 1)
    assert(label[0] == 'label' and x[0] == 'x' and y[0] == 'y')

    data = []
    for batch in y[1].split(';'):
        data.append([float(i) for i in batch.split(',')])
    check_batch_size_is_constant(data)

    x_values = [float(i) for i in x[1].split(',')]
    for batch in data:
        plt.scatter(x_values, batch, label=label[1], s=10, alpha=0.5,)

@static_vars(labels=[], indices=[], tilts=[], xs=[], ys=[])
def add_scatter_shifts(lines: Iterator[str]):
    label = next(lines).strip().split('=', 1)
    indices = next(lines).strip().split('=')
    tilts = next(lines).strip().split('=')
    x = next(lines).strip().split('=')
    y = next(lines).strip().split('=')
    assert(label[0] == 'label' and indices[0] == 'indices' and tilts[0] == 'tilts' and x[0] == 'x' and y[0] == 'y')

    add_scatter_shifts.labels.append(label[1])
    add_scatter_shifts.indices.append([int(i) for i in indices[1].strip().split(',')])
    add_scatter_shifts.tilts.append([float(i) for i in tilts[1].strip().split(',')])
    add_scatter_shifts.xs.append([float(i) for i in x[1].strip().split(',')])
    add_scatter_shifts.ys.append([float(i) for i in y[1].strip().split(',')])


def add_scatter_shifts_final():
    if len(add_scatter_shifts.labels) == 0:
        return

    for label, x, y in zip(add_scatter_shifts.labels, add_scatter_shifts.xs, add_scatter_shifts.ys):
        plt.scatter(x, y, label=label)

    for index, tilt, ix, iy in zip(
            add_scatter_shifts.indices[0],
            add_scatter_shifts.tilts[0],
            add_scatter_shifts.xs[0],
            add_scatter_shifts.ys[0]):
        plt.annotate(index, (ix, iy), textcoords="offset points", xytext=(0, 3), ha='center', fontsize=7)
        plt.annotate(tilt, (ix, iy), textcoords="offset points", xytext=(0, -3), ha='center', fontsize=7)

    plt.plot([x for x in add_scatter_shifts.xs], [y for y in add_scatter_shifts.ys],
             color='black', linestyle='-', linewidth=1)


def plotter(filenames: List[str]):
    has_header = False
    for jj, filename in enumerate(filenames):
        style = ['solid', 'dashed', 'dotted']
        with open(filename) as file:
            lines = filter(lambda l: len(l) > 0 and not l.startswith('#'), (line.rstrip() for line in file))

            header = {}
            for i in range(4):
                key, value = next(lines).split('=', 1)
                header[key.strip()] = value.strip()
            expected_keys = ['uuid', 'title', 'xname', 'yname']
            if not all(key in expected_keys for key in header.keys()):
                raise ValueError(f'Invalid file header: keys={header.keys()}, expected_keys={expected_keys}')

            if not has_header:
                plt.title(header['title'])
                plt.xlabel(header['xname'])
                plt.ylabel(header['yname'])
                has_header = True

            while (line := next(lines, None)) is not None:
                if line.startswith('type='):
                    plot_type = line.split('=')[1]
                    if plot_type == 'scatter':
                        add_scatter(lines)
                    elif plot_type == 'arange':
                        add_arange(lines)
                    elif plot_type == 'linspace':
                        add_linspace(lines, style[jj])
                    elif plot_type == 'scatter-shifts':
                        add_scatter_shifts(lines)
                    else:
                        raise ValueError(f'Invalid plot type: {plot_type}')
            add_scatter_shifts_final()

    plt.style.use('tableau-colorblind10')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot.py <filename>...')

    filenames = []
    for filename in sys.argv[1:]:
        filenames.extend(glob.glob(os.path.expanduser(filename)))
    plotter(filenames)

    # for filename in filenames:
    #     plotter([filename])
