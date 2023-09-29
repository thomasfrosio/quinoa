import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from typing import List

import typer

cli = typer.Typer()


def add_file_(filename: Path):
    with open(filename) as file:
        for line in file:
            values = [float(i) for i in line.split(',')]
            plt.plot(np.arange(len(values)), values)


def add_file_xy_(filename_x: Path, filename_y: Path):
    with open(filename_x) as file_x:
        with open(filename_y) as file_y:
            for line_x, line_y in zip(file_x, file_y):
                values_x = [float(i) for i in line_x.split(',')]
                values_y = [float(i) for i in line_y.split(',')]
                plt.plot(values_x, values_y, 'r+')


@cli.command(no_args_is_help=True)
def plotter(
        filenames: List[Path],
        # x: Path = Path(),
        # y: Path = Path(),
):
    if filenames is not None:
        for filename in filenames:
            add_file_(filename)
        plt.show()

    # if x.is_file() and y.is_file():
    #     add_file_xy_(x, y)
    #     plt.show()


if __name__ == '__main__':
    cli()
    # plotter([Path('/home/thomas/Projects/quinoa/tests/ribo_ctf/ctf/rotational_average_fitting_i.jpg')])
