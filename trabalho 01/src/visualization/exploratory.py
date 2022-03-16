import sys
sys.path.append('src')

import click
import matplotlib.pyplot as plt
from data import read_processed_data


def exploratory_visualization(dframe):
    fig = plt.figure(figsize=(15,6))
    dframe.groupby('main_product').complaint_text.count().plot.bar(ylim=0)
    return fig

@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file):
    print('Plotting distribution...')

    dframe = read_processed_data(input_file)
    plot = exploratory_visualization(dframe)
    plot.savefig(output_file, bbox_inches = "tight")

    print('Complete!')


if __name__ == '__main__':
    main()