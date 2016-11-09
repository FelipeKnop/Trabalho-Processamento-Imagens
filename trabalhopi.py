import click
import matplotlib.pyplot as plt
from scipy import misc


@click.command('open')
@click.option('-i', '--image', help='Imagem a ser processada.')
def mostra(image):
    face = misc.imread(image)
    plt.imshow(face)
    plt.show()


if __name__ == "__main__":
    mostra()