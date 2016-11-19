import click
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage

from structure import cli
from structure import generator
from structure import processor


@cli.command('open')
@click.option('-i', '--image', 'images', type=click.Path(),
              multiple=True, help='Imagem a ser aberta.')
@generator
def open_cmd(images):
    """Carrega uma ou mais imagens para processamento. O parâmetro de entrada
    pode ser especificado várias vezes para carregar mais de uma imagem."""
    for image in images:
        try:
            click.echo('Abrindo "%s"' % image)
            img = misc.imread(image, False, 'RGB')
            yield img
        except Exception as e:
            click.echo('Imagem não pode ser aberta "%s": %s' % (image, e), err=True)

@cli.command('blur')
@click.option('-s', '--sigma', default=1, type=int,
              help='Sigma do filtro gaussiano.', show_default=True)
@processor
def blur_cmd(images, sigma):
    """Borra a imagem usando o filtro gaussiano com SIGMA passado
    como parâmetro."""
    for image in images:
        try:
            click.echo('Aplicando filtro gaussiano com sigma %s' % sigma)
            blurred_img = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 1))
            yield blurred_img
        except Exception as e:
            click.echo('Filtro não pode ser aplicado "%s": %s' % (image, e), err=True)


@cli.command('display')
@processor
def display_cmd(images):
    """Abre todas as imagens em um visualizador de imagens."""
    for image in images:
        click.echo('Exibindo imagem')
        plt.imshow(image)
        plt.show()
        yield image


# TODO(andre:2016-11-18): Permitir passar o nome do arquivo que será salvo
# TODO(andre:2016-11-18): Permitir especificar o formato a ser salvo? (o scipy
# deduz o formato pela a extensão do arquivo)
@cli.command('save')
@processor
def display_cmd(images):
    """Salva a imagem em um arquivo."""
    for image in images:
        click.echo('Salvando imagem')
        misc.imsave('output/temp.png', image)
        yield image

if __name__ == "__main__":
    cli()
