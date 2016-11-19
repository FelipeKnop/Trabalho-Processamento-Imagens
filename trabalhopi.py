#!/usr/bin/env python3

import click
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage

@click.group(chain=True)
def cli():
    """Processa imagens.

    Exemplo:

        trabalhopi open -i digital1.jpg blur -s 3 save
        trabalhopi open -i digital1.jpg blur -s 5 display
    """
    pass

@cli.command('open')
@click.option('-i', '--image', 'image', type=click.Path(), help='Imagem a ser aberta.')
@click.pass_context
def open_cmd(ctx, image):
    """Carrega uma imagem para processamento."""
    try:
        click.echo('Abrindo "%s"' % image)
        img = misc.imread(image, False, 'RGB')
        ctx.obj['result'] = img
    except Exception as e:
        click.echo('Imagem não pode ser aberta "%s": %s' % (image, e), err=True)


@cli.command('blur')
@click.option('-s', '--sigma', default=1, type=int,
              help='Sigma do filtro gaussiano.', show_default=True)
@click.pass_context
def blur_cmd(ctx, sigma):
    """Borra a imagem usando o filtro gaussiano com SIGMA passado
    como parâmetro."""
    image = ctx.obj['result']
    try:
        click.echo('Aplicando filtro gaussiano com sigma %s' % sigma)
        blurred_img = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 1))
        ctx.obj['result'] = blurred_img
    except Exception as e:
        click.echo('Filtro não pode ser aplicado "%s": %s' % (image, e), err=True)


@cli.command('display')
@click.pass_context
def display_cmd(ctx):
    """Abre todas as imagens em um visualizador de imagens."""
    image = ctx.obj['result']
    click.echo('Exibindo imagem')
    plt.imshow(image)
    plt.show()
    # ctx.obj['result'] = image


# TODO(andre:2016-11-18): Permitir passar o nome do arquivo que será salvo
# TODO(andre:2016-11-18): Permitir especificar o formato a ser salvo? (o scipy
# deduz o formato pela a extensão do arquivo)
@cli.command('save')
@click.pass_context
def save_cmd(ctx):
    """Salva a imagem em um arquivo."""
    image = ctx.obj['result']
    click.echo('Salvando imagem')
    misc.imsave('output/temp.png', image)
    return image

# @cli.command('mse')
# @processor
# def mse_cmd(images)
#     """Calcula o erro quadratico medio entre duas imagens"""

if __name__ == "__main__":
    cli(obj={})
