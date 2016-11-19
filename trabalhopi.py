#!/usr/bin/env python3

import click
# import matplotlib.pyplot as plt
# import scipy.misc
# import scipy.ndimage
import PIL.Image
import PIL.ImageFilter

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
def open(ctx, image):
    """Carrega uma imagem para processamento."""
    try:
        click.echo('Abrindo "%s"' % image)
        # img = scipy.misc.imread(image, False, 'RGB')
        img = PIL.Image.open(image).convert('RGB')
        ctx.obj['result'] = img
    except Exception as e:
        click.echo('Imagem não pode ser aberta "%s": %s' % (image, e), err=True)


@cli.command('blur')
@click.option('-r', '--radius', default=2, type=int,
              help='Raio do filtro gaussiano.', show_default=True)
@click.pass_context
def blur(ctx, radius):
    """Borra a imagem usando o filtro gaussiano."""
    image = ctx.obj['result']
    try:
        click.echo('Aplicando filtro gaussiano com raio %s' % radius)
        # blurred_image = scipy.ndimage.gaussian_filter(image, sigma=(sigma, sigma, 1))
        blurred_image = image.filter(PIL.ImageFilter.GaussianBlur(radius))
        ctx.obj['result'] = blurred_image
    except Exception as e:
        click.echo('Filtro não pode ser aplicado "%s": %s' % (image, e), err=True)


@cli.command('display')
@click.pass_context
def display(ctx):
    """Abre todas as imagens em um visualizador de imagens."""
    image = ctx.obj['result']
    click.echo('Exibindo imagem')
    # matplotlib.pyplot.imshow(image)
    # matplotlib.pyplot.show()
    image.show()
    # ctx.obj['result'] = image


# TODO(andre:2016-11-18): Permitir passar o nome do arquivo que será salvo
# TODO(andre:2016-11-18): Permitir especificar o formato a ser salvo? (o scipy
# deduz o formato pela a extensão do arquivo)
@cli.command('save')
@click.pass_context
def save(ctx):
    """Salva a imagem em um arquivo."""
    image = ctx.obj['result']
    click.echo('Salvando imagem')
    # scipy.misc.imsave('output/temp.png', image)
    image.save('output/temp.png')
    return image


@cli.command('convert')
@click.option('-m', '--mode', 'mode')
@click.pass_context
def colorspace(ctx, mode):
    image = ctx.obj['result']
    try:
        converted_image = image.convert(mode)
        ctx.obj['result'] = converted_image
    except Exception as e:
        click.echo('Imagem não pode ser convertida para "%s": %s' % (mode, e), err=True)

# @cli.command('mse')
# @processor
# def mse(images)
#     """Calcula o erro quadratico medio entre duas imagens"""

if __name__ == "__main__":
    cli(obj={})
