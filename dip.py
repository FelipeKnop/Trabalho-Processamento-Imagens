#!/usr/bin/env python3

import math

import click
# import matplotlib.pyplot as plt
# import scipy.misc
import scipy.ndimage
import scipy.signal

import numpy as np

import utils
from commands import *


@click.group(chain=True)
def cli():
    """Processa imagens.

    Exemplo:

    \b
        dip open -i sample/digital1.jpg convolve -k gaussian -s 3 save
        dip open -i sample/digital1.jpg convolve -k gaussian -s 5 display
    """
    pass


@cli.command('open')
@click.option('-i', '--input', 'input', type=click.Path(), help='Imagem a ser aberta.')
@click.pass_context
def open_cmd(ctx, input):
    """Carrega uma imagem para processamento."""
    image = open(input)

    ctx.obj['image'] = image
    ctx.obj['img_mode'] = 'RGB'


@cli.command('display')
@click.option('-p', '--phase', 'phase', is_flag=True)
@click.option('-l', '--logarithm', 'logarithm', is_flag=True)
@click.option('-c', '--center', 'center', is_flag=True)
@click.pass_context
def display_cmd(ctx, phase, logarithm, center):
    """Abre todas as imagens em um visualizador de imagens."""
    image = ctx.obj['image']
    img_mode = ctx.obj['img_mode']

    display(image, img_mode, phase, logarithm, center)


# TODO(andre:2016-11-18): Permitir especificar o formato a ser salvo?
# (atualmente o formato é deduzido pela a extensão do arquivo)
@cli.command('save')
@click.option('-o', '--output', 'output', default='output/temp.png', type=click.Path())
@click.option('-p', '--phase', 'phase', is_flag=True)
@click.option('-l', '--logarithm', 'logarithm', is_flag=True)
@click.option('-c', '--center', 'center', is_flag=True)
@click.pass_context
def save_cmd(ctx, output, phase, logarithm, center):
    """Salva a imagem em um arquivo."""
    image = ctx.obj['image']
    img_mode = ctx.obj['img_mode']

    save(image, output, img_mode, phase, logarithm, center)


# BUG(andre:2016-11-20): Alguns comandos não estão funcionando com imagens
# preto e branco. Isso acontece porque as arrays delas não possuem três
# dimensões
@cli.command('convert')
@click.option('-m', '--mode', 'mode')
@click.pass_context
def convert_cmd(ctx, mode):
    """Converte a imagem"""
    image = ctx.obj['image']
    img_mode = ctx.obj['img_mode']

    converted_image = convert(image, img_mode, mode)

    ctx.obj['image'] = converted_image
    ctx.obj['img_mode'] = mode

    # try:
    #     converted_image = convert(image, img_mode, mode)
    #     ctx.obj['image'] = converted_image
    #     ctx.obj['img_mode'] = mode
    # except Exception as e:
    #     click.echo('Imagem não pode ser convertida para "%s": %s' % (mode, e), err=True)


# TODO(andre:2016-11-19): Verificar se as contas estão certas
@cli.command('mse')
@click.option('-r', '--reference', 'reference')
@click.pass_context
def mse_cmd(ctx, reference):
    """Calcula o erro quadrático médio entre duas imagens"""
    image = ctx.obj['image']
    image_ref = open(reference)

    result_mse = mse(image, image_ref)

    click.echo("Erro quadrático medio (MSE): %s" % result_mse)


# TODO(andre:2016-11-19): Verificar se as contas estão certas
# BUG(andre:2016-11-19): Quando as imagens são iguais o ruido é igual a 0,
# gerando uma divisão por zero
@cli.command('snr')
@click.option('-r', '--reference', 'reference')
@click.pass_context
def snr_cmd(ctx, reference):
    """Calcula a razão sinal-ruído entre duas imagens"""
    image = ctx.obj['image']
    image_ref = open(reference)

    result_snr = snr(image, image_ref)

    click.echo('Razão sinal-ruído (SNR): %s' % result_snr)


@cli.command('gamma')
@click.option('-c', default=1)
@click.option('-g', '--gamma', 'g', default=1, type=click.FLOAT)
@click.pass_context
def gamma_cmd(ctx, c, g):
    """Aplica a transformação gamma: s = c*r^g."""
    image = ctx.obj['image']

    image = gamma(image, c, g)

    ctx.obj['image'] = image


@cli.command('histeq')
@click.option('-b', '--bins', 'bins', default=256)
@click.pass_context
def histeq_cmd(ctx, bins):
    """Aplica a equalização de histograma."""
    image = ctx.obj['image']
    img_mode = ctx.obj['img_mode']

    image = histeq(image, img_mode, bins)

    ctx.obj['image'] = image


@cli.command('threshold')
@click.option('-t', '--threshold', 't', default=128)
@click.option('-a', '--algorithm', 'algorithm', default=None, type=click.Choice(['otsu']))
@click.pass_context
def threshold_cmd(ctx, t, algorithm):
    """Aplica a binarização por limiarização."""
    image = ctx.obj['image']
    img_mode = ctx.obj['img_mode']

    if algorithm == 'otsu':
        t = otsu_threshold(image, img_mode)

    image = threshold(image, img_mode, t)

    ctx.obj['image'] = image


@cli.command('convolve')
@click.option('-a', '--axis', 'axis', default='xy', type=click.Choice(['x', 'y', 'xy']))
@click.option('-m', '--mode', 'mode', default='reflect',
              type=click.Choice(['reflect', 'constant', 'nearest', 'mirror', 'wrap']))
@click.option('-k', '--kernel', 'kernel', default=None,
              type=click.Choice(['gaussian', 'prewitt', 'sobel', 'roberts', 'laplace']))
@click.option('-x', '--dimension-x', 'dimension_x', default=3)
@click.option('-y', '--dimension-y', 'dimension_y', default=3)
@click.option('-g', '--degree', 'degree', default=1)
@click.option('-s', '--sigma', 'sigma', default=1, type=click.FLOAT)
@click.pass_context
def convolve_cmd(ctx, axis, mode, kernel, dimension_x, dimension_y, degree, sigma):
    """Aplica o produto de convolução"""
    image = ctx.obj['image']

    click.echo('Aplicando produto de convolução')

    image = convolve(image, axis, mode, kernel, dimension_x, dimension_y, degree, sigma)

    ctx.obj['image'] = image


@cli.command('fourier')
@click.option('-i', '--inverse', 'inverse', is_flag=True)
@click.option('-n', '--numpy', 'numpy', is_flag=True)
@click.pass_context
def fourier_cmd(ctx, inverse, numpy):
    """Aplica a Tranformada Discreta de Fourier"""
    image = ctx.obj['image']
    img_mode = ctx.obj['img_mode']

    click.echo('Aplicando Transformada Discreta de Fourier')

    M, N, _ = image.shape

    if inverse:
        out_image = np.empty_like(image, dtype='uint8')
        out_image[:,:,1] = np.real(image[:,:,1])
        out_image[:,:,2] = np.real(image[:,:,2])

        if numpy:
            temp = np.fft.ifft2(image[:,:,0])
        else:
            temp = idft(image[:,:,0])

        out_image[:,:,0] = np.real(temp)

        ctx.obj['image'] = convert(out_image, 'YCbCr', 'RGB')
        ctx.obj['img_mode'] = 'RGB'

    else:
        image = convert(image, img_mode, 'YCbCr')

        out_image = np.empty_like(image, dtype='complex')
        out_image[:,:,1] = image[:,:,1]
        out_image[:,:,2] = image[:,:,2]

        if numpy:
            temp = np.fft.fft2(image[:,:,0])
        else:
            temp = dft(image[:,:,0])

        out_image[:,:,0] = temp

        ctx.obj['image'] = out_image
        ctx.obj['img_mode'] = 'spectrum'


@cli.command('product')
@click.option('-k', '--kernel', 'kernel', default=None,
              type=click.Choice(['low-pass', 'band-pass', 'high-pass']))
@click.option('-r', '--radius', 'radius', default=20)
@click.option('-s', '--size', 'size', default=5)
@click.pass_context
def product_cmd(ctx, kernel, radius, size):
    """Aplica o produto"""
    image = ctx.obj['image']

    image = product(image, kernel, radius, size)

    ctx.obj['image'] = image


# Sampling: http://www.cs.tau.ac.il/~dcor/Graphics/adv-slides/sampling05.pdf
@cli.command('resize')
@click.option('-s', '--scale', 'scale', default=1, type=click.FLOAT)
@click.option('-m', '--mode', 'mode', default=None,
              type=click.Choice(['area', 'nearest', 'bilinear', 'bicubic']))
@click.pass_context
def resize_cmd(ctx, scale, mode):
    image = ctx.obj['image']

    image = resize(image, scale, mode)

    ctx.obj['image'] = image


@cli.command('misc')
@click.option('-n', '--number', 'number', default=0)
@click.option('-a', '--par-a', 'par_a', default=0, type=click.FLOAT)
@click.option('-b', '--par-b', 'par_b', default=0, type=click.FLOAT)
@click.pass_context
def misc(ctx, number, par_a, par_b):
    image = ctx.obj['image']

    if number == 0:
        # par_a = w
        # par_b = sigma
        click.echo(par_a)
        click.echo(par_b)
        gaussian_image = convolve(image, 'xy', 'reflect', 'gaussian', 1, 1, 1, par_b)
        image = (1 + par_a) * image - par_a * gaussian_image
        image = np.clip(image, 0, 255).astype('uint8')

    ctx.obj['image'] = image

@cli.command('create')
@click.option('-n', '--number', 'number', default=0)
@click.option('-x', '--dimension-x', 'dimension_x', default=512)
@click.option('-y', '--dimension-y', 'dimension_y', default=512)
@click.option('-a', '--par-a', 'par_a', default=0, type=click.FLOAT)
@click.option('-b', '--par-b', 'par_b', default=0, type=click.FLOAT)
@click.pass_context
def misc(ctx, number, dimension_x, dimension_y, par_a, par_b):
    image = np.zeros((dimension_x, dimension_y, 3), 'uint8')
    mode = 'RGB'

    mid_x = int(dimension_x/2)
    mid_y = int(dimension_y/2)

    if number == 0:
        for j in range(-int(par_a/2), int((par_a+1)/2)):
            for i in range(-int(par_a/2), int((par_a+1)/2)):
                image[mid_x+i,mid_y+j,] = 255

    if number == 1:
        freq = par_a
        phase = par_b
        for j in range(dimension_y):
            for i in range(dimension_x):
                image[i,j,] = 127 * math.sin(utils.dist(mid_x, mid_y, i, j) * 2 * math.pi * freq + phase) + 127
                # image[i,j,] = 127 * math.sin(utils.dist(0, 0, i, j)*freq + phase) + 127

    ctx.obj['image'] = image
    ctx.obj['img_mode'] = mode


if __name__ == "__main__":
    cli(obj={})
