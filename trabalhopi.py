#!/usr/bin/env python3

import click
# import matplotlib.pyplot as plt
# import scipy.misc
import scipy.ndimage
import scipy.signal
import PIL.Image
import PIL.ImageFilter

import numpy as np


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


# TODO(andre:2016-11-18): Permitir especificar o formato a ser salvo?
# (atualmente o formato é deduzido pela a extensão do arquivo)
@cli.command('save')
@click.option('-o', '--output', 'output', default="output/temp.png", type=click.Path())
@click.pass_context
def save(ctx, output):
    """Salva a imagem em um arquivo."""
    image = ctx.obj['result']
    click.echo('Salvando imagem')
    # scipy.misc.imsave('output/temp.png', image)
    image.save(output)


# BUG(andre:2016-11-20): Alguns comandos não estão funcionando com imagens
# preto e branco. Isso acontece porque as arrays delas não possuem três
# dimensões
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


# TODO(andre:2016-11-19): Verificar se as contas estão certas
@cli.command('mse')
@click.option('-r', '--reference', 'reference')
@click.pass_context
def mse(ctx, reference):
    """Calcula o erro quadrático médio entre duas imagens"""
    image = ctx.obj['result']

    try:
        click.echo('Abrindo "%s"' % reference)
        refimage = PIL.Image.open(reference).convert('RGB')
    except Exception as e:
        click.echo('Imagem não pode ser aberta "%s": %s' % (reference, e), err=True)

    np_image = np.array(image)
    np_refimage = np.array(refimage)

    size = np_image.shape
    refsize = np_refimage.shape

    if size != refsize:
        click.echo('Imagem base possui tamanho diferente da imagem de referência')
        click.echo('%s != %s' % (size, refsize))
        return

    diff = (np_image.astype(int) - np_refimage.astype(int))
    mse = (diff**2).mean(axis=(0, 1))
    click.echo("Erro quadrático medio: %s" % mse)


# TODO(andre:2016-11-19): Verificar se as contas estão certas
# BUG(andre:2016-11-19): Quando as imagens são iguais o ruido é igual a 0,
# gerando uma divisão por zero
@cli.command('snr')
@click.option('-r', '--reference', 'reference')
@click.pass_context
def snr(ctx, reference):
    """Calcula a razão sinal-ruído entre duas imagens"""
    image = ctx.obj['result']

    try:
        click.echo('Abrindo "%s"' % reference)
        refimage = PIL.Image.open(reference).convert('RGB')
    except Exception as e:
        click.echo('Imagem não pode ser aberta "%s": %s' % (reference, e), err=True)

    np_image = np.array(image)
    np_refimage = np.array(refimage)

    size = np_image.shape
    refsize = np_refimage.shape

    if size != refsize:
        click.echo('Imagem base possui tamanho diferente da imagem de referência')
        click.echo('%s != %s' % (size, refsize))
        return

    diff = (np_image.astype(int) - np_refimage.astype(int))

    signal = (np_image.astype(int)**2).sum(axis=(0, 1))
    noise = (diff**2).sum(axis=(0, 1))

    snr = 10 * (np.log(signal / noise) / np.log(10))
    click.echo("Razão sinal-ruído: %s" % snr)


@cli.command('gamma')
@click.option('-c', default=1)
@click.option('-g', '--gamma', 'gamma', default=1, type=click.FLOAT)
@click.pass_context
def gamma(ctx, c, gamma):
    """Aplica a transformação gamma: s = c*r^g."""
    image = ctx.obj['result']

    click.echo("Aplicando transformação gamma")

    np_image = np.array(image)
    np_image = (c * 255 * (np_image / 255)**gamma).astype(np.uint8)

    gamma_image = PIL.Image.fromarray(np_image)

    ctx.obj['result'] = gamma_image


@cli.command('histeq')
@click.option('-b', '--bins', 'bins', default=256)
@click.pass_context
def histeq(ctx, bins):
    """Aplica a equalização de histograma."""
    image = ctx.obj['result']

    click.echo("Aplicando equalização de histograma")

    ycbcr_image = image.convert('YCbCr')
    np_image = np.array(ycbcr_image)

    channel = np_image[:,:,0]

    histogram, bins_ar = np.histogram(channel, bins)
    cdf = histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]

    temp_image = np.interp(channel.flatten(), bins_ar[:-1], cdf)
    np_image[:,:,0] = temp_image.reshape(channel.shape)

    eq_image = PIL.Image.fromarray(np_image, 'YCbCr').convert('RGB')
    ctx.obj['result'] = eq_image


@cli.command('threshold')
@click.option('-t', '--threshold', 'threshold', default=128)
@click.option('-a', '--algorithm', 'algorithm', default=None, type=click.Choice(["otsu"]))
@click.pass_context
def threshold(ctx, threshold, algorithm):
    """Aplica a binarização por limiarização."""
    image = ctx.obj['result']

    click.echo("Aplicando binarização por limiarização")

    ycbcr_image = image.convert('YCbCr')
    np_image = np.array(ycbcr_image)

    if algorithm == "otsu":
        histogram, _ = np.histogram(np_image[:,:,0], 256)
        P1 = histogram.cumsum()

        av_intensity = np.arange(256) * histogram
        m = av_intensity.cumsum()

        best_variance = 0
        for i in range(256):
            wB = P1[i]
            if wB == 0:
                continue
            wF = (P1[-1] - wB)
            if wF == 0:
                break
            mB = m[i] / wB
            mF = (m[-1] - m[i]) / wF
            interclass_variance = wB * wF * (mB - mF)**2

            if best_variance < interclass_variance:
                best_variance = interclass_variance
                threshold = i

    mask = np_image[:,:,0] <= threshold
    np_image[:,:,0] = 255
    np_image[mask,0] = 0

    np_image[:,:,1] = 128
    np_image[:,:,2] = 128

    thresholded_image = PIL.Image.fromarray(np_image, 'YCbCr').convert('RGB')
    ctx.obj['result'] = thresholded_image


@cli.command('convolve')
@click.option('-a', '--axis', 'axis', default="xy", type=click.Choice(["x", "y", "xy"]))
@click.option('-m', '--mode', 'mode', default="reflect", type=click.Choice(["reflect", "constant", "nearest", "mirror", "wrap"]))
@click.option('-k', '--kernel', 'kernel', default=None, type=click.Choice(["gaussian", "prewitt", "sobel", "roberts", "laplace"]))
@click.option('-x', '--dimension-x', 'dimension_x', default=3)
@click.option('-y', '--dimension-y', 'dimension_y', default=3)
@click.option('-g', '--degree', 'degree', default=1)
@click.pass_context
def convolve(ctx, axis, mode, kernel, dimension_x, dimension_y, degree):
    image = ctx.obj['result']

    np_image = np.array(image, dtype='float64')

    if kernel == "prewitt":
        Px = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]])
        Py = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]])

        for c in range(0, np_image.shape[2]):
            dx = scipy.ndimage.filters.convolve(np_image[:,:,c], Px, mode=mode)
            dy = scipy.ndimage.filters.convolve(np_image[:,:,c], Py, mode=mode)
            np_image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(np_image[:,:,c])
            np_image[:,:,c] *= 255.0 / max_value

    elif kernel == "sobel":
        Sx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
        Sy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])

        for c in range(0, np_image.shape[2]):
            dx = scipy.ndimage.filters.convolve(np_image[:,:,c], Sx, mode=mode)
            dy = scipy.ndimage.filters.convolve(np_image[:,:,c], Sy, mode=mode)
            np_image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(np_image[:,:,c])
            np_image[:,:,c] *= 255.0 / max_value

    elif kernel == "roberts":
        Rx = np.array([
        [1, 0],
        [0, -1]])
        Ry = np.array([
        [0, 1],
        [-1, 0]])

        for c in range(0, np_image.shape[2]):
            dx = scipy.ndimage.filters.convolve(np_image[:,:,c], Rx, mode=mode)
            dy = scipy.ndimage.filters.convolve(np_image[:,:,c], Ry, mode=mode)
            np_image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(np_image[:,:,c])
            np_image[:,:,c] *= 255.0 / max_value

    elif kernel == "laplace":
        L = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]])

        for c in range(0, np_image.shape[2]):
            np_image[:,:,c] = scipy.ndimage.filters.convolve(np_image[:,:,c], L, mode=mode)
            # np_image[:,:,c] -= np.min(np_image[:,:,c])
            np_image[:,:,c] = np.absolute(np_image[:,:,c])
            max_value = np.max(np_image[:,:,c])
            np_image[:,:,c] *= 255.0 / max_value

    else:
        weights = box = np.ones((dimension_x, dimension_y))
        for g in range(0, degree - 1):
            weights = scipy.signal.convolve2d(weights, box, mode="same")

        weights = weights / weights.sum()

        if axis == "x":
            np_image = scipy.ndimage.filters.convolve1d(np_image, weights[0], axis=0, mode=mode)
        elif axis == "y":
            np_image = scipy.ndimage.filters.convolve1d(np_image, weights[0], axis=1, mode=mode)
        elif axis == "xy":
            for c in range(0, np_image.shape[2]):
                np_image[:,:,c] = scipy.ndimage.filters.convolve(np_image[:,:,c], weights, mode=mode)

    convolved_image = PIL.Image.fromarray(np_image.astype('uint8'))

    ctx.obj['result'] = convolved_image


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


if __name__ == "__main__":
    cli(obj={})
