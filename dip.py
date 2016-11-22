#!/usr/bin/env python3

import math

import click
# import matplotlib.pyplot as plt
# import scipy.misc
import scipy.ndimage
import scipy.signal
import PIL.Image

import numpy as np

import utils


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
def open(ctx, input):
    """Carrega uma imagem para processamento."""
    try:
        click.echo('Abrindo "%s"' % input)
        # scipy_image = scipy.misc.imread(input, False, 'RGB')
        pil_image = PIL.Image.open(input).convert('RGB')
        image = np.array(pil_image)
        ctx.obj['image'] = image
        ctx.obj['img_mode'] = 'RGB'
    except Exception as e:
        click.echo('Imagem não pode ser aberta "%s": %s' % (input, e), err=True)


@cli.command('display')
@click.option('-p', '--phase', 'phase', is_flag=True)
@click.option('-l', '--logarithm', 'logarithm', is_flag=True)
@click.option('-c', '--center', 'center', is_flag=True)
@click.pass_context
def display(ctx, phase, logarithm, center):
    """Abre todas as imagens em um visualizador de imagens."""
    image = ctx.obj['image']
    img_mode = ctx.obj['img_mode']

    click.echo('Exibindo imagem')

    # matplotlib.pyplot.imshow(image)
    # matplotlib.pyplot.show()
    if img_mode == 'spectrum':
        M, N, _ = image.shape

        if phase:
            show_image = np.arctan2(np.imag(image), np.real(image))[:,:,0]
            show_image += math.pi
            show_image *= 255 / (2 * math.pi)

        else:
            show_image = np.sqrt((np.real(image)**2 + np.imag(image)**2))[:,:,0]

            # NOTE(andre:2016-11-21): http://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
            if logarithm:
                show_image = (255 / math.log(1 + np.abs(show_image).max())) * np.log(1 + np.abs(show_image))

        if center:
            show_image = np.roll(show_image, int(M/2), 0)
            show_image = np.roll(show_image, int(N/2), 1)

        pil_image = PIL.Image.fromarray(show_image)
        pil_image.show()
    else:
        pil_image = PIL.Image.fromarray(image, img_mode)
        pil_image.show()


# TODO(andre:2016-11-18): Permitir especificar o formato a ser salvo?
# (atualmente o formato é deduzido pela a extensão do arquivo)
@cli.command('save')
@click.option('-o', '--output', 'output', default="output/temp.png", type=click.Path())
@click.pass_context
def save(ctx, output):
    """Salva a imagem em um arquivo."""
    image = ctx.obj['image']
    click.echo('Salvando imagem')
    # scipy.misc.imsave('output/temp.png', image)
    pil_image = PIL.Image.fromarray(image)
    pil_image.save(output)


# BUG(andre:2016-11-20): Alguns comandos não estão funcionando com imagens
# preto e branco. Isso acontece porque as arrays delas não possuem três
# dimensões
@cli.command('convert')
@click.option('-m', '--mode', 'mode')
@click.pass_context
def convert(ctx, mode):
    """Converte a imagem"""
    image = ctx.obj['image']
    img_mode = ctx.obj['img_mode']
    try:
        converted_image = utils.convert_image(image, mode, from_mode=img_mode)
        ctx.obj['image'] = converted_image
        ctx.obj['img_mode'] = mode
    except Exception as e:
        click.echo('Imagem não pode ser convertida para "%s": %s' % (mode, e), err=True)


# TODO(andre:2016-11-19): Verificar se as contas estão certas
@cli.command('mse')
@click.option('-r', '--reference', 'reference')
@click.pass_context
def mse(ctx, reference):
    """Calcula o erro quadrático médio entre duas imagens"""
    image = ctx.obj['image']

    try:
        click.echo('Abrindo "%s"' % reference)
        pil_refimage = PIL.Image.open(reference).convert('RGB')
        refimage = np.array(pil_refimage)
    except Exception as e:
        click.echo('Imagem não pode ser aberta "%s": %s' % (reference, e), err=True)

    size = image.shape
    refsize = refimage.shape

    if size != refsize:
        click.echo('Imagem base possui tamanho diferente da imagem de referência')
        click.echo('%s != %s' % (size, refsize))
        return

    diff = (image.astype(int) - refimage.astype(int))
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
    image = ctx.obj['image']

    try:
        click.echo('Abrindo "%s"' % reference)
        pil_refimage = PIL.Image.open(reference).convert('RGB')
        refimage = np.array(pil_refimage)
    except Exception as e:
        click.echo('Imagem não pode ser aberta "%s": %s' % (reference, e), err=True)

    size = image.shape
    refsize = refimage.shape

    if size != refsize:
        click.echo('Imagem base possui tamanho diferente da imagem de referência')
        click.echo('%s != %s' % (size, refsize))
        return

    diff = (image.astype(int) - refimage.astype(int))

    signal = (image.astype(int)**2).sum(axis=(0, 1))
    noise = (diff**2).sum(axis=(0, 1))

    snr = 10 * (np.log(signal / noise) / np.log(10))
    click.echo("Razão sinal-ruído: %s" % snr)


@cli.command('gamma')
@click.option('-c', default=1)
@click.option('-g', '--gamma', 'gamma', default=1, type=click.FLOAT)
@click.pass_context
def gamma(ctx, c, gamma):
    """Aplica a transformação gamma: s = c*r^g."""
    image = ctx.obj['image']

    click.echo("Aplicando transformação gamma")

    image = (c * 255 * (image / 255)**gamma).astype(np.uint8)

    ctx.obj['image'] = image


@cli.command('histeq')
@click.option('-b', '--bins', 'bins', default=256)
@click.pass_context
def histeq(ctx, bins):
    """Aplica a equalização de histograma."""
    image = ctx.obj['image']

    click.echo("Aplicando equalização de histograma")

    image = utils.convert_image(image, 'YCbCr')

    channel = image[:,:,0]

    histogram, bins_ar = np.histogram(channel, bins)
    cdf = histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]

    temp_image = np.interp(channel.flatten(), bins_ar[:-1], cdf)
    image[:,:,0] = temp_image.reshape(channel.shape)

    ctx.obj['image'] = utils.convert_image(image, 'RGB', from_mode='YCbCr')


@cli.command('threshold')
@click.option('-t', '--threshold', 'threshold', default=128)
@click.option('-a', '--algorithm', 'algorithm', default=None, type=click.Choice(['otsu']))
@click.pass_context
def threshold(ctx, threshold, algorithm):
    """Aplica a binarização por limiarização."""
    image = ctx.obj['image']

    click.echo("Aplicando binarização por limiarização")

    image = utils.convert_image(image, 'YCbCr')

    if algorithm == "otsu":
        histogram, _ = np.histogram(image[:,:,0], 256)
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

    mask = image[:,:,0] <= threshold
    image[:,:,0] = 255
    image[mask,0] = 0

    image[:,:,1] = 128
    image[:,:,2] = 128

    ctx.obj['image'] = utils.convert_image(image, 'RGB', from_mode='YCbCr')


@cli.command('convolve')
@click.option('-a', '--axis', 'axis', default="xy", type=click.Choice(["x", "y", "xy"]))
@click.option('-m', '--mode', 'mode', default="reflect",
              type=click.Choice(["reflect", "constant", "nearest", "mirror", "wrap"]))
@click.option('-k', '--kernel', 'kernel', default=None,
              type=click.Choice(["gaussian", "prewitt", "sobel", "roberts", "laplace"]))
@click.option('-x', '--dimension-x', 'dimension_x', default=3)
@click.option('-y', '--dimension-y', 'dimension_y', default=3)
@click.option('-g', '--degree', 'degree', default=1)
@click.option('-s', '--sigma', 'sigma', default=1)
@click.pass_context
def convolve(ctx, axis, mode, kernel, dimension_x, dimension_y, degree, sigma):
    """Aplica o produto de convolução"""
    image = ctx.obj['image']

    click.echo("Aplicando produto de convolução")

    image = np.array(image, dtype='float64')

    if kernel == "gaussian":
        dimension_x = 6 * sigma
        # dimension_y = 6 * sigma

        Gx = np.linspace(-int(dimension_x / 2), int(dimension_x / 2), dimension_x)
        Gx = np.exp((-(Gx ** 2) / (2 * (sigma ** 2))))
        Gx /= Gx.sum()

        # Gy = np.linspace(-int(dimension_y / 2), int(dimension_y / 2), dimension_y)
        # Gy = np.exp((-(Gy ** 2) / (2 * (sigma ** 2))))
        # Gy /= Gy.sum()

        image = scipy.ndimage.filters.convolve1d(image, Gx, axis=0, mode=mode)
        image = scipy.ndimage.filters.convolve1d(image, Gx, axis=1, mode=mode)
        # image = scipy.ndimage.filters.convolve1d(image, Gy, axis=1, mode=mode)

    elif kernel == "prewitt":
        Px = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])
        Py = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])

        for c in range(0, image.shape[2]):
            dx = scipy.ndimage.filters.convolve(image[:,:,c], Px, mode=mode)
            dy = scipy.ndimage.filters.convolve(image[:,:,c], Py, mode=mode)
            image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(image[:,:,c])
            image[:,:,c] *= 255.0 / max_value

    elif kernel == "sobel":
        Sx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        Sy = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        for c in range(0, image.shape[2]):
            dx = scipy.ndimage.filters.convolve(image[:,:,c], Sx, mode=mode)
            dy = scipy.ndimage.filters.convolve(image[:,:,c], Sy, mode=mode)
            image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(image[:,:,c])
            image[:,:,c] *= 255.0 / max_value

    elif kernel == "roberts":
        Rx = np.array([
            [1, 0],
            [0, -1]
        ])
        Ry = np.array([
            [0, 1],
            [-1, 0]
        ])

        for c in range(0, image.shape[2]):
            dx = scipy.ndimage.filters.convolve(image[:,:,c], Rx, mode=mode)
            dy = scipy.ndimage.filters.convolve(image[:,:,c], Ry, mode=mode)
            image[:,:,c] = np.hypot(dx, dy)
            max_value = np.max(image[:,:,c])
            image[:,:,c] *= 255.0 / max_value

    elif kernel == "laplace":
        L = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])

        for c in range(0, image.shape[2]):
            image[:,:,c] = scipy.ndimage.filters.convolve(image[:,:,c], L, mode=mode)
            # image[:,:,c] -= np.min(image[:,:,c])
            image[:,:,c] = np.absolute(image[:,:,c])
            max_value = np.max(image[:,:,c])
            image[:,:,c] *= 255.0 / max_value

    else:
        weights = box = np.ones(dimension_x)
        for g in range(0, degree - 1):
            weights = scipy.signal.convolve(weights, box, mode="same")

        weights = weights / weights.sum()

        if axis == "x":
            image = scipy.ndimage.filters.convolve1d(image, weights, axis=0, mode=mode)
        elif axis == "y":
            image = scipy.ndimage.filters.convolve1d(image, weights, axis=1, mode=mode)
        elif axis == "xy":
            image = scipy.ndimage.filters.convolve1d(image, weights, axis=0, mode=mode)
            image = scipy.ndimage.filters.convolve1d(image, weights, axis=1, mode=mode)

    ctx.obj['image'] = image.astype('uint8')


@cli.command('fourier')
@click.option('-i', '--inverse', 'inverse', is_flag=True)
@click.option('-n', '--numpy', 'numpy', is_flag=True)
@click.pass_context
def fourier(ctx, inverse, numpy):
    """Aplica a Tranformada Discreta de Fourier"""
    image = ctx.obj['image']

    click.echo("Aplicando Transformada Discreta de Fourier")

    if inverse:
        out_image = np.empty_like(image, dtype="uint8")
        out_image[:,:,1] = np.real(image[:,:,1])
        out_image[:,:,2] = np.real(image[:,:,2])

        if numpy:
            M, N, _= image.shape
            temp = np.fft.ifft2(image[:,:,0]) * M * N
        else:
            temp = utils.ifft2(image[:,:,0])

        out_image[:,:,0] = np.real(temp)

        ctx.obj['image'] = utils.convert_image(out_image, 'RGB', from_mode='YCbCr')
        ctx.obj['img_mode'] = 'RGB'

    else:
        image = utils.convert_image(image, 'YCbCr')

        out_image = np.empty_like(image, dtype="complex")
        out_image[:,:,1] = image[:,:,1]
        out_image[:,:,2] = image[:,:,2]

        if numpy:
            temp = np.fft.ifft2(image[:,:,0])
        else:
            temp = utils.fft2(image[:,:,0])

        out_image[:,:,0] = temp

        ctx.obj['image'] = out_image
        ctx.obj['img_mode'] = 'spectrum'


if __name__ == "__main__":
    cli(obj={})
