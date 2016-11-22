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
@click.option('-o', '--output', 'output', default='output/temp.png', type=click.Path())
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
    click.echo('Razão sinal-ruído: %s' % snr)


@cli.command('gamma')
@click.option('-c', default=1)
@click.option('-g', '--gamma', 'gamma', default=1, type=click.FLOAT)
@click.pass_context
def gamma(ctx, c, gamma):
    """Aplica a transformação gamma: s = c*r^g."""
    image = ctx.obj['image']

    click.echo('Aplicando transformação gamma')

    image = (c * 255 * (image / 255)**gamma).astype(np.uint8)

    ctx.obj['image'] = image


@cli.command('histeq')
@click.option('-b', '--bins', 'bins', default=256)
@click.pass_context
def histeq(ctx, bins):
    """Aplica a equalização de histograma."""
    image = ctx.obj['image']

    click.echo('Aplicando equalização de histograma')

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

    click.echo('Aplicando binarização por limiarização')

    image = utils.convert_image(image, 'YCbCr')

    if algorithm == 'otsu':
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
@click.option('-a', '--axis', 'axis', default='xy', type=click.Choice(['x', 'y', 'xy']))
@click.option('-m', '--mode', 'mode', default='reflect',
              type=click.Choice(['reflect', 'constant', 'nearest', 'mirror', 'wrap']))
@click.option('-k', '--kernel', 'kernel', default=None,
              type=click.Choice(['gaussian', 'prewitt', 'sobel', 'roberts', 'laplace']))
@click.option('-x', '--dimension-x', 'dimension_x', default=3)
@click.option('-y', '--dimension-y', 'dimension_y', default=3)
@click.option('-g', '--degree', 'degree', default=1)
@click.option('-s', '--sigma', 'sigma', default=1)
@click.pass_context
def convolve(ctx, axis, mode, kernel, dimension_x, dimension_y, degree, sigma):
    """Aplica o produto de convolução"""
    image = ctx.obj['image']

    click.echo('Aplicando produto de convolução')

    image = utils.convolve(image, axis, mode, kernel, dimension_x, dimension_y, degree, sigma)

    ctx.obj['image'] = image


@cli.command('fourier')
@click.option('-i', '--inverse', 'inverse', is_flag=True)
@click.option('-n', '--numpy', 'numpy', is_flag=True)
@click.pass_context
def fourier(ctx, inverse, numpy):
    """Aplica a Tranformada Discreta de Fourier"""
    image = ctx.obj['image']

    click.echo('Aplicando Transformada Discreta de Fourier')

    if inverse:
        out_image = np.empty_like(image, dtype='uint8')
        out_image[:,:,1] = np.real(image[:,:,1])
        out_image[:,:,2] = np.real(image[:,:,2])

        if numpy:
            M, N, _ = image.shape
            temp = np.fft.ifft2(image[:,:,0]) * M * N
        else:
            temp = utils.ifft2(image[:,:,0])

        out_image[:,:,0] = np.real(temp)

        ctx.obj['image'] = utils.convert_image(out_image, 'RGB', from_mode='YCbCr')
        ctx.obj['img_mode'] = 'RGB'

    else:
        image = utils.convert_image(image, 'YCbCr')

        out_image = np.empty_like(image, dtype='complex')
        out_image[:,:,1] = image[:,:,1]
        out_image[:,:,2] = image[:,:,2]

        if numpy:
            temp = np.fft.ifft2(image[:,:,0])
        else:
            temp = utils.fft2(image[:,:,0])

        out_image[:,:,0] = temp

        ctx.obj['image'] = out_image
        ctx.obj['img_mode'] = 'spectrum'


@cli.command('product')
@click.option('-k', '--kernel', 'kernel', default=None,
              type=click.Choice(['low-pass', 'band-pass', 'high-pass']))
@click.option('-r', '--radius', 'radius', default=20)
@click.option('-s', '--size', 'size', default=5)
@click.pass_context
def product(ctx, kernel, radius, size):
    """Aplica o produto"""
    image = ctx.obj['image']

    M, N, _ = image.shape
    xx, yy = np.indices((M, N))
    xx -= int(M/2)
    yy -= int(N/2)

    filter = np.ones((M, N))

    if kernel == 'low-pass':
        mask = xx**2 + yy**2 > radius**2
        filter[mask] = 0
    elif kernel == 'high-pass':
        mask = xx**2 + yy**2 < radius**2
        filter[mask] = 0
    elif kernel == 'band-pass':
        mask1 = xx**2 + yy**2 < (radius - size/2)**2
        mask2 = xx**2 + yy**2 > (radius + size/2)**2
        filter[mask1] = 0
        filter[mask2] = 0

    filter = np.roll(filter, -int(M/2), 0)
    filter = np.roll(filter, -int(N/2), 1)

    image[:,:,0] *= filter

    ctx.obj['image'] = image


@cli.command('resize')
@click.option('-s', '--scale', 'scale', default=1, type=click.FLOAT)
@click.option('-m', '--mode', 'mode', default=None,
              type=click.Choice(['area', 'nearest', 'bilinear', 'bicubic']))
@click.pass_context
def resize(ctx, scale, mode):
    image = ctx.obj['image']

    M, N, _ = image.shape
    pil_image = PIL.Image.fromarray(image)
    new_size = (int(N*scale), int(M*scale))

    # Amostragem
    if scale < 1:
        if mode == 'nearest':
            pil_image = pil_image.resize(new_size, PIL.Image.NEAREST)

        # if mode == 'area':
        else:
            mask_size = int(1/scale)
            blurred_image = utils.convolve(pil_image, 'xy', 'reflect', None, mask_size, mask_size, 3, 1)
            pil_image = PIL.Image.fromarray(blurred_image)
            pil_image = pil_image.resize(new_size, PIL.Image.NEAREST)

    # Reconstrução
    elif scale > 1:
        if mode == 'bicubic':
            pil_image = pil_image.resize(new_size, PIL.Image.BICUBIC)

        elif mode == 'nearest':
            pil_image = pil_image.resize(new_size, PIL.Image.NEAREST)

        # elif mode == 'bilinear':
        else:
            pil_image = pil_image.resize(new_size, PIL.Image.BILINEAR)

    ctx.obj['image'] = np.array(pil_image)


if __name__ == "__main__":
    cli(obj={})
