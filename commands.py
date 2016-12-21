import math
import struct

import click
import PIL.Image
import scipy.ndimage
import scipy.signal
import numpy as np

import matplotlib.pyplot as plt

import utils
import huffman


def open_image(input):
    click.echo('Abrindo "%s"' % input)
    pil_image = PIL.Image.open(input).convert('RGB')
    return np.array(pil_image)

    # try:
    #     pil_image = PIL.Image.open(input).convert('RGB')
    #     return np.array(pil_image)
    # except Exception as e:
    #     click.echo('Imagem não pode ser aberta "%s": %s' % (input, e), err=True)


def display(image, img_mode='RGB', phase=False, logarithm=False, center=False):
    click.echo('Exibindo imagem')

    if img_mode == 'spectrum':
        M, N, _ = image.shape

        if phase:
            show_image = np.arctan2(np.imag(image), np.real(image))[:,:,0]
            show_image += math.pi
            show_image *= 255 / (2 * math.pi)

        else:
            show_image = np.sqrt((np.real(image)**2 + np.imag(image)**2))[:,:,0]

            if logarithm:
                # NOTE(andre:2016-11-21): http://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
                show_image = (255 / math.log(1 + np.abs(show_image).max())) * np.log(1 + np.abs(show_image))
            else:
                show_image = utils.normalize(show_image, 0, 255, ignore_first_max=True)

        if center:
            show_image = np.roll(show_image, int(M/2), 0)
            show_image = np.roll(show_image, int(N/2), 1)

        pil_image = PIL.Image.fromarray(show_image)
        pil_image.show()

    else:
        pil_image = PIL.Image.fromarray(image, img_mode)
        pil_image.show()


def save(image, output, img_mode='RGB', phase=False, logarithm=False, center=False):
    click.echo('Salvando imagem em "%s"' % output)

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
            else:
                show_image = utils.normalize(show_image, 0, 255, ignore_first_max=True)

        if center:
            show_image = np.roll(show_image, int(M/2), 0)
            show_image = np.roll(show_image, int(N/2), 1)

        pil_image = PIL.Image.fromarray(show_image.astype('uint8'))
        pil_image.save(output)

    else:
        pil_image = PIL.Image.fromarray(image)
        pil_image.save(output)

def convert(image, from_mode, mode):
    click.echo('Convertendo a imagem para %s' % mode)
    if from_mode is not None:
        pil_image = PIL.Image.fromarray(image, from_mode)
    else:
        pil_image = PIL.Image.fromarray(image)

    return np.array(pil_image.convert(mode))


def mse(image, image_ref):
    click.echo('Calculando MSE')

    size = image.shape
    size_ref = image_ref.shape

    if size != size_ref:
        click.echo('Imagem base possui tamanho diferente da imagem de referência')
        click.echo('%s != %s' % (size, size_ref))
        return

    diff = (image.astype(int) - image_ref.astype(int))
    mse = (diff**2).mean()

    return mse


def snr(image, image_ref):
    size = image.shape
    size_ref = image_ref.shape

    if size != size_ref:
        click.echo('Imagem base possui tamanho diferente da imagem de referência')
        click.echo('%s != %s' % (size, size_ref))
        return

    diff = (image.astype(int) - image_ref.astype(int))

    signal = (image.astype(int)**2).mean()
    noise = mse(image, image_ref)

    snr = 10 * (np.log(signal / noise) / np.log(10))

    return snr


def gamma(image, c, g):
    click.echo('Aplicando transformação gamma')

    image = (c * 255 * (image / 255)**g).astype(np.uint8)

    return image


def histeq(image, img_mode, bins=256):
    click.echo('Aplicando equalização de histograma')

    image = convert(image, img_mode, 'YCbCr')

    channel = image[:,:,0]

    histogram, bins_ar = np.histogram(channel, bins)
    cdf = histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]

    temp_image = np.interp(channel.flatten(), bins_ar[:-1], cdf)
    image[:,:,0] = temp_image.reshape(channel.shape)

    image = convert(image, 'YCbCr', img_mode)

    return image


def threshold(image, img_mode, t):
    click.echo('Aplicando binarização por limiarização')

    image = convert(image, img_mode, 'YCbCr')

    mask = image[:,:,0] <= t
    image[:,:,0] = 255
    image[mask,0] = 0

    image[:,:,1] = 128
    image[:,:,2] = 128

    image = convert(image, 'YCbCr', img_mode)

    return image


def otsu_threshold(image, img_mode):
    image = convert(image, img_mode, 'YCbCr')

    t = 0

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
            t = i

    return t


def convolve(image, axis, mode, kernel, dimension_x, dimension_y, degree, sigma):
    image = np.array(image, dtype='float64')

    if kernel == 'gaussian':
        dimension_x = 6 * sigma

        Gx = np.linspace(-int(dimension_x / 2), int(dimension_x / 2), dimension_x)
        Gx = np.exp((-(Gx ** 2) / (2 * (sigma ** 2))))
        Gx /= Gx.sum()

        image = scipy.ndimage.filters.convolve1d(image, Gx, axis=0, mode=mode)
        image = scipy.ndimage.filters.convolve1d(image, Gx, axis=1, mode=mode)

    elif kernel == 'prewitt':
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

    elif kernel == 'sobel':
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

    elif kernel == 'roberts':
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

    elif kernel == 'laplace':
        L = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])

        for c in range(0, image.shape[2]):
            image[:,:,c] = scipy.ndimage.filters.convolve(image[:,:,c], L, mode=mode)
            # image[:,:,c] = np.absolute(image[:,:,c])
            # max_value = np.max(image[:,:,c])
            # image[:,:,c] *= 255.0 / max_value
            image[:,:,c] = image[:,:,c] * (127 / np.max(np.absolute(image[:,:,c]))) + 127

    else:
        weights_x = box_x = np.ones(dimension_x)
        for g in range(0, degree - 1):
            weights_x = scipy.signal.convolve(weights_x, box_x, mode='full')

        weights_x = weights_x / weights_x.sum()

        weights_y = box_y = np.ones(dimension_y)
        for g in range(0, degree - 1):
            weights_y = scipy.signal.convolve(weights_y, box_y, mode='full')

        weights_y = weights_y / weights_y.sum()

        if axis == 'x':
            image = scipy.ndimage.filters.convolve1d(image, weights_x, axis=0, mode=mode)
        elif axis == 'y':
            image = scipy.ndimage.filters.convolve1d(image, weights_y, axis=1, mode=mode)
        elif axis == 'xy':
            image = scipy.ndimage.filters.convolve1d(image, weights_x, axis=0, mode=mode)
            image = scipy.ndimage.filters.convolve1d(image, weights_y, axis=1, mode=mode)

    return image.astype('uint8')


# NOTE(andre:2016-11-21): Matrix Form of 2D DFT: http://fourier.eng.hmc.edu/e101/lectures/Image_Processing/node6.html
def dft(image):
    M, N = image.shape

    temp_m = np.arange(0, M)
    temp_n = np.arange(0, N)
    m, k = np.meshgrid(temp_m, temp_m)
    n, l = np.meshgrid(temp_n, temp_n)

    Wm = np.exp(-1j * 2 * math.pi * m * k / M)
    Wn = np.exp(-1j * 2 * math.pi * n * l / N)

    return np.matmul(np.matmul(Wm, image), Wn)


def idft(image):
    M, N = image.shape

    temp_m = np.arange(0, M)
    temp_n = np.arange(0, N)
    m, k = np.meshgrid(temp_m, temp_m)
    n, l = np.meshgrid(temp_n, temp_n)

    Wm = (1 / M) * np.exp(1j * 2 * math.pi * m * k / M)
    Wn = (1 / N) * np.exp(1j * 2 * math.pi * n * l / N)

    return np.matmul(np.matmul(Wm, image), Wn)


def fourier(image, img_mode, inverse, numpy):
    if inverse:
        out_image = np.empty_like(image, dtype='uint8')
        # out_image[:,:,1] = image[:,:,1].real
        # out_image[:,:,2] = image[:,:,2].real

        if numpy:
            # temp = np.fft.ifft2(image[:,:,0])
            out_image[:,:,0] = np.fft.ifft2(image[:,:,0]).real.astype('uint8')
            out_image[:,:,1] = np.fft.ifft2(image[:,:,1]).real.astype('uint8')
            out_image[:,:,2] = np.fft.ifft2(image[:,:,2]).real.astype('uint8')
        else:
            # temp = idft(image[:,:,0])
            out_image[:,:,0] = idft(image[:,:,0]).real.astype('uint8')
            out_image[:,:,1] = idft(image[:,:,1]).real.astype('uint8')
            out_image[:,:,2] = idft(image[:,:,2]).real.astype('uint8')

        # out_image[:,:,0] = np.real(temp)

        return convert(out_image, 'YCbCr', 'RGB')

    else:
        image = convert(image, img_mode, 'YCbCr')

        out_image = np.empty_like(image, dtype='complex')
        # out_image[:,:,1] = image[:,:,1]
        # out_image[:,:,2] = image[:,:,2]

        if numpy:
            # temp = np.fft.fft2(image[:,:,0])
            out_image[:,:,0] = np.fft.fft2(image[:,:,0])
            out_image[:,:,1] = np.fft.fft2(image[:,:,1])
            out_image[:,:,2] = np.fft.fft2(image[:,:,2])
        else:
            # temp = dft(image[:,:,0])
            out_image[:,:,0] = dft(image[:,:,0])
            out_image[:,:,1] = dft(image[:,:,1])
            out_image[:,:,2] = dft(image[:,:,2])

        # out_image[:,:,0] = temp

        return out_image


def product(image, kernel, radius, size):
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

    return image


def resize(image, scale, mode):
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
            blurred_image = convolve(pil_image, 'xy', 'reflect', None, mask_size, mask_size, 3, 1)
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

    image = np.array(pil_image)

    return image


GAMMA = 1.6
GAMMA_R = 1.38
GAMMA_I = 0.9
CS = 8 # Chroma subsampling

def compress_channel(channel):
    nrow, ncol = channel.shape

    # Center the channel

    channel = np.roll(channel, int(nrow / 2), 0)
    channel = np.roll(channel, int(ncol / 2), 1)

    # Spiral ordering

    channel = utils.matrix_to_spiral(channel)

    # Quantization

    # channel = channel / utils.quantization_matrix(channel)

    channel_min = channel.min()
    channel_max = channel.max()

    # channel = ((channel - channel_min) * (2 / (channel_max - channel_min))) - 1
    # neg_mask = channel < 0
    # channel = np.abs(channel) ** (1 / GAMMA)
    # channel[neg_mask] *= -1
    # channel = (channel + 1) / 2
    # channel = channel * 254 + 1

    channel = (channel - channel_min) / (channel_max - channel_min)
    channel = channel ** (1 / GAMMA)
    channel = channel * 254 + 1

    channel = channel.astype('int')

    # Run-length encode

    channel_zero = int(round((-channel_min / (channel_max - channel_min)) * 254.0, 10) + 1)
    channel = utils.rl_encode(channel, channel_zero)

    # Huffman encode

    channel = channel.tobytes()
    channel = huffman.encode(channel)

    return channel, channel_min, channel_max, len(channel)


def compress(image, img_mode, output):
    # a = base = np.arange(-512, 512)
    #
    # a_min = a.min()
    # a_max = a.max()
    #
    # a = ((a - a_min) * (2 / (a_max - a_min))) - 1
    # neg_mask = a < 0
    # a = np.abs(a) ** (1 / GAMMA)
    # a[neg_mask] *= -1
    # a = (a + 1) / 2
    # a = a * 254 + 1
    #
    # plt.plot(base, a)
    # plt.show()
    #
    # a = (a - 1) / 254
    # a = (a * 2) - 1
    # neg_mask = a < 0
    # a = np.abs(a) ** GAMMA
    # a[neg_mask] *= -1
    # a = ((a + 1) * ((a_max - a_min) / 2)) + a_min
    #
    # plt.plot(base, a)
    # plt.show()

    shape = image.shape

    image = convert(image, img_mode, 'YCbCr')

    # Chroma subsampling

    Y = image[:,:,0]
    Cb = utils.submatrices(image[:,:,1], CS, CS).mean((2, 3))
    Cr = utils.submatrices(image[:,:,2], CS, CS).mean((2, 3))

    # Fourier transform

    Y = dft(Y)
    Cb = dft(Cb)
    Cr = dft(Cr)

    # Channel compression

    Yr, Yr_min, Yr_max, Yr_length = compress_channel(Y.real)
    Yi, Yi_min, Yi_max, Yi_length = compress_channel(Y.imag)
    Cbr, Cbr_min, Cbr_max, Cbr_length = compress_channel(Cb.real)
    Cbi, Cbi_min, Cbi_max, Cbi_length = compress_channel(Cb.imag)
    Crr, Crr_min, Crr_max, Crr_length = compress_channel(Cr.real)
    Cri, Cri_min, Cri_max, Cri_length = compress_channel(Cr.imag)

    test = np.empty((shape[0], shape[1], 3), dtype='complex')
    test[:,:,0] = (Y.real + Y.imag * 1j)
    test[:,:,1] = 128
    test[:,:,2] = 128
    display(test, 'spectrum', phase=False, logarithm=True, center=True)

    click.echo((Yr_length, Yi_length, Cbr_length, Cbi_length, Crr_length, Cri_length))

    fmt = 'iiffffffffffffiiiiii%us%us%us%us%us%us' % (Yr_length, Yi_length,
                                                      Cbr_length, Cbi_length,
                                                      Crr_length, Cri_length)

    with open(output, 'wb') as file:
        file.write(struct.pack(fmt,
                               shape[0], shape[1],
                               Yr_min, Yr_max, Yi_min, Yi_max,
                               Cbr_min, Cbr_max, Cbi_min, Cbi_max,
                               Crr_min, Crr_max, Cri_min, Cri_max,
                               Yr_length, Yi_length,
                               Cbr_length, Cbi_length,
                               Crr_length, Cri_length,
                               Yr, Yi, Cbr, Cbi, Crr, Cri))


def decompress_channel(channel, nrow, ncol, channel_min, channel_max):
    # Huffman decode

    channel = huffman.decode(channel)
    channel = np.frombuffer(channel, dtype='int')

    # Run-length decode

    channel_zero = int(round((-channel_min / (channel_max - channel_min)) * 254.0, 10) + 1)
    channel = utils.rl_decode(channel, channel_zero)

    # Reconstruction

    # channel = channel * utils.quantization_matrix(channel)

    # channel = (channel - 1) / 254
    # channel = (channel * 2) - 1
    # neg_mask = channel < 0
    # channel = np.abs(channel) ** GAMMA
    # channel[neg_mask] *= -1
    # channel = ((channel + 1) * ((channel_max - channel_min) / 2)) + channel_min

    channel = (channel - 1) / 254
    channel = channel ** GAMMA
    channel = (channel * (channel_max - channel_min)) + channel_min

    # Matrix ordering

    channel = utils.matrix_from_spiral(channel, nrow, ncol)

    # Decenter the channel

    channel = np.roll(channel, int(nrow / 2), 0)
    channel = np.roll(channel, int(ncol / 2), 1)

    return channel


def decompress(input):
    with open(input, 'rb') as file:
        data = file.read()

    shape, data = struct.unpack('ii', data[:8]), data[8:]
    (Yr_min, Yr_max, Yi_min, Yi_max), data = struct.unpack('ffff', data[:16]), data[16:]
    (Cbr_min, Cbr_max, Cbi_min, Cbi_max), data = struct.unpack('ffff', data[:16]), data[16:]
    (Crr_min, Crr_max, Cri_min, Cri_max), data = struct.unpack('ffff', data[:16]), data[16:]

    (Yr_length, Yi_length), data = struct.unpack('ii', data[:8]), data[8:]
    (Cbr_length, Cbi_length), data = struct.unpack('ii', data[:8]), data[8:]
    (Crr_length, Cri_length), data = struct.unpack('ii', data[:8]), data[8:]

    fmt = '%us%us%us%us%us%us' % (Yr_length, Yi_length,
                                  Cbr_length, Cbi_length,
                                  Crr_length, Cri_length)
    (Yr, Yi, Cbr, Cbi, Crr, Cri) = struct.unpack(fmt, data)

    Yr = decompress_channel(Yr, shape[0], shape[1], Yr_min, Yr_max)
    Yi = decompress_channel(Yi, shape[0], shape[1], Yi_min, Yi_max)
    Cbr = decompress_channel(Cbr, int(shape[0] / CS), int(shape[1] / CS), Cbr_min, Cbr_max)
    Cbi = decompress_channel(Cbi, int(shape[0] / CS), int(shape[1] / CS), Cbi_min, Cbi_max)
    Crr = decompress_channel(Crr, int(shape[0] / CS), int(shape[1] / CS), Crr_min, Crr_max)
    Cri = decompress_channel(Cri, int(shape[0] / CS), int(shape[1] / CS), Cri_min, Cri_max)

    test = np.empty((shape[0], shape[1], 3), dtype='complex')
    test[:,:,0] = (Yr + Yi * 1j)
    test[:,:,1] = 128
    test[:,:,2] = 128
    display(test, 'spectrum', phase=False, logarithm=True, center=True)

    Y = (Yr + Yi * 1j)
    Cb = (Cbr + Cbi * 1j)
    Cr = (Crr + Cri * 1j)

    # Fourier transform

    Y = idft(Y).real.astype('uint8')
    Cb = idft(Cb).real.astype('uint8')
    Cr = idft(Cr).real.astype('uint8')

    # Chroma resampling

    Cb = np.repeat(Cb, CS, 0)
    Cb = np.repeat(Cb, CS, 1)
    Cr = np.repeat(Cr, CS, 0)
    Cr = np.repeat(Cr, CS, 1)

    decoded_image = np.empty((shape[0], shape[1], 3), dtype='uint8')
    decoded_image[:,:,0] = Y
    decoded_image[:,:,1] = Cb[:shape[0],:shape[1]]
    decoded_image[:,:,2] = Cr[:shape[0],:shape[1]]

    decoded_image = convert(decoded_image, 'YCbCr', 'RGB')

    return decoded_image


# a = base = np.arange(-512, 512)
#
# a_min = a.min()
# a_max = a.max()
#
# a = (a - a_min) * (2 / (a_max - a_min)) -1
# neg_mask = a < 0
# a = np.abs(a) ** (1 / GAMMA)
# a[neg_mask] *= -1
# a = a * 254 + 1
#
# plt.plot(base, a)
# plt.show()
