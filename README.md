# Trabalho Processamento Imagens

## Instalação

Requisitos de software:
- [Python 3](https://www.python.org/downloads/)
- [pip](https://pypi.python.org/pypi/pip/)
- [Git](https://git-scm.com/download)
- [Git LFS](https://git-lfs.github.com/)

No terminal execute os seguintes comandos:

```
# pip install click scipy numpy matplotlib pillow
$ git lfs install
$ git clone https://github.com/FelipeKnop/Trabalho-Processamento-Imagens.git
$ cd Trabalho-Processamento-Imagens
$ chmod +x trabalhopi.py
```

## Como contribuir

Usar:
- PEP 8 (exceto E231; E501 com 110 caracteres)
- PEP 8 naming
- PEP 257
- Git LFS

## Como usar

### Abrir/Exibir

Os comandos básicos para uso da ferramenta são o **open** para abrir e **display** para exibir.
O comando **open** suporta a opção **-i** para especificar o arquivo a ser processado.
Enquanto **display** não tem nenhuma opção.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" display
```

### Salvar

O comando **save** pode substituir a exibição na tela, bem como ser encadeado com o **display**.
O comando **save** pode receber a opção **-o** seguido do caminho do arquivo de destino.
Caso não seja especificado o destino, o arquivo será salvo como *output/temp.jpg*.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" save -o output/digital.jpg
$ ./trabalhopi.py open -i "sample/digital1.jpg" save -o output/digital.jpg display
```

### Converter

O comando **convert** agrupa os parâmetros para alterações básicas na imagem.
A opção **-m** permite a especificação de vários modos que definem o tipo e profundidade do pixel.
Alguns modos comuns são "L" (tons de cinza), "RGB", "RGBA", "YCbCr". Para mais módulos ver [aqui](https://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#concept-modes).

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" convert -m L display
```

É possível também converter o formato da imagem usando o comando save.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" save -o output/digital1.png
```

### Espaços de cores

É possível aplicar transformações **gamma**. Esse commando pode receber a opção **-g**
seguido de um número de ponto flutuante para alterar o resultado da transformação.


```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" gamma -g 2.2 display
```

Também é possível equalizar o histograma com **histeq**. Esse comando não recebe parâmetros.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" histeq display
```

O comando **threshold** permite a binarização por limiarização.
É possível a especificação do limear através da opção **-t** ou a seleção do algoritmo
para escolha automática do mesmo com a opção **-o**.
Atualmente o único algoritmo disponível é o de Otsu.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" threshold -t 128 display
$ ./trabalhopi.py open -i "sample/digital1.jpg" threshold -a otsu display
```

### Estatísticas

Algumas estatísticas das imagens podem ser obtidas através dos comandos **mse** e o **snr**.
Ambos comandos suportam a opção **-r** para especificação da imagem de referência.

```shell
./trabalhopi.py open -i "sample/digital1.jpg" blur -r 10 mse -r "sample/digital1.jpg"
./trabalhopi.py open -i "sample/digital1.jpg" blur -r 10 snr -r "sample/digital1.jpg"
```

### Encadeamento

Como pode-se notar os comandos podem ser encadeados de várias formas a fim de realizar o processamento da image.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" \
  convert -m L save -o "output/grayscale.png" \
  blur -r 30 save -o "output/distortion.png" \
  display
```
