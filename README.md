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
- PEP 8 (exceto E231; E501 com 100 caracteres)
- PEP 8 naming
- PEP 257
- Git LFS

## Como usar

### Abrir/Exibir

Os comandos básicos para uso da ferramenta são o **open** para abrir e **display** para exibir.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" display
```

### Salvar

O comando **save** pode substituir a exibição na tela, bem como ser encadeado com o **display**.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" save -o output/temp.jpg
$ ls output/temp.jpg
$ ./trabalhopi.py open -i "sample/digital1.jpg" save -o output/temp.jpg display
```

### Converter

O comando **convert** agrupa os parâmetros para alterações básicas na imagem como as cores.
É possível também converter o formato usando o comando save.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" convert -m L display
$ ./trabalhopi.py open -i "sample/digital1.jpg" save -o output/digital1.png
```

### Espaços de cores

É possível aplicar transformações **gamma**.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" gamma -g 2.2 display
```

Também é possível equalizar o histograma com **histeq**.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" histeq display
```

O comando **threshold** permite a binarazação por limiarização.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" threshold -t 128 display
$ ./trabalhopi.py open -i "sample/digital1.jpg" threshold -a otsu display
```

### Filtros

Alguns filtros podem ser aplicados através de seus respectivos comandos, por exemplo, **blur**.

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" blur -r 50 display
```

### Estatísticas

Algumas estatísticas das imagens podem ser obtidas, por exemplo, **mse** e o **snr**.

```shell
./trabalhopi.py open -i "sample/digital1.jpg" blur -r 10 mse -r sample/digital1.jpg
./trabalhopi.py open -i "sample/digital1.jpg" blur -r 10 snr -r sample/digital1.jpg
```

### Encadeamento

Pode-se encadear todos esses comandos de maneiras mais complexas, como em:

```shell
$ ./trabalhopi.py open -i "sample/digital1.jpg" \
  convert -m L save -o output/grayscale.png \
  blur -r 30 save -o output/distortion.png \
  display
```
