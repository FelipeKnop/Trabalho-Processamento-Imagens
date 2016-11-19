# Trabalho Processamento Imagens

## Instalação

```
# pip install click scipy numpy matplotlib pillow
$ chmod +x trabalhopi.py
```

## Como contribuir

Seguir: PEP 8, PEP 8 naming, PEP 257

## Como usar

### Abrir/Exibir

Os comandos básicos para uso da ferramenta são o **open** para abrir e **display** para exibir.

```shell
$ ./trabalhopi.py open -i "digital1.jpg" display
```

### Salvar

O comando **save** pode substituir a exibição na tela, bem como ser encadeado com o **display**.

```shell
$ ./trabalhopi.py open -i "digital1.jpg" save -o output/temp.jpg
$ ls output/temp.jpg
$ ./trabalhopi.py open -i "digital1.jpg" save -o output/temp.jpg display
```

### Converter

O comando **convert** agrupa os parâmetros para alterações básicas na imagem como as cores.
É possível também converter o formato usando o comando save.

```shell
$ ./trabalhopi.py open -i "digital1.jpg" convert -m L display
$ ./trabalhopi.py open -i "digital1.jpg" save -o output/digital1.png
```

### Espaços de cores

É possivel aplicar transformações **gamma**

```shell
$ ./trabalhopi.py open -i "digital1.jpg" gamma -g 2.2 display
```

Também é possivel equalizar o histograma com **histeq**
```shell
$ ./trabalhopi.py open -i "digital1.jpg" histeq display
```

### Filtros

Alguns filtros podem ser aplicados através de seus respectivos comandos, por exemplo, **blur**.

```shell
$ ./trabalhopi.py open -i "digital1.jpg" blur -r 50 display
```

### Estatísticas

Algumas estatísticas das imagens podem ser obtidas, por exemplo, **mse** e o **snr**.

```shell
./trabalhopi.py open -i "digital1.jpg" blur -r 10 mse -r digital1.jpg
./trabalhopi.py open -i "digital1.jpg" blur -r 10 snr -r digital1.jpg
```

### Encadeamento

Pode-se encadear todos esses comandos de maneiras mais complexas, como em:

```shell
$ ./trabalhopi.py open -i "digital1.jpg" \
  convert -m L save -o output/grayscale.png \
  blur -r 30 save -o output/distortion.png \
  display
```
