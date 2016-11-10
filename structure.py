from functools import update_wrapper

import click


@click.group(chain=True)
def cli():
    """"Esse programa processa imagens de acordo com os parâmetros
    passados na linha de comando.

    Exemplo:

    \b
        trabalhopi open -i digital1.jpg blur -s 3 save
        trabalhopi open -i digital1.jpg blur -s 5 display
    """



@cli.resultcallback()
def process_commands(processors):
    """"Esse callback é invocado com um iterável de todos os subcomandos
    ligados. Como cada subcomando retorna uma função, podemos ligar
    todos eles e um recebe o resultado do outro."""

    stream = ()

    for processor in processors:
        stream = processor(stream)

    for _ in stream:
        pass



def processor(f):
    """"Helper que rescreve uma função para que ela retorne outra função."""
    def new_func(*args, **kwargs):
        def processor(stream):
            return f(stream, *args, **kwargs)
        return processor
    return update_wrapper(new_func, f)



def generator(f):
    """"Similar à :func:'processor', mas passa valores antigos não modificados
    e não passa os valores como parâmetro."""
    @processor
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in f(*args, **kwargs):
            yield item
    return update_wrapper(new_func, f)



def copy_filename(new, old):
    new.filename = old.filename
    return new