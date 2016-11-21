#!/bin/bash

[ ${EUID} -ne 0 ] && {
   echo "É necessário ser root para executar a instalação"
   exit 1
}

SRC_DIR=$(pwd)
SRC_BIN=${SRC_DIR}/dip.py
DEST_DIR=/usr/local/bin
DEST_BIN=${DEST_DIR}/dip

[ -f ${DEST_BIN} ] && rm ${DEST_BIN}
ln -s ${SRC_BIN} ${DEST_BIN}

cp ${SRC_DIR}/script/dip.bash-completion /etc/bash_completion.d/dip
source /etc/bash_completion.d/dip

echo Instalação executada com sucesso
