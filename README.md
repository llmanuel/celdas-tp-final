Para lograr correr esta aplicación deberán tener instalado Python 3, algunas bibliotecas y creado el entorno virtual:

**Bibliotecas necesarias**

* numpy
* pygame
* virtualenv
* pylint
* tensorflow
* getkey

Las mismas pueden ser instaladas utilizando el sistema de instalación de paquetes de Python: PIP


Instalar PIP en Linux (Debian, ubuntu, Mint y derivados):
========================================================

sudo apt install python3-pip

python3 -m pip install --upgrade pip


Instalar Virtualenv o actualizarlo:
==================================

pip3 install virtualenv
pip3 install --upgrade virtualenv

Crear el Virtual Env:
=====================
python3 -m venv .

Start Virtual Env:
=====================
source bin/activate

Instalar bibliotecas:
=====================

pip3 install -r requirements/requirements.txt

Para que la red juegue 
======================
Asegurarse de tener un modelo entrenado. Tener bien seteada la ruta de lectura del mismo en el archivo `deepQNetwork/modelPlaying.py`.

Usando vscode:
* Play usando el comando Python: Start DQN
* Elegir la opción 'p'

Por consola:

```sh
python3 start.py -m p
```

Para entrenar la red 
====================
Usando vscode:
* Play usando el comando Python: Start DQN
* Elegir la opción 't'

Por consola:
```sh
python3 start.py -m t
```

Observaciones sobre la red
==========================

Para entrenar un modelo nuevo se debe cambiar las variables para leer y escribir archivos en `deepQNetwork/modelTraining.py`. 

Ademas se debe tener en cuenta de descomentar la linea 46 del mismo archivo, comentando las lineas 48 y 55.

Para restaurar la metadata y el checkpoint logrado se debe comentar la linea 46 y descomentar las lineas 48 y 55.

Para entrenar esta red neuronal se utilizaron aproximadamente 170GB de memoria RAM.

Iniciar tensorboard
===================
tensorboard --logdir=tensorboard/dqn/e/4