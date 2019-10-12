# kaggle



## Introduction

Solution to kaggle MNIST competition.

## Utilisation du code

### Création d'un environnement virtuel

Pour utiliser le code de ce dépôt, il faut d'abord créer un environnement virtuel avec *virtualenv* :


+ Se placer à la racine du projet
+ Installer virtualenv via pip : `pip install virtualenv`
+ Créer l'environnement virtuel : `virtualenv .env`
+ activer l'envirennoment virtuel : `source .env/bin/activate`
+ installer les librairies : `pip install -r requirements.txt`


### Lancement de l'entraînement du modèle

Pour faire tourner le script principal *main.py* :

+ `python main.py` : permet d'entrainer le modèle et d'obtenir une évaluation des performances sur le jeu d'entrainement



## DVC
### Utilisation de DVC
* DVC est une librairie permettant grâce à un espace de stockage de versionner des données, à l’instar de git pour le code.
* DVC permet par ailleurs de créer et versionner des pipes de traitement de données, et de générer des métriques associées, nous autorisant à conserver et reproduire un état exact du projet à un instant donné
Par exemple, les commandes suivantes ont été utilisés à l’initialisation

```
dvc init

git add dvc/.gitignore
git add .dvc/config

# Définition des données à versionner
dvc add test_data.csv

# Création du pipe de traitement
dvc run -d test_data.csv -d main.py -f main.dvc python main.py

# Définition et push de l'espace de stockage
dvc remote add -d upstream gs://ds_exo_data
dvc push

# Reproduction  du pipeline ainsi créé
dvc repro main.dvc
```

### Utilisation d'un espace de stockage GCP

* Couplé à DVC, il sera choisi d'utiliser GCS pour versionner les données
* Pour utiliser DVC en accord avec GCS, il est nécessaire de s'authentifier auprès de l'API Cloud 


#### Accès au bucket GCP

* Un compte de service (disposant des droits sur GCS) est utilisé.
* Une authentification est nécessaire de sorte à permettre à l'application d'accéder à GCS pour la récupération et 
l'écriture des données : https://cloud.google.com/docs/authentication/getting-started#auth-cloud-implicit-python

* Une fois le SDK installé, il suffit d'exporter les credentials pour que DVC accède à GCS : 
`export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"`



## Docker

* L'outil Docker est envisagé de sorte à normaliser l'environnement de travail et opérer le package sur n'importe quelle machine.

### How to run Docker

* Création de l'image : 

`docker image build -t mnist-kaggle:x.x .`

* Lancer le container à partir de l'image en mode interactif

`docker run -it mnist-kaggle:x.x /bin/bash`