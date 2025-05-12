# Construcción de las imagenes

docker build -f docker/Dockerfile.entrenamiento -t entrenamiento_app .
docker build -f docker/Dockerfile.inferencias -t inferencias_app .
