#!/bin/bash
# Script para construir e implantar o serviço no Cloud Run

# Configurar variáveis
PROJECT_ID="seu-projeto-gcp"
IMAGE_NAME="smart-ads-gmm-inference"
REGION="us-central1"  # Ou a região que você preferir
SERVICE_NAME="smart-ads-inference"

# Construir a imagem Docker
echo "Construindo imagem Docker..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest .

# Enviar a imagem para o Container Registry
echo "Enviando imagem para o Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

# Implantar no Cloud Run
echo "Implantando no Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME:latest \
  --platform managed \
  --region $REGION \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated

echo "Implantação concluída!"