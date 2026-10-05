#!/usr/bin/env bash
# Run on a fresh Ubuntu 22.04/24.04 EC2 (t3.large+, 30GB EBS). Open ports 22 and 80 in the Security Group.
set -euo pipefail
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git awscli
sudo usermod -aG docker ubuntu
git clone https://github.com/<you>/cortex.git ~/cortex
cd ~/cortex && cp .env.example .env && echo ">> edit .env with your GROQ/PINECONE keys, then: docker compose up -d --build"
