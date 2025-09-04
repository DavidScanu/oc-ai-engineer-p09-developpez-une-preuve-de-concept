# Rebuild l'image avec les corrections
docker build -t sentiment-dashboard .

# Test local
docker run -p 8080:8080 --env PORT=8080 sentiment-dashboard