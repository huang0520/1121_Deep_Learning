# Ask for the name of container
echo "Enter the name of container: "
read name

# Create container
docker run -d --gpus all \
    --name $name \
    -p 8888:8888 \
    -v "$(pwd)":/tf \
    deeplearning:latest

# Set notebook password
docker exec -it $name jupyter notebook --generate-config
docker exec -it $name jupyter notebook password

# Restart container
docker restart $name
