# Web app
## Frontend
* gradio
## Backend
* flask
## Other
* nginx
* docker
* ec2

### Details for API integration
Please go through the test_model.py in the network directory

File Locations
1. Input -> backend/dataset
2. Checkpoint -> backend/checkpoint
3. Result image -> backend/output

Flow
1. Store the image in the dataset directory and Open the image using Image.Open 
2. Call restore_image() to initiate the image restoration process 
3. Extract the generated result in the output directory and pass it to the frontend layer
