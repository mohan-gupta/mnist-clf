## Project Details

I have created a simple CNN model for MNIST Digit Classification. I have trained the model for 10 epochs, using Cross Entropy Loss function and Adam optimizer with a learning rate of 3e-4. The model's accuracy after 10 epochs, on Training data is 98.5% and Validation data is 98.7%

## Project Structure

- `src`: This directory contains all the files for training and save the model.
- `app`: This directory contains the file for FastAPI application.
- `data`: This directory contains the mnist images downloaded using PyTorch's `torchvision` library.
- `model`: This directory contains the saved model.
- `samples`: This directory contains sample images from MNIST dataset.

## How to Run

1. Set up your credentials by creating a `.env` file in the `app` directory with the following content:

   ```
    MYSQL_HOST=<Host url for MySQL>
    MYSQL_USER=<User name for MySQL>
    MYSQL_PASS=<Password for MySQL>
    MYSQL_DB=<Database used>
   ```

2. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/T3xtifyai/sensifix-demo.git
   ```

   and move the .env file `app` folder.

3. Navigate to the project directory:

   ```bash
   cd sensifix-demo
   ```

4. There are two ways you can run the app.

   a. Use Docker (recommended)

   first build the docker image
   ```bash
   docker build -t mnist-app:new .
   ```

   second run the docker image
   ```bash
   docker run -d --name mnist_container -p 5000:8000 mnist-app:new
   ```

   b. Virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

   Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   ```bash
   uvicorn app.main:app
   ```

5. K8 deployment. (Building the Docker image is mandatory for this step)

    Once, docker image is build, In the project folder open the terminal, and build the k8 deployment using `deployment.yaml` file:

    First start minikube
    ```bash
    minikube start
    ```

    For Loading the docker image locally, run below command
    ```bash
    minikube image load mnist_image
    ```

    Build the deployment
    ```bash
    kubectl apply -f deployment.yaml
    ```

    Run the deployment
    ```bash
    kubectl port-forward service/mnist-app 7080:3000
    ```

6. Access the API in your web browser or use a tool like `curl` or `httpie` to make HTTP requests.

## API Endpoints

1. `/predict`: Either Upload the image in the /docs page of the API, or send the image byte string by making a POST request either using `curl` or `httpie`