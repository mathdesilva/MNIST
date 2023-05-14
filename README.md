# Shaw And Partners Coding Challenge

Classification Model for the Fashion-MNIST Dataset.

---
## How to run the API

The API can be executed using Docker or not.

#### Using Docker

1. run docker compose
    ```sh
    docker compose up
    ```
    or use ```-d``` to keep the container running in background
    ```sh
    docker compose up -d
    ```

#### Local

1. Install the requirements
    ```sh
    pip install -r backend/requirements.txt
    ```

2. Run the API:

    ```sh
    uvicorn backend.main:app
    ```

---

## How to use the API

1. To use the API, you need to do a POST request at ```localhost:8000/model/prediction``` containing the image on ```file``` key at the body.

2. The response is a JSON file with the following keys:
2.1. **class_id**: Prediction resul class id (0 to 9);
2.2. **class_name**: Prediction result class name;
2.3. **confidence**: Confidence of model for the prediction (0 to 1);
2.4. **probs**: List of all model prediction probabilities;
2.5. **labels**: List of the classes name;
2.6. **filename**: Name of the file used on prediction.
---

## Project folders
* **backend**: API implementation;
* **models**: Model used on the API;
* **train**: Python Notebook used to train the models.

---

By Matheus de Andrade Silva