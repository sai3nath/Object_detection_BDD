## Running Ground Truth Data Analysis via Docker

This section describes how to use the provided Docker container to perform Ground Truth (GT) analysis of the BDD100k dataset. This process will parse the dataset annotations, calculate various statistics (class distribution, object sizes, locations, etc.), and generate corresponding visualizations.

**Prerequisites:**

1.  **Docker Installed:** Ensure Docker Desktop (or Docker Engine on Linux) is installed and running on your system.
2.  **BDD100k Dataset:** You must have the BDD100k dataset downloaded locally. The process requires the original JSON annotation files (e.g., `bdd100k_labels_images_train.json`, `bdd100k_labels_images_val.json`).
* Place the dataset in a local directory. You will need to update the paths in the configuration files to point to your local dataset location. For example, you might have a structure like:
        ```
        /path/to/your/datasets/
        └── bdd100k/ => this is the path that we use while running the docker container.
            ├── images/
            │   └── 100k/
            │       ├── train/
            │       └── val/
            └── labels/ # Or labels_original/
                ├── bdd100k_labels_images_train.json
                └── bdd100k_labels_images_val.json
        ```

**Configuration (`configs/docker_gt_config.yaml`):**

* A specific configuration file, `configs/docker_gt_config.yaml`, is used for this Dockerized task.
* **Paths inside this file** (e.g., for `bdd_labels_train_json` and `main_output_dir`) are set to internal Docker paths like `/data_mount/...` and `/output_mount/...`. These are linked to your local system paths via Docker volume mounts at runtime.
* The `run_modules` section in this config is set to execute **only** the ground truth analysis:
    ```yaml
    run_modules:
      ground_truth_analysis: true
    ```

**Steps:**

**1. Build the Docker Image:**
Navigate to the root directory of this project in your terminal. Run the following command:
```bash
docker build -t bdd_gt_analyzer -f docker/data_analysis/Dockerfile .
```

**2. Run the Docker Container:**
```bash
docker run --rm \
  -v "/path/to/your/datasets/bdd100k/:/data_mount:ro" \
  -v "$(pwd)/gt_analysis_docker_output:/output_mount" \
  bdd_gt_analyzer
  ```
