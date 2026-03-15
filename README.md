# A Novel Lightweight CNN for Constrained IoT Devices: Achieving High Accuracy With Parameter Efficiency on the MSTAR Dataset

This repository contains the implementation of the research paper:

**"A Novel Lightweight CNN for Constrained IoT Devices: Achieving High Accuracy With Parameter Efficiency on the MSTAR Dataset"**

The proposed **NLCNN (Novel Lightweight Convolutional Neural Network)** model is designed for **resource-constrained IoT devices**, achieving high classification accuracy while maintaining **parameter efficiency**.

---

## Paper Information

Published in: **IEEE Access (2024)**

Paper link:
https://doi.org/10.1109/ACCESS.2024.3487313

Please refer to the paper for detailed methodology, experiments, and results.

---

## Hardware and Software Environment

The experiments in this study were conducted using the following system configuration:

* **Laptop:** Dell Latitude 7480
* **Processor:** Intel Core i7-7600U (7th Generation) @ 2.80 GHz
* **CPU:** 4 logical processors (~2.9 GHz base clock speed)
* **RAM:** 16 GB
* **Graphics:** Intel HD Graphics 620
* **Operating System:** Windows 10 Pro (64-bit)
* **Programming Environment:** MATLAB R2019a

---

## Dataset

The experiments were conducted using the **MSTAR (Moving and Stationary Target Acquisition and Recognition) dataset**, a widely used benchmark dataset for **SAR target recognition**.

The dataset contains multiple military vehicle classes captured using **Synthetic Aperture Radar (SAR)** imagery.

Classification experiments in this study include:

* **3-Class classification**
* **10-Class classification**

Due to dataset licensing restrictions, the dataset is not included in this repository.

Researchers can obtain the dataset from the official source:

MSTAR Dataset:
https://www.sdms.afrl.af.mil/index.php?collection=mstar

After downloading the dataset, update the dataset path in the MATLAB scripts before running the training and testing code.


---

## Repository Structure

```
LSARNet/
│
├── code/
│   ├── LSARNet_Train.m
│   └── LSARNet_Test.m
│
├── models/
│   ├── LSARNet_3class.mat
│   └── LSARNet_10class.mat
│
└── README.md
```

# Instructions

## 1. Dataset Path
Set the `dataset_path` variable in `LSARNet_Train.m` and `LSARNet_Test.m` to your dataset location.  


## 2. Fully Connected Layer

Use fullyConnectedLayer(3) for 3-class classification

Use fullyConnectedLayer(10) for 10-class classification

Update this layer in LSARNet_Train.m according to your experiment.


## 3. Feature Visualization Code

LSARNet_Train.m contains code for:

Feature maps (activations)

Heatmaps

Convolutional kernel visualization

During training, you can comment out these sections.


---

## Usage

### Training

Run the training script in MATLAB:

```
LSARNet_Train
```

### Testing

Run the testing script:

```
LSARNet_Test
```

Make sure the **dataset path is correctly configured in the scripts before execution**.

---

## Applications

The proposed lightweight CNN model is particularly suitable for:

* IoT-based intelligent sensing systems
* Embedded vision systems
* Real-time target recognition
* Resource-constrained edge devices

---

## Citation

If you use this code in your research, please cite the following paper:

```
@article{rahman2024novel,
  title={A novel lightweight CNN for constrained IoT devices: Achieving high accuracy with parameter efficiency on the MSTAR dataset},
  author={Rahman, Noor and Khan, Muzammil and Ullah, Israr and Kim, Do-Hyeun},
  journal={IEEE Access},
  volume={12},
  pages={160284--160298},
  year={2024},
  publisher={IEEE}
}
```

---

## License

This repository is intended for **academic and research purposes only**.
