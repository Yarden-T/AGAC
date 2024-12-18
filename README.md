# AGAC: All Gaps Are Covered - Sim2Real Using Contrastive Unpaired Translation and Multimodal Fusion

![Solution Flow Chart](./Images/Solution%20flow%20chart2%20(1).png)

## Project Overview
AGAC (All Gaps Are Covered) is a deep learning project designed to bridge the **sim2real gap** in robotic tactile sensing. In robotic systems, models are often trained using simulated data, but they perform poorly when transferred to real-world environments due to differences between the simulation and reality. This project addresses the gap using two core models:
1. **Contrastive Unpaired Translation (CUT)** for image domain adaptation.
2. **FusiMod**, a custom-built multimodal fusion model for adapting numerical data.
3.**Data cleaning**, comparing the generated and  simulated sets of XYZ coordinates and detecting errors to improve the accuracy of the data.


The goal is to enhance the realism of simulated tactile sensor images and improve the accuracy of the numerical data associated with them. By doing so, AGAC improves performance in real-world applications without the need for extensive data collection from real environments.

## Key Achievements
- Successfully implemented **image domain adaptation** using Contrastive Unpaired Translation (CUT) to convert simulated sensor images into realistic representations.
- Developed **FusiMod**, a fusion model that bridges the gap between simulated and real-world numerical data (e.g., XYZ coordinates) from tactile sensors.
- Achieved a **62.6% improvement** in prediction accuracy after applying both CUT and FusiMod, demonstrating significant progress in tackling the sim2real gap.
- Demonstrated the model’s capability for **zero-shot learning**, enabling the system to generalize to new sensors and environments with minimal additional training.
- Utilized a dataset of **10,000 images** (5,000 simulation, 5,000 real) and developed novel methods for image and numerical data pairing, yielding enhanced real-world performance.

![Generated vs Real Comparison](./Images/generate%20compare%20(2).png)

## System Design
- **AllSight Sensor**: Optical-based tactile sensor that tracks the deformation of a soft elastomer upon contact with an object. Data collected both from a real robotic arm and simulated via the TACTO physics engine.
- **SRC Model**: Focuses on unpaired image translation using **patch-wise contrastive loss**, which maximizes similarity between corresponding patches of simulated and real images.
- **FusiMod**: This model adapts the **numerical features** (XYZ coordinates) of the tactile sensor's simulation to match real-world data. It uses a ResNet-based architecture to extract image features and combines them with the simulation data.

## Methodology
1. **Image Domain Adaptation**: The SRC model generates realistic images by minimizing the difference between patches in simulated and real images using contrastive learning.
2. **Numerical Domain Adaptation**: FusiMod processes the simulated XYZ numerical data, adjusting it based on the generated images to reduce the sim2real gap in numerical features.

## Experimental Results
- **Image Domain Results**: FID (Frechet Inception Distance) and RMSE (Root Mean Squared Error) were used to evaluate the SRC model. A **52.6% improvement** in RMSE was observed after applying the SRC model.
- **Numerical Domain Results**: The FusiMod model reduced XYZ errors by over 42% when compared to the original simulation data.
- **Zero-shot Learning**: AGAC was successfully tested on new sensors and environments, showing **70.2% improvement** in accuracy for unseen data.

## Future Work
- Extend the approach to other robotic tasks that experience a significant sim2real gap in both images and numerical domains.
- Further optimize the models to minimize the amount of real-world data required for training.
