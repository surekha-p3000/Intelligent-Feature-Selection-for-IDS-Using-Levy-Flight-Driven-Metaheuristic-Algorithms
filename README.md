# Intelligent Feature Selection for IDS Using Levy Flight-Driven Metaheuristic Algorithms

## Overview
This project implements intelligent feature selection for Intrusion Detection Systems (IDS) using Levy Flight-driven metaheuristic algorithms. The approach leverages advanced optimization techniques to identify the most relevant features for improving intrusion detection accuracy while reducing computational complexity.

## Features
- Implementation of Levy Flight-driven metaheuristic algorithms
- Intelligent feature selection for intrusion detection
- Support for multiple optimization algorithms
- Comparative analysis and performance metrics
- Scalable architecture for large datasets

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Required libraries listed in `requirements.txt`

### Setup
```bash
git clone https://github.com/surekha-p3000/Intelligent-Feature-Selection-for-IDS-Using-Levy-Flight-Driven-Metaheuristic-Algorithms.git
cd Intelligent-Feature-Selection-for-IDS-Using-Levy-Flight-Driven-Metaheuristic-Algorithms
pip install -r requirements.txt
```

## Usage

### Running Feature Selection
```bash
python feature_selection.py --dataset <path_to_dataset> --algorithm <algorithm_name>
```

### Example
```bash
python feature_selection.py --dataset data/intrusion_detection.csv --algorithm levy_flight
```

### Configuration
Edit the configuration file to customize parameters:
- Population size
- Number of iterations
- Feature selection thresholds
- Dataset path

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   └── intrusion_detection.csv
├── src/
│   ├── feature_selection.py
│   ├── algorithms/
│   │   ├── levy_flight.py
│   │   ├── particle_swarm.py
│   │   └── genetic_algorithm.py
│   ├── utils/
│   │   ├── data_preprocessing.py
│   │   └── evaluation_metrics.py
│   └── config.py
├── results/
│   ├── selected_features.csv
│   └── performance_metrics.json
└── tests/
    └── test_algorithms.py
```

## Methodology

### Levy Flight-Driven Metaheuristic Approach

Levy Flight is a random walk pattern used in nature, particularly observed in animal foraging behavior. This project leverages Levy Flight properties to enhance metaheuristic optimization algorithms:

1. **Levy Flight Mechanism**: Implements probabilistic movement patterns that balance exploration and exploitation
2. **Feature Selection Process**:
   - Initialize population of candidate feature subsets
   - Evaluate fitness based on classification performance and feature count
   - Update positions using Levy Flight-driven movement
   - Iteratively refine feature selection until convergence

3. **Optimization Algorithms**:
   - A novel Lévy-flight Enhanced Crayfish Optimization (LECO) algorithm is proposed to improve the exploration capability and convergence behaviour of the            conventional Crayfish Optimization Algorithm for wrapper-based feature selection.
   - A novel Gliding Lévy-flight Adaptive Aquila Metaheuristic (GLAM) is developed by integrating gliding position control and Lévy-flight mutation into the
     Aquila Optimization framework to achieve a more effective exploration--exploitation balance during feature subset optimization.
   - Ant Lion Optimization (ALO)
   - Social Spider Optimization (SSO)

4. **Fitness Function**: Balances detection accuracy with feature reduction
   ```
   F = w_1 \times \text{Accuracy} + w_2 \times (1 - \text{False Positives})- w_3 \times \text{Computational Cost}
   ```

### IDS Dataset
- Gotham available at https://iotdataset.com/data/gotham-dataset-iot-ids-2025
- CICIDS2017 dataset at https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15 dataset [UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." In 2015 military communications and information systems conference (MilCIS), pp. 1-6. Ieee, 2015]  

## Results

### Performance Metrics
- **Accuracy**: Percentage of correct classifications
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Feature Reduction Rate**: Percentage of features eliminated
- **Computational Time**: Time taken for classification

### Comparative Analysis
Results comparing different algorithms:
Table 1: Performance summary of ML classifiers with metaheuristic optimizers 
| **Classifier** | **Best Performance & Selector** | **Worst Performance & Selector** | **Time Taken (Range)** | **Key Insight** |
|---|---|---|---|---|
| Logistic Regression | SSO (0.8611) | GLAM (0.7361) | 11–45 s | Performance is highly sensitive to the quality of feature selection. |
| K-Nearest Neighbors (KNN) | SSO (0.9752) | LECO (0.9587) | 0.6–1.1 s | A reliable and efficient algorithm that maintains stable performance across different feature sets. |
| Support Vector Machine (SVM) | ALO (0.7525) | LECO (0.6409) | 90–128 s | Consistently underperforms and is the slowest algorithm, making it less suitable for this problem. |
| Gaussian Naïve Bayes | SSO (0.7971) | GLAM (0.6690) | 0.03–0.05 s | An extremely fast algorithm, making it useful for speed-critical applications, but with moderate accuracy. |
| AdaBoost | GLAM (0.3983) | SSO (0.3212) | 0.69–1.73 s | Consistently low performance across all feature selection methods, suggesting it is not suitable for this dataset. |
| LightGBM | ALO (0.9993) | GLAM (0.3747) | 6.4–10.65 s | Performance is highly volatile and extremely sensitive to the feature selection method used. |

*Results are indicative and may vary based on dataset and configuration*

Table 2: Comparative Performance against Recent SOTA Feature Selection Methods
| **Method** | **Initial Features** | **Selected Features** | **Best Fitness** | **XGBoost Accuracy** | **Random Forest Accuracy** | **Execution Time (s)** |
|---|---:|---:|---:|---:|---:|---:|
| ChOA [ref45] | 23 | 11 | 1.8410 | 0.9792 | 0.9764 | 12.45 |
| GJO [ref46] | 23 | 9 | 1.8654 | 0.9823 | 0.9811 | 10.12 |
| HHO [ref47] | 23 | 10 | 1.8702 | 0.9854 | 0.9832 | 14.89 |
| GMSMFO [ref49] | 23 | 10 | 1.8890 | 0.9842 | 0.9810 | 11.23 |
| TPSOSA [ref50] | 23 | 9 | 1.8912 | 0.9867 | 0.9854 | 14.50 |
| **LECO** | **23** | **12** | **1.9009** | **0.9943** | **0.9911** | **7.03** |
| **GLAM** | **23** | **7** | **1.9541** | **0.9965** | **0.9971** | **6.40** |

Table 3: Metaheuristic Feature Selection Performance on Alternative IDS Benchmarks
| **Dataset** | **Feature Selector** | **Dimension (Init → Sel)** | **Classifier Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|---|---|---:|---|---:|---:|---:|---:|
| **CICIDS 2017** | Baseline (No FS) | 78 → 78 | XGBoost | 0.9642 | 0.9630 | 0.9642 | 0.9636 |
| **CICIDS 2017** | ALO | 78 → 16 | XGBoost | 0.9789 | 0.9791 | 0.9789 | 0.9790 |
| **CICIDS 2017** | SSO | 78 → 14 | Random Forest | 0.9812 | 0.9805 | 0.9812 | 0.9808 |
| **CICIDS 2017** | **LECO** | **78 → 12** | XGBoost | **0.9914** | **0.9910** | **0.9914** | **0.9912** |
| **CICIDS 2017** | **GLAM** | **78 → 9** | Random Forest | **0.9945** | **0.9942** | **0.9945** | **0.9943** |
| **UNSW NB15** | Baseline (No FS) | 42 → 42 | XGBoost | 0.9310 | 0.9295 | 0.9310 | 0.9302 |
| **UNSW NB15** | ALO | 42 → 11 | XGBoost | 0.9512 | 0.9520 | 0.9512 | 0.9516 |
| **UNSW NB15** | SSO | 42 → 10 | Random Forest | 0.9567 | 0.9558 | 0.9567 | 0.9562 |
| **UNSW NB15** | **LECO** | **42 → 8** | XGBoost | **0.9705** | **0.9711** | **0.9705** | **0.9708** |
| **UNSW NB15** | **GLAM** | **42 → 6** | XGBoost | **0.9822** | **0.9819** | **0.9822** | **0.9820** |
## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Author**: Surekha P3000  
**Last Updated**: August 2026
