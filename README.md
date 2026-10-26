# Network Detection Intrusion System

## 1. Project Title

**Network Detection Intrusion System using Machine Learning**

---

## 2. Problem Statement

With the rapid growth of computer networks and internet-based services, cyberattacks have become increasingly common. Traditional security systems often struggle to detect new and sophisticated attacks. The objective of this project is to develop a Machine Learning-based Network Intrusion Detection System (NIDS) that can analyze network traffic and accurately identify malicious activities, helping organizations improve network security and reduce potential threats.

---

## 3. Team Members

* Aaditya Sogani
* Hrishikesh
* Shaik Zaid Kamal

Each team member works on a separate network intrusion dataset while following the same workflow of data preprocessing, model training, and evaluation.

---

## 4. Dataset Information

The project uses network traffic datasets for intrusion detection.

### Aaditya Sogani Dataset

* Botnet Friday Dataset
* PortScan Friday Dataset
* Web Attacks Thursday Dataset

### Other Team Members

* Separate intrusion detection datasets for independent analysis and comparison.

### Dataset Features

* Source IP
* Destination IP
* Protocol
* Packet Length
* Flow Duration
* Number of Packets
* Traffic Statistics
* Attack Labels

The datasets contain both normal and malicious network traffic records.

---

## 5. Algorithms Used

The following Machine Learning algorithms are used for intrusion detection:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

Data preprocessing techniques:

* Missing Value Handling
* Feature Scaling
* Label Encoding
* Feature Selection

---

## 6. Results

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Key Findings

* Machine Learning models successfully classified normal and malicious network traffic.
* Random Forest achieved strong performance in intrusion detection.
* Comparative analysis was performed across different datasets.

*Final accuracy values and confusion matrices will be added after model evaluation.*

---

## 7. Installation Instructions

### Clone the Repository

```bash
git clone https://github.com/AadityaSogani/Network-Detection-Intrusion-System.git
cd Network-Detection-Intrusion-System
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## 8. Usage Instructions

### Run Data Preprocessing

```bash
python preprocessing.py
```

### Train the Model

```bash
python training.py
```

### View Results

The generated results, evaluation metrics, and visualizations can be found in the respective team member folders.

---

## Future Enhancements

* Real-time network traffic monitoring
* Deep Learning-based intrusion detection
* Web dashboard for visualization
* Automated alert generation
* Cloud deployment support

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Git & GitHub
