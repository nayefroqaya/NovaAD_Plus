# LOGAD
LOGAD: Semi-Supervised Anomaly Detection in Log Series

## 📌 Description
**LOGAD** is a novel approach for log-based anomaly detection, i.e., textual event series.
Main steps:

(1) Leverage a broad range of features.
(2) based on a subset of only normal data, we use OC-SVM to generate pseudo-labels and estimate labels for unlabeled log sequences in the training set.
(3) Employ an ensemble learning framework to train the dataset.

We carried out empirical evaluations on two open-source datasets (HDFS and BGL), achieving strong performance.
In addition, we assessed the runtime of LOAGAD on a CPU and presented comparative results.

---
## 📊 Datasets
We used two open-source log datasets (more will be added in the future):

| Software System | Description                          | Time Span  | # Messages   | Data Size | Link |
|-----------------|--------------------------------------|------------|--------------|-----------|------|
| **HDFS**        | Hadoop Distributed File System log   | 38.7 hours | 11,175,629   | 1.47 GB   | [LogHub](https://github.com/logpai/loghub) |
| **BGL**         | Blue Gene/L supercomputer log        | 214.7 days | 4,747,963    | 708.76 MB | [LogHub](https://github.com/logpai/loghub)  |
---
## ⚙️ Environment
All libraries are specified with their versions in the requirements file.

---
## 🛠️ Preparation
Steps to run LOGAD:

1. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`).
2. In main.py file, make the name of the dataset BGL or HDFS.
3. Be sure you install all libraries in requirmwnt file.
---
## 📌 Data Parsing
1. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
2. On GitHub, only a subset of the dataset is used. You should use the full datasets (HDFS and BGL).
   For new dataset: 
1. Place name of the dataset in main.py file.  
2. Implement a new case for new dataset in the LogdataRead class (see `src/logdata_reader.py`).
---
## 🚨 Anomaly Detection
To apply LOGAD on new log data:

2. Place name of the dataset in main.py file.  
3. Start the file: `src/main.py`).
---
## 📬 Contact
We are happy to answer your questions:   

| Name               | Email Address                             |
|--------------------|-------------------------------------------|
| Nayef Roqaya       | roqaya@staff.uni-marburg.de               |
| Thorsten Papenbrock| papenbrock@informatik.uni-marburg.de      |
| Hajira Jabeen      | hajira.jabeen@uk-koeln.de                 |
  



   
