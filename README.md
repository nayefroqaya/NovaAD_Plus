## SparkADLS
## 📌 Description
In this paper, we present SparkADLS, a semi-supervised and scalable log anomaly detection system that translates the sequential pipeline of the known NovaADLS algorithm into a PySpark-based parallel execution environment. SparkADLS reduces computational runtime, demonstrates effective scalability to large-scale datasets, and improves anomaly detection accuracy.

---

## Project Structure
<pre>
├─ datasets/               # Main entry point for NovaADLS datasets  
├─ drain_parser/           # Configuration and parser scripts for Drain  with its references
├─ src/  
│  ├─ pyspark_main.py              # Main script to trigger the full pipeline  
│  ├─ logdata_read.py      # Reads parsed data and performs cleaning and column selection  
│  ├─ pyspark_features_extracting.py  # Extracts features from log events  
│  ├─ pyspark_features_engineering.py # Performs feature transformation  
│  ├─ pyspark_anomaly_detection.py    # Contains anomaly detection modules  
│  ├─ pyspark_model_evaluation.py     # Evaluation module (Precision, Recall, F1-score, etc.)  
│  └─ pyspark_utility.py              # Helper functions used across different stages  
</pre>   
---
## 📊 Datasets
We used 4 open-source log datasets (BGL, HDFS, Thunderbird, Spirit):

| Software System   | Description                        | Data Size | Link                                         |
|-------------------|------------------------------------|-----------|----------------------------------------|
| HDFS              | Hadoop Distributed File System log | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)   |
| BGL               | Blue Gene/L supercomputer log      | 708.76 MB | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (1G)  | Thunderbird supercomputer log      | 1 GB      | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (3G)  | Thunderbird supercomputer log      | 3 GB      | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (9G)  | Thunderbird supercomputer log      | 9 GB      | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (12G) | Thunderbird supercomputer log      | 12 GB     | [LogHub](https://github.com/logpai/loghub)   |
| Spirit (SP_150MB) | Supercomputing system log          | 150 MB    | [Figshare](https://figshare.com/s/6d3c6a83f4828d17be79?file=27775929) |

---

---

## ⚙️ Environment
All libraries are specified with their versions in the requirements file (e.g., Main folder/requirements.txt).

---
## Single-Node Configurations:

Only one server evaluates anomaly detection ac-
curacy and runtime efficiency on all datasets. Baselines and experiments run
on a single-node AMD EPYC 7513 server with 32 CPU cores and ∼251 GB
RAM. The PySpark environment uses eight workers, each with four CPU cores.
Each worker has 28 GB of memory (224 GB total for all executors), while the
remaining memory is reserved for the OS and Spark overhead. The driver is also
assigned 28 GB for stable coordination. This setup provides balanced compute
and memory resources for efficient parallel processing.


## Multi-Node Configurations:

The multi-node setup evaluates the anomaly de-
tection accuracy, runtime efficiency, and scalability on the Thunderbird datasets
(3G, 9G, 12G). All experiments are executed on a Google Cloud Dataproc clus-
ter in the europe-west1 region, with one master and one to three worker nodes.
Each node uses an n4-standard-8 machine (8 vCPUs, 32 GB RAM) with 100 GB
boot disks, running Dataproc image version 2.2 (Debian 12). This configuration
enables scalable distributed processing using Spark’s resource management.
---

## 🛠️ Preparation
Steps to run SparkADLS:

1. Install all required libraries from the requirements file (e.g., Main folder/requirements.txt).
2. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`,`TH_1G`, `TH_3G`,`TH_9G`,`TH_12G`, `SP_150MB`) and upload the (datasetname.log) to this directory.
3. In main.py, set the dataset name (e.g., `HDFS`, `BGL`,`TH_1G`, `TH_3G`,`TH_9G`,`TH_12G`, `SP_150MB`)
4. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
5. The parsing code is available in the `drain_parser` folder.
6. Specify the dataset name in `demo.py` (e.g., BGL). The code is available for all datasets. Uncomment the lines of the dataset you need to use

---
## 📌 Data Parsing
1. For data parsing, all libraries are specified with their versions in the requirements file (e.g., drain_parser/requirements.txt). 
2. To start the parsing process, run (drain_parser/demo.py). 
3. The parsing output will be generated and saved in the datasets' directory.

---
---

## 🚨 Anomaly Detection 
To apply the SparkADLS pipeline on log data. Before start running, you must specify the following parameters :

-  DATASET = 'BGL'  # (e.g., `HDFS`, `BGL`,`TH_1G`, `TH_3G`,`TH_9G`,`TH_12G`, `SP_150MB`) .
-  DATASETS_FOLDER = 'datasets'.
-  Round= '1'   # we run the system three time. Round flag helps to save the data with three versions.
-  Run the main function (`src/main.py`).
-  The main function executes all stages as one pipeline: data preprocessing, anomaly detection, and evaluation.

