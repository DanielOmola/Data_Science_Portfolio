<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
 -->


<!-- PROJECT LOGO -->
<br />
<h3 align="center">Flight Delay Prediction with Spark ML and Scala</h3>
<p align="center">(end of study project)</p>
<p align="center">
  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_2.gif" alt="Logo" width="750" height="350">
  </a>




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2> Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#description">Description</a></li>
      </ul>
      <ul>
        <li><a href="#datasets">Datasets</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
  
This project aims at predicting whether a flight in the United States will be delayed due to adverse weather conditions.
It is based on the scientific publication <a href="https://www.coursehero.com/file/63005424/TIST-Flight-Delay-finalpdf/" target="_blank">
Using scalable Data Mining for Predicting Flight Delays</a> publish on
ACM Transactions on Intelligent Systems and Technology by Belcastra, Taila, Marozzo and Trunfino on January 2016.

Note that this is only a part of the original project that aimed at conducting a machine learning project
from start to finish. The original project, conducted with two of my classmates, covers data acquisition, data preprocessing, data analysis, feature engineering, ML pipeline and use a wide range of technologies including python, Scala, Hadoop, Spark, Databricks, on different distributed
system and cloud platforms (S2, EMR, Paris Dauphine's cluster, Databricks). For those who want to go deeper I provided the final report of the original project and the PowerPoint presentation used for the

What I present here covers only the part of the project build with Scala and spark :
* merging of flights and weather data,
* feature engineering,
* pipeline building,
* model selection and optimization. 

Everything has been repackaged as two Jupyter Notebooks. The first one covers merging data and feature engineering.
The second one build and run ML pipeline. Both the notebooks can be easily executed through docker jupyter/all-spark-notebook.
I also included one month of data, but the original project has been executed on data covering up to two years.


### Description
The project contains:

```sh
- 2 Jupyter Notebooks as the main files:

	* 1_Flight_Project_Data_Preparation.ipynb
	
	* 2_Flight_Project_ML_Model_Selection.ipynb
```

<a href="https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Spark_ML_Flight_Delay_Prediction" target="_blank">Project Link</a>
	

### Datasets
* <a href="https://www.transtats.bts.gov/Tables.asp?DB_ID=120&DB_Name=Airline%20On-Time%20Performance%20Data&DB_Short_Name=On-Time" target="_blank">
Flights data from the Bureau of Transportation Statistics</a>

* <a href="https://www.ncdc.noaa.gov/data-access/land-based-station-data/land-based-datasets/quality-controlled-local-climatological-data-qclcd" target="_blank">
Weather data from the National Center for Environmental information </a>

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites
*  Jupyter Notebook with Spylon Kernel


### Installation

If you chose the first installation method, make sure the prerequisites are available in your system.

#### Method - 1
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Spark_ML_Flight_Delay_Prediction
```
2. Run the two Jupyter Notebook running on spylon kernel.
 <p>Start with the first one to build a parquet file.<br>
Then, you can execute the second one for ML pipeline. </p>

```JS
	* 1_Flight_Project_Data_Preparation.ipynb
	* 2_Flight_Project_ML_Model_Selection.ipynb
```
<!-- -->

#### Method - 2
(the easiest way if docker is already installed in your system)

1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Spark_ML_Flight_Delay_Prediction
```
2. Open the terminal and move to the cloned directory 
```JS
   cd PATH/TO/THE/DIRECTORY
```
3. Create a Docker image from the terminal with the comand below
```JS
   docker build . --no-cache=true -f Dockerfile.txt -t flight
```
4. Run the Docker image
```JS
 docker run -it -p 8888:8888 flight
```


<!-- USAGE EXAMPLES -->
## Usage

Play with it as you want.


<!-- CONTACT -->
## Contact

Daniel OMOLA - daniel.omola@gmail.com


<!-- Recommended links -->
## Recommended links

* <a href="https://spark.apache.org/docs/latest/ml-guide.html" target="_blank">Spark Machine Learning Library (MLlib) Guide</a>
* <a href="https://spark.apache.org/docs/latest/ml-features" target="_blank">Extracting, transforming and selecting features</a>
* <a href="https://www.analyticsvidhya.com/blog/2020/10/all-about-decision-tree-from-scratch-with-python-implementation/" target="_blank">All About Decision Tree</a>
