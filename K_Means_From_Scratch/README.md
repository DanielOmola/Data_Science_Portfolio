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
<h3 align="center">K-Means from scratch</h3>
<p align="center">(experiment with spherical and non-spherical shape clusters)</p>
<p align="center">
  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_1.gif" alt="Logo" width="400" height="350">
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
  
**K-Means** Clustering is an **unsupervised learning algorithm**, which **groups the unlabeled dataset into different clusters**.
**K** defines the **number o**f pre-defined **clusters** that need to be created in the process.
It is an **iterative algorithm that divides the unlabeled dataset into k different clusters** 
in such a way that each dataset belongs only to one group that has similar properties.
K-Means is a convenient way **to discover the categories of groups in the unlabeled dataset** on its own without the need for any training.

<img src="images/image_2.gif" alt="Logo" width="600" height="350">

This project is a example of K-Means algorithm that I implemented from scratch. My main objective was to understand what under the hood and gain a better intuition.
The algorithm is applyed to different shape clusters in order to understand when it suitable and when is not.** 


### Description
<p style='color:red'>Much of the code has been stored in my own package and modules to make the Jupyter Notebook more readable.</p>
The project contains:

```sh
- 1 Jupyter Notebooks as the main files:
	* K_Means.ipynb
	
- 1 package: mypackage
	* module : clustering.py	
	* module : ploter.py

- 1 Docker File for building a docker container:
	* Dockerfile.txt	
```

<a href="https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/K_Means_From_Scratch" target="_blank">Project Link</a>
	

### Datasets
Simulated Data and Sci-kit learn data sets.

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites
*  Python3
*  Jupyter Notebook
*  Pandas
*  Numpy
*  Plotly
*  sklearn

### Installation

If you chose the first installation method, make sure the prerequisites are available in your system.

#### Method - 1
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/K_Means_From_Scratch
```
2. Open one of the file below in Jupyter Notebook
```JS
   K_Means.ipynb
```
<!-- -->

#### Method - 2
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/K_Means_From_Scratch
```
2. Open the terminal and move to the cloned directory 
```JS
   cd PATH/TO/THE/DIRECTORY
```
3. Create a Docker image from the terminal
```JS
   docker build . --no-cache=true -f Dockerfile.txt -t kmeans
```
4. Run the Docker image
```JS
 docker run -it -p 8888:8888 kmeans
```


<!-- USAGE EXAMPLES -->
## Usage

Play with it as you want.


<!-- CONTACT -->
## Contact

Daniel OMOLA - daniel.omola@gmail.com


<!-- Recommended links -->
## Recommended links

* <a href="https://en.wikipedia.org/wiki/K-means_clustering" target="_blank">k-means clustering</a>
* <a href="https://www.youtube.com/watch?v=4b5d3muPQmA" target="_blank">StatQuest: K-means clustering</a>
* <a href="https://www.youtube.com/watch?v=AWKCCK5YHsE" target="_blank">K Means Clustering Intuition</a>