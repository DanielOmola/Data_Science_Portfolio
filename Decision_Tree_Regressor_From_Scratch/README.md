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
<h3 align="center">Decision Tree Regression from scratch</h3>
<p align="center">(performance compared with Sklearn implementation)</p>
<p align="center">
  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_1.gif" alt="Logo" width="750" height="350">
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
  
A **Decision Tree** is a **supervised algorithm** that **can be viewed as a series of if questions** aim to **predict a numerical
 value** when facing a **regression problem** or **predict a class** when facing a **classification problem**.

The order of if questions/conditions is based on:

* **impurity criteria for classification: GINI or Enthropy**
* **homogeneity criterion for regression: Variance or Standard deviation**	


The algorithms for building trees breaks down a data set into smaller and smaller subsets while an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.
A decision node has two or more branches. Leaf node represents a classification or decision (used for regression).
The topmost decision node in a tree which corresponds to the best predictor (most important feature) is called a root node.

<img src="images/image_2.png" alt="Logo" width="550" height="300">


Decision trees can handle both categorical and numerical data. They can handle missing data too.


**This project is an example of a Decision Tree Regressor that I implemented from scratch. My main objective was to understand what under the hood and gain a better intuition.
The results are compared with sklearn for consistency.**

### Description

<p style='color:red'>Much of the code has been stored in my own package and modules to make the Jupyter Notebook more readable.</p>

The project contains:

```sh
- 1 Jupyter Notebooks as the main files:
	* Decision_Tree_Regression.ipynb
	
- 1 package: mypackage
	* module : regressor.py

- 1 Docker File for building a docker container:
	* Dockerfile.txt	
```

<a href="https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Decision_Tree_Regressor_From_Scratch" target="_blank">Project Link</a>
	

### Datasets
Housing_Data

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites
*  Python3
*  Jupyter Notebook
*  Pandas
*  Numpy
*  Plotly
*  sklearn (for comparaison)

### Installation

If you chose the first installation method, make sure the prerequisites are available in your system.

#### Method - 1
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Decision_Tree_Regressor_From_Scratch
```
2. Open the file below in Jupyter Notebook
```JS
   Decision_Tree_Regression.ipynb
```
<!-- -->

#### Method - 2
(the easiest way if docker is already installed in your system)

1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Decision_Tree_Regressor_From_Scratch
```
2. Open the terminal and move to the cloned directory 
```JS
   cd PATH/TO/THE/DIRECTORY
```
3. Create a Docker image from the terminal
```JS
   docker build . --no-cache=true -f Dockerfile.txt -t regression_tree
```
4. Run the Docker image
```JS
 docker run -it -p 8888:8888 regression_tree
```
<p style="color:red"> No password should be requested after executing the above command.<br> If that is the case, close any jupyter notebook running and
Anconda, then reexecute the command above. <br>If the password request keeps showing you should reboot your computer. I hope this last part will not be necessary.</p>


<!-- USAGE EXAMPLES -->
## Usage

Play with it as you want.


<!-- CONTACT -->
## Contact

Daniel OMOLA - daniel.omola@gmail.com


<!-- Recommended links -->
## Recommended links

* <a href="https://www.youtube.com/watch?v=g9c66TUylZ4" target="_blank">Regression Trees, Clearly Explained!!!</a>
* <a href="https://www.youtube.com/watch?v=UhY5vPfQIrA" target="_blank">Decision Tree Regression Clearly Explained!</a>

