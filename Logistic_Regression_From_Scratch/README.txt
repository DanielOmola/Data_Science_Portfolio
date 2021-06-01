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
<h3 align="center">Linear Regressions, Ridge Regression and Lasso Regression from scratch</h3>
<p align="center">Comparaison with Normal Equation and Sklearn implementataion</p>
<p align="center">
  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_4.gif" alt="Logo" width="600" height="250">
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
  
Logistic Regression is a classification technique that finds a 'line of best fit' to classify data points. 
However, unlike linear regression, where the line of best fit is found using least squares, logistic regression finds the line (logistic curve)
of best fit using maximum likelihood or gradient descent. This is done because the y value can only be one or zero.

To find the line of best fit, Logistic Regression can:
* Minimize binary Cross Entropy loss with Gradient Descent optimisation,
* or Maximise likelyhood with Maximum Likelyhood optimisation.

  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_2.gif" alt="Logo" width="400" height="300">
  </a>

  
Logistic regression can be represented as a Neural Network (NN) and build using deep learning libraries(Tensorflow-Keras and Pytorch).

  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_1.png" alt="Logo" width="500" height="200">
  </a>

This project is a example of implementation of Logistic Regression that I implemented from scratch. 
My main objective was to understand what under the hood and gain a better intuition. The results are compared with sklearn for consistency.**
 

### Description
**Much of the code has been stored in my own package and modules to make the Jupyter Notebook more readable.**
The project contains:

```sh
- 1 Jupyter Notebooks as the main files:
	* Linear_Regression.ipynb
	
- 1 package: mypackage
	* module : classifier.py	
	* module : ploter.py
```




<a href="https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Logistic_Regression_From_Scratch" target="_blank">Project Link</a>
	

### Datasets
Some simulated Data.

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites
*  Python3
*  Jupyter Notebook
*  Pandas
*  Numpy
*  Plotly
*  Sklearn (for comparaison)
*  keras
*  tensorflow 2

### Installation

If you chose the first installation method, make sure the prerequisites are available in your system.

#### Method - 1
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Logistic_Regression_From_Scratch
```
2. Open one of the file below in Jupyter Notebook
```JS
   Linear_Regression.ipynb
```
<!-- -->

#### Method - 2
(the easiest way if docker is already installed in your system)
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Logistic_Regression_From_Scratch
```
2. Open the terminal and move to the cloned directory 
```JS
   cd PATH/TO/THE/DIRECTORY
```
3. Create a Docker image from the terminal
```JS
   docker build . --no-cache=true -f Dockerfile.txt -t linreg
```
4. Run the Docker image
```JS
 docker run -it -p 8888:8888 linreg
```



<!-- USAGE EXAMPLES -->
## Usage

Play with it as you want.


<!-- CONTACT -->
## Contact

Daniel OMOLA - daniel.omola@gmail.com


<!-- Recommended links -->
## Recommended links

* <a href="https://www.youtube.com/watch?v=nk2CQITm_eo" target="_blank">StatQuest: Linear Models Pt.1 - Linear Regression</a>
* <a href="https://www.youtube.com/watch?v=Q81RR3yKn30" target="_blank">Regularization Part 1: Ridge (L2) Regression</a>
* <a href="https://www.youtube.com/watch?v=NGf0voTMlcs" target="_blank">Regularization Part 2: Lasso (L1) Regression</a>
* <a href="https://www.youtube.com/watch?v=1dKRdX9bfIo" target="_blank">Regularization Part 3: Elastic Net Regression</a>
* <a href="https://www.youtube.com/watch?v=sDv4f4s2SB8&t=1138s" target="_blank">Gradient Descent, Step-by-Step</a>
