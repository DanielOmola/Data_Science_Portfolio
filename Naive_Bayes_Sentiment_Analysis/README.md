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
<h3 align="center">Sentiment Analysis with Naive Bayes</h3>
<p align="center">My implementation from scratch vs. Sklearn</p>
<p align="center">
  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_1.png" alt="Logo" width="600" height="300">
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
  
Naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features.
Every pair of features being classified is independent of each other.

Naive Bayes classifiers mostly used in text classification. It is widely used in :

* Spam filtering (identify spam e-mail)

* and Sentiment Analysis (in social media analysis, to identify positive and negative customer sentiments).

This project is a example of implementenation of Naive Bayes Classifier for airline tweets sentiment analysis.
I compare my own (from scratch) implementation with Scikit Learn version. 

### Description
<p style='color:red'>Much of the code has been stored in my own package and modules to make the Jupyter Notebook more readable.</p>
This project contains:

```sh
- 1 jupyter Notebook : Sentiment_Analysis_Naive_Bayes.ipynb
```
<a href="https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Naive_Bayes_Sentiment_Analysis" target="_blank">Project Link</a>

### Datasets

* [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
* 3 Class: positive, negative, neutral

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites
*  Python3
*  Jupyter Notebook
*  Pandas
*  Numpy
*  Plotly
*  Sklearn

### Installation

If you chose the first installation method, make sure the prerequisites are available in your system.

#### Method - 1
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Naive_Bayes_Sentiment_Analysis
```
2. Open the file below in Jupyter Notebook
```JS
Sentiment_Analysis_Naive_Bayes.ipynb
```


#### Method - 2
(the easiest way if docker is already installed in your system)

1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Naive_Bayes_Sentiment_Analysis
```
2. Open the terminal and move to the cloned directory 
```JS
   cd PATH/TO/THE/DIRECTORY
```
3. Create a Docker image from the terminal
```JS
   docker build . --no-cache=true -f Dockerfile.txt -t naive_bayes
```
3. Run the Docker image
```JS
 docker run -it -p 8888:8888 naive_bayes
```



<!-- USAGE EXAMPLES -->
## Usage

Play with it as you want.


<!-- CONTACT -->
## Contact

Daniel OMOLA - daniel.omola@gmail.com


<!-- Recommended links -->
## Recommended links
* <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier" target="_blank">Naive Bayes classifier</a>
* <a href="https://scikit-learn.org/stable/modules/naive_bayes.html" target="_blank">Scikit Learn Naive Bayes classifier</a>
* <a href="https://www.geeksforgeeks.org/naive-bayes-classifiers/" target="_blank">geeksforgeeks.org</a>

