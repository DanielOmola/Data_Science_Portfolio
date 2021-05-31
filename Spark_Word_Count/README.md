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
<!-- <h3 align="center">Word Count With Spark and Scala</h3> -->

<a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
	<img src="images/image_2.jpg" alt="Logo" width="700" height="400"></a>





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
## About The Projects

This tiny project is an implementation of word count with RDD. 
My aim is just to give some examples of basic transformations and actions in the context of text analysis. 


### Description

The project contains:

```sh
- 1 Jupyter Notebooks as the main file:

	* Word_Counts_Spark_Scala.ipynb
	
```

<a href="https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Spark_Word_Count" target="_blank">Project Link</a>
	

### Dataset a .txt version of the book economic Hitman From by John Perkins

* ecohitman.txt

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites
*  Jupyter Notebook with spylon kernel


### Installation

If you chose the first installation method, make sure the prerequisites are available in your system.

#### Method - 1
1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Spark_Word_Count
```
2. Run the notebook with Jupyter based on the spylon kernel.

```JS
	* Word_Counts_Spark_Scala.ipynb

```
<!-- -->

#### Method - 2
(the easiest way if docker is already installed in your system)

1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/Spark_Word_Count
```
2. Open the terminal and move to the cloned directory 
```JS
   cd PATH/TO/THE/DIRECTORY
```
3. Create a Docker image from the terminal with the comand below
```JS
   docker build . --no-cache=true -f Dockerfile.txt -t wordcount
```
4. Run the Docker image
```JS
 docker run -it -p 8888:8888 wordcount
```


<!-- USAGE EXAMPLES -->
## Usage

Play with it as you want.


<!-- CONTACT -->
## Contact

Daniel OMOLA - daniel.omola@gmail.com


<!-- Recommended links -->
## Recommended links

* <a href="https://data-flair.training/blogs/spark-rdd-operations-transformations-actions/" target="_blank">Spark RDD Operations-Transformation & Action with Example</a>
* <a href="https://spark.apache.org/docs/latest/rdd-programming-guide.html#working-with-key-value-pairs" target="_blank">Spark : Working with Key-Value Pairs</a>

