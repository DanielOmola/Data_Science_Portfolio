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
<h3 align="center">MongoDB with Docker</h3>
<p align="center">(how to deploy and use it)</p>
<p align="center">
  <a href="https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees" target="_blank">
    <img src="images/image_2.gif" alt="Logo" width="600" height="350">
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
  
This project aims at showing how to run MongoDB as a Docker Container. 
It covers :
* how to create and launch a MongoDB container,
* how to create and feed a data base,
* how to prepare and query a data base,
* and how to check performance.

Note that this project is only a part of the original project that aimed at comparing two major NoSQL document oriented
data base, including MongoDB and Couchbase. 
	

### Description
The project contains:

```sh
- 2 Jupyter Notebooks as the main files:

	* 1_Flight_Project_Data_Preparation.ipynb
	
	* 2_Flight_Project_ML_Model_Selection.ipynb
```

<a href="https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/LSTM_Anomaly_Detection" target="_blank">Project Link</a>
	

### Datasets

My background as a financial professional and my interest in textual categorization applied to finance led me to choose the 
<a href="https://chewbii.com/wp-content/uploads/2015/11/reuters_26-09-1997.json_.zip" target="_blank">Reuters-21578</a> dataset,
published by Carnegie Group, Inc. and Reuters, Ltd (provider of financial information). 
This dataset consists of Reuters journalistic articles published during the year 1987 and covers the periods before and after the October 19, 1987 crash, commonly known as "Black Monday.
From a financial research and journalistic perspective, this dataset is interesting in many ways, especially for sentiment analysis.


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites
*  Docker installed in your system and already running.


### Installation

1. Clone the repo
```JS
   git clone https://github.com/DanielOmola/Data_Science_Portfolio/tree/main/LSTM_Anomaly_Detection
```
2. Open a terminal and download MongoDB image with one of the command bellow.
Note that, the second query should be used if you want a specific version of MongoDB (here the 4.2)

```JS
	docker pull mongo	
```

```JS
	docker pull mongo:4.2
```
3. Check if the image installation was successfull. 

```JS
	docker images
```

4. Run mongoDB. 

```JS
	docker run -d -p 27017-27019:27017-27019 --name mongodb mongo:4.2
```
```JS
	docker exec -it mongodb bash
```
```JS
	mongo
```
5. Create a new data base called reuters. 
```JS
	use reuters
```
6. Create a collection named articles within the reuters database. 

```JS
	db.createCollection('articles')
```
8. Stop the MongoDB by stricking : ctrl + c + z at once

9. Open a new terminal, go to the clowned directory and copy reuters data to mongo temporary directory with docker. 

```JS
	docker cp data/reuters_26-09-1997.json mongodb:/tmp/reuters_26-09-1997.json
```

10. Feed the reuters data base with data from reuters_26-09-1997.json. 
```JS
	docker exec -it mongodb bash
```
```JS
	mongoimport --db reuters --collection articles --file /tmp/reuters_26-09-1997.json
```

11. Check that the creation and feed of the DB has been successfull. 

```JS
	mongo
```
```JS
	show dbs
```

12. Check that the articles collection has been successfully feed. 
```JS
	use reuters
```
```JS
	db.articles.find().pretty()
```
13. Display the statistics of the collection. 
```JS
	db.articles.stats()
```
14. Now you are ready to play.


<!-- USAGE EXAMPLES -->
## Usage

I provided several queries (in the md file queries.md) that you can use to query
the data base. Play with it as you want. You can also create new queries that you think are relevant. 


<!-- CONTACT -->
## Contact

Daniel OMOLA - daniel.omola@gmail.com


<!-- Recommended links -->
## Recommended links

* <a href="https://docs.mongodb.com/manual/tutorial/query-documents/" target="_blank">MongoDB: Query Documents</a>
* <a href="https://docs.mongodb.com/manual/crud/" target="_blank">MongoDB: CRUD Operations</a>