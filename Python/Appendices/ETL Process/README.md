###### How to install:

1) Download Docker from docs.docker.com/get-docker, open it and let it run in the background.

2) Open a terminal and run the following commands:

	> cd „/path/to/ETL Process“
	> docker-compose up

This might take a few minutes.

3) In Docker, you can now find the ETL Process. Click on the blue localhost „8080:8080“.

4) You should now be able to access the directory „notebooks“ and run iPython Notebook.

5) For future access you may start Docker, turn on ETL Process and start the blue localhost „8080:8080“, which will automatically guide you to Jupyter Notebooks and access the directory „notebooks“ and lets you correspond with the databases.


###### Explanation:

„Docker-compose up“ starts three services: a PostgreSQL database (db), a Jupyter notebook server (jupyter) and a database management UI (adminer). The PostgreSQL container uses a custom script „install.sh“ to initialize the database and sets the password to „example“. The Jupyter container builds from your local Dockerfile, mounts your „./notebooks/“ directory so you can edit and run notebooks and exposes port 8888. The Adminer container gives you a simple web interface to explore your database at „http:localhost:8080“. So this setup runs isolatedservices that communicate with each other, useful for data science and database development.