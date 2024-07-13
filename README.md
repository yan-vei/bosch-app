# Bosch Shipment Processing Dashboard
This project is a dashboard created as a tracking system
for Bosch company internal shipment processing by group 7
for the Data Science course (Summer 2024) at UdS.

Below you find installation guidelines.

## UI
Setting up of the local copy of the UI is simple. Just open the index.html file in your browser.
The rest is handles via the script.js file in the same folder.

## Database
To access the database, you need a MongoDB connecton string.
Once you have the credentials (username and password), do the following:
```
cd backend
touch .env
```
Set up the .env file as follows:
```
MONGODB_URL="mongodb+srv://USERNAME:PASSWORD@bosch.aq3hbwj.mongodb.net/?retryWrites=true&w=majority&appName=bosch&tlsCAFile=isrgrootx1.pem"
```
Additionally, you need a certificate file (isrgrootx1.pem), that you have to generate 
to be able to query the database. Instructions on generating this certificate for your machine are available on the Internet.


## Backend
Setting up the backend is a bit tricky. The steps are the following:
1. Make sure you have Python >=3.9 installed.
2. The best is to use conda package manager (this ensures easy installation for Mac, Windows, and Linux-based systems.)
3. If you have conda, run the following:
```
cd backend
conda env create -f environment.yml
```
4. Once the installation completes, start the API (based on FastAPI):
```
uvicorn main:app --reload
```