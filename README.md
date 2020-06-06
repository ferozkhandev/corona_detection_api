# Corona Detection Android App

This app is for detect corona virus in an xray image.

## Tools

- Pycharm
- Python
- Django

## Installation

- Clone this project to your computer.
- run the project using `python manage.py runserver`.


## Directory Structure

|Directory                      |Purpose                          
|-------------------------------|-------------------------------
|coronapneumoniadetectionapi    |It contains API logic, gets image from Android app and call method to detect corona and return response to Android App.           
|image                          |It contains actual corona detection logic and trained model.
|media                          |It contains images sent from the Android App. If this directory is not already created, it will automatically be created upon first image sent to API.

## Note

Make sure to place trained model naming as vgg16_FC_Only.pth in the image directory.

## Link to Android App Repository

https://github.com/ferozkhandev/Corona_Detection_Android_App

## Link to trained model

https://drive.google.com/file/d/1_D0BAWI1U38syrVFY9CYp9V1us7vEum9/view?usp=sharing