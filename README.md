# AI Tutor API

This is the API used for using Age Estimation and Confustion Detection Models for AI Tutor (https://github.com/nishita0512/AI_Tutor)

## Requirements

1. python: v3.10.11
2. Python Packages:
```bash
pip install -r requirements.txt
```
4. Download the models and their weights from Releases: https://github.com/nishita0512/AI_Tutor_API/releases

## How to Run

1. Put all the following 4 files in the project directory:
    a. age_estimation_model.hdf5
    b. age_estimation_model.json
    c. confused_expression_model.h5
    d. confused_expression_model.json
2. run the app:
```bash
flask run
```
or (to run on your local network)
```bash
flask run -h <YOUR_LOCAL_IP_ADDRESS>
```

To get your local IP address (192.168.X.XX):  
Linux
```bash
ifconfig
```
Windows
```bash
ipconfig
```
