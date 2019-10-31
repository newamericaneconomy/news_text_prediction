# News / Document Text Prediction
This is a repository with a script and models used to predict the sentiment in documents related to immigration. 

To run sentiment_prediction_script_news.py first open up a terminal or command prompt and then enter the following:
```
python sentiment_prediction_script_news.py
```

## Before running ensure:
1. That you have all text files you would like to predict in your intended directory and have made the necessary updates to the file so that the script is pointing to the proper directory and file names. Currently it is set to read CSV files. 
2. That you have the pre-trained models and pickled document term matrix in the proper directory and the path is updated as needed in the script. 
3. If you have more than one document that you wish to read in, make sure their file locations are listed in the respective lists in the main function. 
