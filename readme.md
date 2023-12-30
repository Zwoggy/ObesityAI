# Ai for risk of obesity

The AI can be used to predict whether a Person, given some information, is likely to be in a certain weight category in the future.

## Description

The task was to develop a model that can predict whether a patient is likely to get obesity (predict an obesity level).
The different levels of obesity are: 
* Insufficient Weight
* Normal Weight
* Overweight Level I
* Overweight Level II
* Obesity Type I
* Obesity Type II
* and Obesity Type III.

The Data used for training and test can be found in ObesityDataSet.csv.
The dataset consists of the estimation of obesity levels in people from the countries of Mexico, Peru and Colombia, with ages between 14 and 61 and diverse eating habits and physical condition , data was collected using a web platform with a survey where anonymous users answered each question, then the information was processed obtaining 17 attributes and 2111 records.

The attributes related with eating habits are: Frequent consumption of high caloric food (FAVC), Frequency of consumption of vegetables (FCVC), Number of main meals (NCP), Consumption of food between meals (CAEC), Consumption of water daily (CH20), and Consumption of alcohol (CALC). The attributes related with the physical condition are: Calories consumption monitoring (SCC), Physical activity frequency (FAF), Time using technology devices (TUE), Transportation used (MTRANS)

variables obtained :

Gender, Age, Height and Weight.

NObesity values are:

•Underweight Less than 18.5

•Normal 18.5 to 24.9

•Overweight 25.0 to 29.9

•Obesity I 30.0 to 34.9

•Obesity II 35.0 to 39.9

•Obesity III Higher than 40
## Getting Started

### Dependencies

* Python + IDE (i.e. Pycharm)
* libraries are listed in requirements.txt

### Installing

* pull from GitHub: https://github.com/Zwoggy/ObesityAI 
* ```python
    # In line 21 change the Path in the line below according to your own path
        
    dataset = pd.read_csv('G:/Users/tinys/PycharmProjects/ObesityRiskAI/ObesityDataSet.csv')
    ```

### Executing program

Execute in main.py to train the model and predict on the given data.
```python
if __name__ == "__main__":
    get_data()
```



## Authors
Florian Zwicker

## Version History

* 0.1
    * Initial Release

## License

This project is not licensed

## Acknowledgments


* Inspiration for the readme.md: https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc