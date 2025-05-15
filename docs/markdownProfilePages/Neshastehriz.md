# Masoud Neshastehriz
![Profile Picture](../images/Masoud.jpg)

## About Me

Hello! My name is Masoud and I am a PhD student in Financial Services Analytics! I love Swimming and Running!

## Research Interests

- Asset Pricing
- Machine Learning
- Risk Management
- Financial Services Analytics

## Contact

- Email: mneshast@Udel.edu

## XML to xarray Challenge

Explanation:
The xml_to_xarray.py script processes an XML file containing race results by parsing it with xml.etree.ElementTree, extracting relevant data such as race details, horse names, jockeys, and trainers, and storing this information in lists. These lists are then converted into a pandas DataFrame, which is subsequently transformed into an xarray Dataset. This Dataset includes coordinates for track and race date, and variables for each race, such as horse names, positions, and odds. The script also includes functionality to save the Dataset to a NetCDF file for further analysis, and it handles errors related to file parsing and data extraction. This setup allows for efficient data manipulation and querying using Python's data science libraries.

### Code:

```python
for race_num in ds.coords['RACE_NUMBER'].values:
        positions = ds[f'race_{race_num}_positions'].values
        horses = ds[f'race_{race_num}_horses'].values
        odds = ds[f'race_{race_num}_odds'].values
        jockeys_names = ds[f'race_{race_num}_jockeys'].values
        trainers_names  = ds[f'race_{race_num}_trainers'].values
        race_df = pd.DataFrame({'horse': horses, 'position': positions, 'odds': odds, 'jockey': jockeys_names, 'trainer': trainers_names})
        top_horses = race_df.sort_values('position').head(n)

        results[int(race_num)] = top_horses
```

### Output:

Here, I put the result of the last three races:

Race 8:
            horse  position  odds               jockey               trainer
3  Prowling Tiger         1   0.5  Daniel None Centeno  Arnaud None Delacour
1  Twist 'n Twirl         2   2.9  Florent None Geroux       H Graham Motion
5       In Effect         3  43.8       Angel S Arroyo         Neil R Morris

Race 9:
            horse  position  odds                   jockey          trainer
2       Idiomatic         1   0.4      Florent None Geroux       Brad H Cox
4  Classy Edition         2   2.5  Kendrick None Carmouche  Todd A Pletcher
5  Morning Matcha         3   5.9          Paco None Lopez    Robert E Reid

Race 10:
        horse  position  odds                jockey            trainer
3   Dialherup       1.0   0.9       Paco None Lopez  Robert None Mosco
2  Mo Traffic       2.0   1.8      Mychel J Sanchez     Diane D Morici
1    Kilmaley       3.0   5.5  Jaime None Rodriguez     Michael V Pino"""



