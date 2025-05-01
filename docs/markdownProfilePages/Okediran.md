# Tunmbi

![Profile Picture](../images/Tunmbi.jpg)

## About Me

I am a human being who loves to learn and grow. I am interested in machine learning and its applications in Asset Pricing. I am also interested in the live sound engineering and production, and gospel guitar playing.

## Research Interests
- Machine Learning
- Asset Pricing
- Optimization

## XML to Xarray Challenge

### Brief explanation of my approach

I modified the example_xml_to_xarray.py file to include the jockey and trainer names in the xarray dataset. In the parse_xml_to_xarray function, I added the jockey and trainer names to the xarray dataset. In the query_top_horses function, I added the jockey and trainer names to the results. This seems to be an easy way to get the jockey and trainer names in the results.

### Code Snippet
```python
def parse_xml_to_xarray(xml_path):

    # existing code
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
     # I added this to debug, print to see what we're working with
    print(f"Root tag: {root.tag}")
    race_date = root.get('RACE_DATE')
    
    # Find track information
    track_elem = root.find('.//TRACK')
    track_id = track_elem.find('CODE').text
    track_name = track_elem.find('NAME').text

    # existing code

    # add jockey and trainer names to the xarray dataset
    ds[f'race_{race_num}_jockey'] = xr.DataArray(
        race_df['jockey'].values,
        dims=['entry'],
        coords={'entry': np.arange(len(race_df))}
    )
    
    # add trainer name to the xarray dataset
    ds[f'race_{race_num}_trainer'] = xr.DataArray(
        race_df['trainer'].values,
        dims=['entry'],
        coords={'entry': np.arange(len(race_df))}
    )
    # existing code

def query_top_horses(ds, n=3):
    # existing code

    for race_num in ds.coords['RACE_NUMBER'].values:
        # Get positions and horse names
        horses = ds[f'race_{race_num}_horses'].values
        jockey_names = ds[f'race_{race_num}_jockey'].values
        trainer_names = ds[f'race_{race_num}_trainer'].values
        positions = ds[f'race_{race_num}_positions'].values
        odds = ds[f'race_{race_num}_odds'].values
        
        # Create a DataFrame for this race
        race_df = pd.DataFrame({
            'horse': horses,
            'jockey_name': jockey_names,
            'trainer_name': trainer_names,
            'position': positions,
            'odds': odds
        })

        # existing code
```
    
### Sample output

# Race Results

Here are the top horses for each race:

```plaintext
Race 6:
        horse             jockey_name             trainer_name  position  odds
0     Sun Bee        Mychel J Sanchez          H Graham Motion         1   0.8
4     Heckled     Florent None Geroux     Michael None Stidham         2   2.0
1  Anna Lucia  Vincent None Cheminaud  Christophe None Clement         4   7.8

Race 7:
              horse          jockey_name         trainer_name  position  odds
0      Doppelganger   Jevian None Toledo   Brittany T Russell       1.0   0.9
1        Forewarned  Dexter None Haddock  Uriah None St Lewis       2.0  11.1
2  Ridin With Biden      Paco None Lopez        Robert E Reid       3.0   0.6

Race 8:
            horse          jockey_name          trainer_name  position  odds
3  Prowling Tiger  Daniel None Centeno  Arnaud None Delacour         1   0.5
1  Twist 'n Twirl  Florent None Geroux       H Graham Motion         2   2.9
5       In Effect       Angel S Arroyo         Neil R Morris         3  43.8

Race 9:
            horse              jockey_name     trainer_name  position  odds
2       Idiomatic      Florent None Geroux       Brad H Cox         1   0.4
4  Classy Edition  Kendrick None Carmouche  Todd A Pletcher         2   2.5
5  Morning Matcha          Paco None Lopez    Robert E Reid         3   5.9
```
