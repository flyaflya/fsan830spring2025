# Chenchuan HE

![Profile Picture](../images/chenchuan.jpg)

## About Me
I can sing Chinese rap songs and am a HIIT beginner (welcome to join me!).

## Research Interests
- How can humans work better with AI agents
- How can we improve the design of AI to better facilitate business processes

## XML to xarray Challenge

### Approach
I approached the XML to xarray conversion by first parsing the XML structure to extract race and entry information into a pandas DataFrame. Then, I transformed this data into an xarray Dataset using pivot operations to create 2D arrays with races and horses as dimensions, while preserving all race details and participant information as data variables and coordinates.

### Query Implementation
Here's how I implemented the query to find the top 3 horses in each race:

```python
def get_top_three_horses(dataset):
    results = []
    
    for race_num in dataset.race.values:
        # Get finish positions for current race
        finish_pos = dataset.finish_pos.sel(race=race_num).values
        
        # Create a mask for valid positions (exclude NaN values)
        valid_mask = ~np.isnan(finish_pos)
        
        if np.any(valid_mask):
            # Get valid positions and corresponding horse numbers
            valid_positions = finish_pos[valid_mask]
            horse_numbers = dataset.horse.values[valid_mask]
            
            # Sort horses by finish position
            sorted_indices = np.argsort(valid_positions)
            top_3_indices = sorted_indices[:3]  # Get indices of top 3 finishers
            top_3_horses = horse_numbers[top_3_indices]
```

### Sample Output
Here's an example output showing the top 3 horses from Race 1:

```
Race 1:
---------
Position: 1.0
Horse: Insouciant
Jockey: Jevian Toledo
Trainer: Brittany T Russell
Odds: 2.3

Position: 2.0
Horse: Amorica
Jockey: Jeremy Rose
Trainer: Anthony Pecoraro
Odds: 2.4

Position: 3.0
Horse: Princess Peanut
Jockey: Xavier Perez
Trainer: John J Robb
Odds: 6.1
```

Also Race 9
```
Race 9:
---------
Position: 1.0
Horse: Idiomatic
Jockey: Florent Geroux
Trainer: Brad H Cox
Odds: 0.4

Position: 2.0
Horse: Classy Edition
Jockey: Kendrick Carmouche
Trainer: Todd A Pletcher
Odds: 2.5

Position: 3.0
Horse: Morning Matcha
Jockey: Paco Lopez
Trainer: Robert E Reid
Odds: 5.9
```