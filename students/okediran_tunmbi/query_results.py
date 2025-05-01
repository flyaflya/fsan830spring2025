Root tag: CHART
Dataset information:
<xarray.Dataset> Size: 3kB
Dimensions:            (TRACK: 1, RACE_DATE: 1, RACE_NUMBER: 10, entry: 6)
Coordinates:
  * TRACK              (TRACK) <U3 12B 'DEL'
  * RACE_DATE          (RACE_DATE) <U10 40B '2023-07-08'
  * RACE_NUMBER        (RACE_NUMBER) int64 80B 1 2 3 4 5 6 7 8 9 10
  * entry              (entry) int64 48B 0 1 2 3 4 5
Data variables: (12/50)
    race_1_horses      (entry) object 48B 'Princess Peanut' ... 'Ninetynineca...
    race_1_positions   (entry) int64 48B 3 5 6 1 2 4
    race_1_odds        (entry) float64 48B 6.1 20.9 38.2 2.3 2.4 1.4
    race_1_jockey      (entry) object 48B 'Xavier None Perez' ... 'Raul E Mena'
    race_1_trainer     (entry) object 48B 'John J Robb' ... 'Richard P Sillaman'
    race_2_horses      (entry) object 48B 'Satisfied' ... 'My Jo Jo'
    ...                 ...
    race_9_trainer     (entry) object 48B 'Albert M Stall' ... 'Robert E Reid'
    race_10_horses     (entry) object 48B 'Wright Up Front' 'Kilmaley' ... nan
    race_10_positions  (entry) float64 48B 4.0 3.0 2.0 1.0 nan nan
    race_10_odds       (entry) float64 48B 3.9 5.5 1.8 0.9 nan nan
    race_10_jockey     (entry) object 48B 'Daniel None Centeno' ... nan
    race_10_trainer    (entry) object 48B 'James J Toner' ... nan
Attributes:
    track_name:  DELAWARE PARK

Top 3 horses in each race:

Race 1:
             horse         jockey_name           trainer_name  position  odds
3       Insouciant  Jevian None Toledo     Brittany T Russell         1   2.3
4          Amorica    Jeremy None Rose  Anthony None Pecoraro         2   2.4
0  Princess Peanut   Xavier None Perez            John J Robb         3   6.1

Race 2:
              horse             jockey_name      trainer_name  position  odds
5          My Jo Jo  Alexander None Crispin  Chelsey E Moysey         1   8.3
0         Satisfied           Jean C Alvelo    A Ferris Allen         2   0.9
3  One Clear Moment             Raul E Mena  Michael E Gorham         3  12.0

Race 3:
                horse           jockey_name  ... position  odds
3  Thtwasthenthisisnw   Daniel None Centeno  ...      1.0   1.1
0             Keranos      Jeremy None Rose  ...      2.0   3.1
1            Orocovix  Jaime None Rodriguez  ...      3.0   3.3

[3 rows x 5 columns]

Race 4:
              horse       jockey_name        trainer_name  position  odds
3   Bring the Magic    Angel S Arroyo    Tim None Woolley         1   0.9
1            Jaylee  Kevin None Gomez     Keri None Brion         3   4.6
0  Wine in the Dark  Jeremy None Rose  Richard J Hendriks         4  12.8

Race 5:
            horse             jockey_name          trainer_name  position  odds
1      Alva Starr        Mychel J Sanchez      Brett A Brinkman         1   1.4
5  Cheetara (CHI)  Vincent None Cheminaud  Ignacio None Correas         2   6.3
0    Sweet Gracie           Jean C Alvelo        A Ferris Allen         3  30.4

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

Race 10:
        horse           jockey_name       trainer_name  position  odds
3   Dialherup       Paco None Lopez  Robert None Mosco       1.0   0.9
2  Mo Traffic      Mychel J Sanchez     Diane D Morici       2.0   1.8
1    Kilmaley  Jaime None Rodriguez     Michael V Pino       3.0   5.5

Dataset saved to .\race_results.nc
