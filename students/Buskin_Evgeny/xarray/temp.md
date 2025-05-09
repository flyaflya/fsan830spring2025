Track: coordinates include
 Track: coordinates include:
    trackID: The track identifier (from the CODE element in XML)
    trackName: The track name (from the NAME element in XML)
Race_date: The date of the race (from the RACE_DATE element in XML)
Race_number: The number of the race (from the RACE element in XML)
Entry: information about the entry
    horse: information about the horse
        horse: horse name
        jockey: jockey name (combining FIRST_NAME, MIDDLE_NAME, and LAST_NAME under jockey tag)
        trainer: trainer name (combining FIRST_NAME, MIDDLE_NAME, and LAST_NAME under trainer tag)
        

The dataset should also include relevant variables such as:
    Finish Position (OFFICIAL_FIN)
    Odds (DOLLAR_ODDS)
    Race purse (PURSE)
    Race distance (DISTANCE)
    Track condition (TRK_COND)

parse xml file using elemntTree library
Extract the relevant data for each dimension and variable 
Organize the data into appropriate data structures
Create an xarray dataset with the proper dimensions and coordinates
save the dataset in a suitable format (e.g., netCDF)

