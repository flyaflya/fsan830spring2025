import pandas as pd
import xml.etree.ElementTree as ET
import os

def extract_result_data(xml_file):
    """
    Extract HorseName, RaceNumber, and OfficialFinish from result XML.
    
    Args:
        xml_file (str): Path to the XML file
        
    Returns:
        pd.DataFrame: DataFrame with HorseName, RaceNumber, OfficialFinish
    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        data = []
        for race in root.findall('.//RACE'):
            race_number = race.get('NUMBER')  # Attribute, not subtag
            for entry in race.findall('./ENTRY'):
                horse_name = entry.findtext('NAME')
                official_finish = entry.findtext('OFFICIAL_FIN')
                
                data.append({
                    'HorseName': horse_name,
                    'RaceNumber': race_number,
                    'OfficialFinish': pd.to_numeric(official_finish, errors='coerce')
                })

        df = pd.DataFrame(data)
        # df = df.dropna(subset=['OfficialFinish'])  # Optional: drop rows without valid finishes
        df['OfficialFinish'] = df['OfficialFinish'].astype(int)
        return df 
    
    except ET.ParseError as e:
        raise Exception(f"Error parsing XML file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    xml_path = "data/rawDataForTraining/resultsData/cd20230502tch.xml"  
    output_path = "race_results.csv"

    try:
        df_result = extract_result_data(xml_path)
        print(df_result.head())
        df_result.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
