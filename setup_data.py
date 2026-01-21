import os
import zipfile
import shutil
import logging

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/setup_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
# Add console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def setup_data():
    logging.info("Starting data setup process.")
    # Define paths
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logging.info(f"Created directory: {data_dir}")
    
    # Files to process
    zip_file = 'train-corrected.zip'
    csv_files = ['test_Vges7qu.csv', 'sample_submission_V9Inaty.csv']
    
    # Unzip train data
    if os.path.exists(zip_file):
        logging.info(f"Extracting {zip_file}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logging.info("Extraction complete.")
        except Exception as e:
            logging.error(f"Failed to extract {zip_file}: {e}")
    else:
        logging.warning(f"{zip_file} not found.")

    # Move other CSV files
    for file in csv_files:
        src = os.path.join(base_dir, file)
        dst = os.path.join(data_dir, file)
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                logging.info(f"Moved {file} to {data_dir}")
            except Exception as e:
                logging.error(f"Failed to move {file}: {e}")
        else:
            logging.warning(f"{file} not found.")

    logging.info("Data setup process finished.")

if __name__ == "__main__":
    setup_data()