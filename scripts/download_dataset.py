import logging
import argparse
import os

DATASET_LINK = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"

def _get_arg_parser():
    parser = argparse.ArgumentParser(description="Download dataset")
    
    parser.add_argument("--output-folder", type=str, required=True, help="Output directory")
    
    return parser

if __name__ == "__main__":
    
    args = _get_arg_parser().parse_args()

    if os.path.exists(args.output_folder) and os.listdir(args.output_folder):
        logging.critical(f"Output folder '{args.output_folder}' is not empty")
        exit(1)

    elif not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        

    if not os.path.exists(".icbhi-data") or not os.listdir(".icbhi-data"):
        os.system(f"mkdir .icbhi-data")
        os.system(f"wget {DATASET_LINK} --no-check-certificate -O .icbhi-data/ICBHI_final_database.zip")
    
    os.system(f"unzip .icbhi-data/ICBHI_final_database.zip -d {args.output_folder}")