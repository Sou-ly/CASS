import os
import argparse
import csv
import json
from typing import List, Dict, Any, Optional
import time
import dotenv

import pandas as pd 
from datasets import load_dataset
import google.generativeai as genai
from google.generativeai import GenerativeModel

def parse_arguments():
    parser = argparse.ArgumentParser("Augment code data")
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="out", help="output folder")
    parser.add_argument("--extract_inputs", action="store_true", default=False)
    parser.add_argument("--generate_test_cases", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of samples to process in each batch")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini model to use")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process (for debugging, None=process all)")
    parser.add_argument("--force_download", action="store_true", default=False, help="Force re-download of dataset (ignore cache)")
    args = parser.parse_args()
    return args

def setup_gemini(api_key: str) -> GenerativeModel:
    """Setup Gemini API client"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    return model

def extract_inputs_from_both_sources(model: GenerativeModel, cuda_source: str, amd_source: str) -> Optional[Dict[str, Any]]:
    """
    Extract hardcoded inputs from both CUDA and AMD source code in the same prompt.
    Returns the same inputs for both versions since they represent the same algorithm.
    """
    prompt = f"""
    Analyze these two source codes that represent the same algorithm - one in CUDA and one in AMD HIP.
    Identify all hardcoded input values that could be made configurable.
    
    CUDA Source:
    {cuda_source}
    
    AMD HIP Source:
    {amd_source}
    
    Please:
    1. Identify all hardcoded values that could be inputs (like array sizes, matrix dimensions, constants, etc.)
    2. List them in order of appearance as a JSON array of strings
    3. Provide modified versions of both codes where these hardcoded values are replaced with command line arguments
    
    Return your response as a JSON object with this structure:
    {{
        "inputs": ["value1", "value2", ...],
        "modified_cuda": "// modified CUDA code with command line args",
        "modified_amd": "// modified AMD code with command line args"
    }}
    
    If no hardcoded inputs are found, return:
    {{
        "inputs": [],
        "modified_cuda": null,
        "modified_amd": null
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        result = json.loads(response_text)
        return result
    except Exception as e:
        print(f"Error processing code with Gemini: {e}")
        return {"inputs": [], "modified_cuda": None, "modified_amd": None}

def download_dataset(dataset_name_or_path: str, force_download: bool = False) -> pd.DataFrame:
    """Download dataset from Hugging Face and convert to pandas DataFrame"""
    print(f"Loading dataset: {dataset_name_or_path}")
    
    try:
        # Load the dataset with explicit caching
        download_mode = "force_redownload" if force_download else "reuse_cache_if_exists"
        print(f"Download mode: {download_mode}")
        
        dataset = load_dataset(
            dataset_name_or_path,
            cache_dir=None,  # Use default cache directory
            trust_remote_code=True,  # Trust the dataset code
            download_mode=download_mode  # Reuse cache if available
        )
        
        # Convert to pandas DataFrame
        if isinstance(dataset, dict):
            # If it's a dict of datasets, take the first one
            first_key = list(dataset.keys())[0]
            df = dataset[first_key].to_pandas()
        else:
            df = dataset.to_pandas()
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if dataset was loaded from cache
        from datasets import config
        cache_dir = config.HF_DATASETS_CACHE
        print(f"Cache directory: {cache_dir}")
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def process_dataset(df: pd.DataFrame, model: GenerativeModel, batch_size: int = 10, max_rows: int = None, output_folder: str = "out") -> pd.DataFrame:
    """Process the dataset to extract inputs from source code"""
    print("Processing dataset to extract inputs...")
    
    # Limit rows if max_rows is specified
    if max_rows is not None:
        df = df.head(max_rows)
        print(f"Limited to processing {max_rows} rows for debugging")
    
    # Initialize new columns
    df['extracted_inputs'] = None
    
    # Create directories for source files and inputs
    cuda_dir = os.path.join(output_folder, "cuda_sources")
    amd_dir = os.path.join(output_folder, "amd_sources")
    inputs_dir = os.path.join(output_folder, "inputs")
    os.makedirs(cuda_dir, exist_ok=True)
    os.makedirs(amd_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)
    
    total_samples = len(df)
    processed = 0
    successful_extractions = 0
    total_inputs_found = 0
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} (samples {i+1}-{batch_end})")
        
        for idx in range(i, batch_end):
            if idx >= len(df):
                break
                
            # Get both CUDA and AMD source code
            cuda_source = df.iloc[idx]['cuda_source']
            amd_source = df.iloc[idx]['amd_source']
            
            if pd.isna(cuda_source) or cuda_source == "":
                print(f"Sample {idx+1}: Skipping empty CUDA source code")
                processed += 1
                continue
            
            sample_num = idx + 1
            
            # Process CUDA source
            print(f"Sample {sample_num}: Processing CUDA source...", end="", flush=True)
            cuda_result = extract_inputs_from_both_sources(model, cuda_source, amd_source)
            
            if cuda_result and cuda_result['inputs']:
                # Store the extracted inputs from CUDA
                df.at[idx, 'extracted_inputs'] = json.dumps(cuda_result['inputs'])
                
                # Save inputs as text file
                inputs_filename = f"input_{sample_num}.txt"
                inputs_path = os.path.join(inputs_dir, inputs_filename)
                with open(inputs_path, 'w') as f:
                    # Write inputs as space-separated values for easy CLI usage
                    f.write(' '.join(cuda_result['inputs']))
                
                # Save modified CUDA source
                cuda_filename = f"cuda_source_{sample_num}.cu"
                cuda_path = os.path.join(cuda_dir, cuda_filename)
                with open(cuda_path, 'w') as f:
                    f.write(cuda_result['modified_cuda'])
                
                # Save modified AMD source
                amd_filename = f"amd_source_{sample_num}.cpp"
                amd_path = os.path.join(amd_dir, amd_filename)
                with open(amd_path, 'w') as f:
                    f.write(cuda_result['modified_amd'])
                
                print(f" ✓ Found {len(cuda_result['inputs'])} inputs")
                successful_extractions += 1
                total_inputs_found += len(cuda_result['inputs'])
                
            else:
                print(f" ✗ No inputs found")
            
            processed += 1
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
    
    print(f"\nProcessing Summary:")
    print(f"Total samples processed: {processed}")
    print(f"Samples with extracted inputs: {successful_extractions}")
    print(f"Total inputs found: {total_inputs_found}")
    print(f"Average inputs per sample: {total_inputs_found/successful_extractions:.1f}" if successful_extractions > 0 else "Average inputs per sample: 0")
    print(f"Modified source files saved to: {cuda_dir}/ and {amd_dir}/")
    print(f"Input files saved to: {inputs_dir}/")
    
    return df

def save_dataset(df: pd.DataFrame, output_folder: str):
    """Save the processed dataset as CSV only (modified sources saved as separate files)"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Save as CSV (only with extracted inputs, no modified source)
    csv_path = os.path.join(output_folder, "augmented_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path}")
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(df)}")
    if 'extracted_inputs' in df.columns:
        print(f"Samples with extracted inputs: {df['extracted_inputs'].notna().sum()}")
    print(f"Modified source files saved to: {output_folder}/cuda_sources/ and {output_folder}/amd_sources/")

def main():
    dotenv.load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Download dataset
    df = download_dataset(args.dataset_name_or_path, args.force_download)
    
    if args.extract_inputs:
        # Setup Gemini
        model = setup_gemini(gemini_api_key)
        
        # Process dataset to extract inputs
        df = process_dataset(df, model, args.batch_size, args.max_rows, args.output)
    
    # Save the processed dataset
    save_dataset(df, args.output)
    
    print("Dataset augmentation completed!")

if __name__ == "__main__":
    main()
