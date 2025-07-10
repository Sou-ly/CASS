# Dataset Augmentation Tool

This tool downloads datasets from Hugging Face and processes them to extract hardcoded inputs from CUDA source code using Google's Gemini AI.

## Features

- Download datasets from Hugging Face
- Extract hardcoded inputs from CUDA source code using Gemini AI
- Generate modified versions of code with command-line arguments instead of hardcoded values
- Save processed datasets in CSV format with the same structure as input plus augmented columns

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

3. Create a `.env` file in your project directory and add your API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage (Download Only)

```bash
python augment_dataset.py --dataset_name_or_path "your/dataset/name" --output "output_folder"
```

### Extract Inputs from Source Code

```bash
python augment_dataset.py \
    --dataset_name_or_path "your/dataset/name" \
    --output "output_folder" \
    --extract_inputs
```

### Advanced Options

```bash
python augment_dataset.py \
    --dataset_name_or_path "your/dataset/name" \
    --output "output_folder" \
    --extract_inputs \
    --batch_size 20 \
    --max_rows 5 \
    --force_download
```

## Arguments

- `--dataset_name_or_path`: The Hugging Face dataset name or path (required)
- `--output`: Output folder for processed dataset (default: "out")
- `--extract_inputs`: Enable input extraction using Gemini AI
- `--batch_size`: Number of samples to process in each batch (default: 10)
- `--model`: Gemini model to use (default: "gemini-2.5-pro")
- `--max_rows`: Maximum number of rows to process (for debugging, None=process all)
- `--force_download`: Force re-download of dataset (ignore cache)
- `--generate_test_cases`: Enable test case generation (not implemented yet)

## API Key Configuration

The Gemini API key must be set in your `.env` file as `GEMINI_API_KEY`.

## Caching

The script uses Hugging Face's built-in caching system:
- Datasets are automatically cached after first download
- Subsequent runs will load from cache (much faster)
- Use `--force_download` to ignore cache and re-download
- Cache directory location is displayed during loading

## Debugging

Use `--max_rows` to limit the number of samples processed for testing:
```bash
# Process only first 5 samples
python augment_dataset.py --dataset_name_or_path "your/dataset" --extract_inputs --max_rows 5
```

## Performance

The script uses multiprocessing to speed up Gemini API calls:
- Default: Uses up to 8 worker processes (capped to avoid rate limiting)
- Custom: Use `--num_workers` to specify exact number of processes
- Each worker processes samples in parallel, significantly reducing total processing time

## Output

The script creates the following files in the output folder:
- `augmented_dataset.csv`: CSV format of the processed dataset with extracted inputs
- `cuda_sources/`: Directory containing modified CUDA source files (cuda_source_1.cpp, cuda_source_2.cpp, etc.)
- `amd_sources/`: Directory containing modified AMD source files (amd_source_1.cpp, amd_source_2.cpp, etc.)
- `inputs/`: Directory containing input files (input_1.txt, input_2.txt, etc.) with extracted hardcoded values

### New Columns Added

- `extracted_inputs`: JSON array of hardcoded values found in the source code (same for both CUDA and AMD versions)

### File Structure

```
output/
├── augmented_dataset.csv          # Dataset with extracted inputs
├── cuda_sources/
│   ├── cuda_source_1.cpp         # Modified CUDA source for sample 1
│   ├── cuda_source_2.cpp         # Modified CUDA source for sample 2
│   └── ...
├── amd_sources/
│   ├── amd_source_1.cpp          # Modified AMD source for sample 1
│   ├── amd_source_2.cpp          # Modified AMD source for sample 2
│   └── ...
└── inputs/
    ├── input_1.txt               # Extracted inputs for sample 1 (space-separated)
    ├── input_2.txt               # Extracted inputs for sample 2 (space-separated)
    └── ...
```

### Usage Example

After processing, you can compile and run the programs with their extracted inputs:

```bash
# Compile the CUDA program
nvcc cuda_source_1.cpp -o cuda_program_1

# Run with extracted inputs
./cuda_program_1 $(cat input_1.txt)

# Or compile the AMD program
hipcc amd_source_1.cpp -o amd_program_1

# Run with extracted inputs
./amd_program_1 $(cat input_1.txt)
```

## Example

```bash
# Download and process a dataset
python augment_dataset.py \
    --dataset_name_or_path "microsoft/codebert-base" \
    --output "processed_data" \
    --extract_inputs
```

## Notes

- The script automatically detects source code columns in the dataset (looks for columns containing "source" in the name)
- Processing is done in batches to avoid rate limiting
- A small delay (0.5 seconds) is added between API calls to respect rate limits
- If no hardcoded inputs are found, the `extracted_inputs` will be an empty array and modified source columns will be null
- **Important**: Since CUDA and AMD sources represent the same algorithm, the same hardcoded inputs are extracted and applied to both versions
