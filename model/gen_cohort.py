from sdv.single_table import CTGANSynthesizer as CTGAN, TVAESynthesizer as TVAE
from sdv.metadata import Metadata
from sdv.datasets.demo import download_demo
import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic cohort data using CTGAN.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input PARQUET file containing cohort data.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the generated synthetic cohort data.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of synthetic samples to generate.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for CTGAN training.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for CTGAN training.')
    parser.add_argument('--model', type=str, default="CTGAN", help='Select between CTGAN or TVAE.')
    args = parser.parse_args()

    ## Load cohort data
    cohort = pd.read_parquet(args.input)

    ## Define metadata for CTGAN
    metadata = Metadata.detect_from_dataframe(
        data=cohort,
        table_name='cohort')

    # Initialize and train CTGAN model
    if args.model == 'CTGAN':
        model = CTGAN(metadata,
                    verbose=True,
                    enable_gpu=torch.cuda.is_available(),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    pac = 8)
    elif args.model == 'TVAE':
        model = TVAE(
            metadata, # required
            verbose=True,
            enforce_min_max_values=True,
            enforce_rounding=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            enable_gpu=torch.cuda.is_available(),
        )
    else:
        raise ValueError("Expect 'CTGAN' or 'TVAE'")
    
    model.fit(cohort)

    # Generate synthetic samples
    sample_list = []
    for _ in tqdm(range((args.num_samples // args.batch_size) + 1), desc="Generating samples"):
        samples = model.sample(args.batch_size)
        sample_list.append(samples)
    
    # Save generated samples to PARQUET
    samples = pd.concat(sample_list,axis=0)[:args.num_samples]
    print(samples.head())
    samples.to_parquet(args.output, index=False)
    print(f"Synthetic cohort data saved to {args.output}")

if __name__ == "__main__":
    main()