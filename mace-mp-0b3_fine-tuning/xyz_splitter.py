import random
import re
import argparse
import sys

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Split XYZ dataset into training, validation, and test sets')
    parser.add_argument('input_file', help='Input XYZ file path')
    parser.add_argument('--train_size', type=int, required=True, help='Number of configurations for training')
    parser.add_argument('--valid_size', type=int, required=True, help='Number of configurations for validation')
    parser.add_argument('--test_size', type=int, required=True, help='Number of configurations for testing')
    parser.add_argument('--rand_split', type=str, choices=['true', 'false'], default='false', 
                       help='Whether to use random splitting (true/false)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_prefix', type=str, default='water_xe_dataset', 
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Validate that rand_split is properly formatted
    use_random_split = args.rand_split.lower() == 'true'
    
    try:
        # Open the original file
        with open(args.input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.input_file}'")
        sys.exit(1)
    
    # Find the indices of the structure count lines (lines with just a number)
    # These lines appear before each configuration
    count_line_indices = []
    for i, line in enumerate(lines):
        if i > 0 and line.strip().isdigit():
            count_line_indices.append(i)
        # Also include the first line (first atom count)
        elif i == 0 and line.strip().isdigit():
            count_line_indices.append(i)
    
    # Calculate the total number of configurations
    total_configs = len(count_line_indices)
    
    # Validate that we have enough configurations
    requested_total = args.train_size + args.valid_size + args.test_size
    if requested_total > total_configs:
        print(f"Error: Requested total configurations ({requested_total}) exceeds available configurations ({total_configs})")
        sys.exit(1)
    elif requested_total < total_configs:
        print(f"Warning: Using {requested_total} out of {total_configs} available configurations")
    
    # Create a list of configuration indices
    if use_random_split:
        # Random splitting: shuffle all indices and take the first N
        all_indices = list(range(total_configs))
        random.shuffle(all_indices)
        selected_indices = all_indices[:requested_total]
        
        # Split the selected indices
        train_indices = sorted(selected_indices[:args.train_size])
        valid_indices = sorted(selected_indices[args.train_size:args.train_size + args.valid_size])
        test_indices = sorted(selected_indices[args.train_size + args.valid_size:args.train_size + args.valid_size + args.test_size])
    else:
        # Sequential splitting: take the first N configurations in order
        train_indices = list(range(args.train_size))
        valid_indices = list(range(args.train_size, args.train_size + args.valid_size))
        test_indices = list(range(args.train_size + args.valid_size, args.train_size + args.valid_size + args.test_size))
    
    # Function to extract configuration blocks and make the specified replacements
    def extract_blocks(indices):
        blocks = []
        for idx in indices:
            start_idx = count_line_indices[idx]
            # If this is the last configuration, extract to end of file
            if idx == len(count_line_indices) - 1:
                end_idx = len(lines)
            else:
                end_idx = count_line_indices[idx + 1]
                
            # Get the block lines
            block = lines[start_idx:end_idx]
            
            # Modify the header line (which is the line after the atom count)
            if len(block) >= 2:
                # The second line in each block is the header line
                header_line = block[1]
                
                # Make all the required replacements
                modified_header = header_line
                
                # Remove free_energy completely
                modified_header = re.sub(r'\s+free_energy=-?\d+\.\d+', '', modified_header)
                
                # Replace energy with REF_energy
                modified_header = re.sub(r'energy=', 'REF_energy=', modified_header)
                
                # Replace forces with REF_forces in Properties field
                modified_header = re.sub(r'forces:', 'REF_forces:', modified_header)
                
                # Replace stress with REF_stress
                modified_header = re.sub(r'stress=', 'REF_stress=', modified_header)
                
                # Add config_type and head fields, which MACE requires
    #            if "config_type=" not in modified_header:
    #                modified_header = modified_header.strip() + " config_type=\"default\" head=\"main\" pbc=\"T T T\"\n"
    #            else:
    #                # If config_type exists but head doesn't, add head
    #                if "head=" not in modified_header:
    #                    modified_header = modified_header.strip() + " head=\"main\"\n"
                
                block[1] = modified_header
            
            blocks.extend(block)
        return blocks
    
    # Extract blocks for each set
    train_blocks = extract_blocks(train_indices)
    valid_blocks = extract_blocks(valid_indices)
    test_blocks = extract_blocks(test_indices)
    
    # Write the parts to separate files
    with open(f'{args.output_prefix}_train.xyz', 'w') as f:
        f.writelines(train_blocks)
    
    with open(f'{args.output_prefix}_valid.xyz', 'w') as f:
        f.writelines(valid_blocks)
    
    with open(f'{args.output_prefix}_test.xyz', 'w') as f:
        f.writelines(test_blocks)
    
    # Print statistics
    print(f"Total configurations available: {total_configs}")
    print(f"Split method: {'Random' if use_random_split else 'Sequential'}")
    print(f"Training set: {len(train_indices)} configurations")
    print(f"Validation set: {len(valid_indices)} configurations")
    print(f"Testing set: {len(test_indices)} configurations")
    print(f"Total used: {len(train_indices) + len(valid_indices) + len(test_indices)} configurations")
    print(f"\nOutput files created:")
    print(f"  - {args.output_prefix}_train.xyz")
    print(f"  - {args.output_prefix}_valid.xyz")
    print(f"  - {args.output_prefix}_test.xyz")

if __name__ == "__main__":
    main()
