def sample_xyz(input_file, output_file, step=10):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        
        # Index to track the line number
        line_idx = 0
        total_lines = len(lines)
        structure_count = 0
        sampled_count = 0

        print(f"Total number of lines in the input file: {total_lines}")
        
        while line_idx < total_lines:
            # Read the first line of the structure
            header = lines[line_idx]
            
            if '144' in header:
                # Determine the number of lines in this structure
                num_atoms = int(header.strip())
                structure_lines = lines[line_idx:line_idx + num_atoms + 2]
                structure_count += 1
                
                # Print the current structure number being processed
                print(f"Processing structure {structure_count}...")
                
                # Write the structure to the output file if it matches the step condition
                if (line_idx // (num_atoms + 2)) % step == 0:
                    outfile.writelines(structure_lines)
                    sampled_count += 1
                    print(f"  -> Structure {structure_count} sampled (sampled count: {sampled_count})")
                    
                # Move to the next structure
                line_idx += num_atoms + 2
            else:
                # Move to the next line if the current line doesn't start with '144'
                line_idx += 1

        print(f"Total number of structures processed: {structure_count}")
        print(f"Total number of structures sampled: {sampled_count}")

# Usage
input_file = 'monoclinic_11K_NpT_DFTD4_ML-dataset.xyz'
output_file = 'sampled-2_monoclinic_11K_NpT_DFTD4_ML-dataset.xyz'
sample_xyz(input_file, output_file, step=10)

