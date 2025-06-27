def clean_rna_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            cleaned = []
            for char in line.strip():
                if char == 'T':
                    cleaned.append('U')
                elif char in ['A', 'U', 'C', 'G']:
                    cleaned.append(char)
                else:
                    cleaned.append('A')
            outfile.write(''.join(cleaned) + '\n')

# Example usage
clean_rna_file('lnc_valid.txt', 'lnc_valid1.txt')

