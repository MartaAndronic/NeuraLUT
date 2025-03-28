import shutil
import os
import re
import subprocess

def convert_verilog_to_hex(log_dir):
    hex_dir = os.path.join(log_dir, 'hex')
    if not os.path.exists(hex_dir):
        os.makedirs(hex_dir)
    for file_name in os.listdir(log_dir):
        if re.match(r'layer\d+_N\d+\.v', file_name):
            verilog_file = os.path.join(log_dir, file_name)
            hex_file = os.path.join(hex_dir, file_name.replace('.v', '.hex'))
            in_bits, out_bits = bitwidths(verilog_file)
            toHex(verilog_file, hex_dir, in_bits, out_bits)

def bitwidths(verilog_file):
    with open(verilog_file, 'r') as file:
        content = file.read()
    
    module_pattern = re.compile(r'module\s+\w+\s*\(\s*input\s*\[(\d+):\d+\]\s*\w+,\s*output\s*\[(\d+):\d+\]\s*\w+\s*\);')
    match = module_pattern.search(content)
    
    if match:
        input_bitwidth = int(match.group(1)) + 1
        output_bitwidth = int(match.group(2)) + 1
        return input_bitwidth, output_bitwidth
    else:
        raise ValueError("Could not find module definition or bit widths in the Verilog file.")

def remove_verilog_luts(verilog_dir):
    for file_name in os.listdir(verilog_dir):
        if re.match(r'layer\d+_N\d+\.v', file_name) and 'top' not in file_name:
            os.remove(os.path.join(verilog_dir, file_name))

def tidy(output_dir):
    clean_folder(output_dir)
    sum_bits(output_dir)

def clean_folder(output_dir):
    if not os.path.exists(output_dir):
        print(f"Directory '{output_dir}' does not exist.")
        return

    files = os.listdir(output_dir)
    v_files = {}
    replacements = {
        "address": "M0",
        "data": "M1"
    }

    for filename in files:
        file_path = os.path.join(output_dir, filename)

        if filename.endswith('.v'):
            match = re.search(r'_v(\d+)', filename)
            if match:
                replace_in_file(file_path, replacements)
                version = int(match.group(1))
                base_name = filename[:match.start()] + '.v'

                if base_name not in v_files:
                    v_files[base_name] = (version, filename)
                else:
                    if version > v_files[base_name][0]:
                        old_file_path = os.path.join(output_dir, v_files[base_name][1])
                        try:
                            os.remove(old_file_path)
                            print(f"Removed older version file: {old_file_path}")
                        except Exception as e:
                            print(f"Error removing file {old_file_path}: {e}")

                        v_files[base_name] = (version, filename)

                    else:
                        try:
                            os.remove(file_path)
                            print(f"Removed older version file: {file_path}")
                        except Exception as e:
                            print(f"Error removing file {file_path}: {e}")
            else:
                base_name = filename
                if base_name not in v_files or v_files[base_name][0] == 0:
                    if base_name in v_files:
                        old_file_path = os.path.join(output_dir, v_files[base_name][1])
                        try:
                            os.remove(old_file_path)
                            print(f"Removed file: {old_file_path}")
                        except Exception as e:
                            print(f"Error removing file {old_file_path}: {e}")
                    v_files[base_name] = (0, filename)

        else:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed non-.v file: {file_path}")
                else:
                    print(f"Skipped non-file entry: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

    for base_name, (_, filename) in v_files.items():
        old_file_path = os.path.join(output_dir, filename)
        new_file_path = os.path.join(output_dir, base_name)
        if old_file_path != new_file_path:
            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed file: {old_file_path} to {new_file_path}")
            except Exception as e:
                print(f"Error renaming file {old_file_path}: {e}")

def run_reducedlut(output_dir, exiguity_value, bits=False, rarity_value=1):
    clear_bits_file(output_dir)
    reducedlut_path = os.environ.get("REDUCEDLUT")
    if not reducedlut_path:
        raise EnvironmentError("REDUCEDLUT environment variable not set. Please set it to the path of the reducedlut executable.")

    reducedlut_path = os.path.abspath(reducedlut_path)
    exe_dir = os.path.dirname(reducedlut_path)
    inputs_dir = os.path.join(output_dir, 'neuron_logs')
    table_dir = os.path.join(output_dir, 'hex')

    bits_cmd = 0
    if bits:
        bits_cmd = 1
    
    input_files = [f for f in os.listdir(inputs_dir) if f.endswith('_input.txt')]
    
    for input_filename in input_files:
        input_file = os.path.join(inputs_dir, input_filename)
        table_filename = input_filename.replace('_input.txt', '_hex.txt')
        table_file = os.path.join(table_dir, table_filename)
        
        name_value = input_filename.replace('_input.txt', '')
    
        try:
            result = subprocess.run(
                [
                    reducedlut_path,
                    '-table', table_file,
                    '-input', input_file,
                    '-rarity', str(rarity_value),
                    '-exiguity', str(exiguity_value),
                    '-name', name_value,
                    '-output', output_dir,
                    '-bits', str(bits_cmd)
                ],
                check=True,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Output from C++ executable:")
            print(result.stdout)
    
        except subprocess.CalledProcessError as e:
            print(f"Error running the executable for file {input_filename}: {e}")
            print("Standard output:")
            print(e.stdout)
            print("Standard error:")
            print(e.stderr)
    
        except FileNotFoundError:
            print("Executable not found. Please check the path.")

def bin_to_hex(bin_str, bits):
    hex_value = int(bin_str, 2)
    hex_width = (bits + 3) // 4  
    return '{:0{width}X}'.format(hex_value, width=hex_width)

def toHex(file, hex_dir, input_bits=12, output_bits=4):
    inname = file
    outname = os.path.join(hex_dir, os.path.splitext(os.path.basename(file))[0] + "_hex.txt")

    try:
        if not os.path.isfile(inname):
            raise FileNotFoundError(f"File '{inname}' does not exist.")

        hex_pairs = []

        with open(inname, 'r') as input_file:
            for line in input_file:
                input_pos = line.find(f"{input_bits}'b")
                output_pos = line.find(f"{output_bits}'b")

                input_hex = ''
                output_hex = ''

                if input_pos != -1 and (input_pos + 4 + input_bits <= len(line)):
                    bin_value = line[input_pos + 2 + len(str(input_bits)):input_pos + 2 + len(str(input_bits)) + input_bits]
                    input_hex = bin_to_hex(bin_value, input_bits)

                if output_pos != -1 and (output_pos + 3 + output_bits <= len(line)):
                    bin_value = line[output_pos + 2 + len(str(output_bits)):output_pos + 2 + len(str(output_bits)) + output_bits]
                    output_hex = bin_to_hex(bin_value, output_bits)

                if input_hex and output_hex:
                    hex_pairs.append((input_hex, output_hex))

        hex_pairs.sort(key=lambda pair: pair[0])

        with open(outname, 'w') as final_file:
            for _, output_hex in hex_pairs:
                final_file.write(output_hex + '\n')

        print(f"Conversion completed successfully for {inname}!")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return 1
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1

def replace_in_file(file_path, replacements):
    changes_made = False
    with open(file_path, 'r') as file:
        file_contents = file.read()

    original_contents = file_contents
    
    for old, new in replacements.items():
        if old in file_contents:
            file_contents = file_contents.replace(old, new)
            changes_made = True
    
    if changes_made:
        with open(file_path, 'w') as file:
            file.write(file_contents)
        
        print(f"Changes made in file: {file_path}")
        for old, new in replacements.items():
            if old in original_contents:
                occurrences = original_contents.count(old)
                print(f"Replaced {occurrences} occurrences of '{old}' with '{new}'")

def clear_bits_file(output_dir):
    parent_dir = os.path.abspath(os.path.join(output_dir, '..'))
    bits_file_path = os.path.join(parent_dir, 'bits.txt')

    print(f"Clearing file at: {bits_file_path}")
    
    try:
        with open(bits_file_path, 'w') as bits_file:
            bits_file.write("")  # This clears the file
        print(f"{bits_file_path} has been cleared.")
    except Exception as e:
        print(f"Error clearing {bits_file_path}: {e}")

def sum_bits(output_dir):
    total_initial_bits = 0
    total_final_bits = 0

    parent_dir = os.path.abspath(os.path.join(output_dir, '..'))
    files = os.listdir(parent_dir)

    for file_name in files:
        file_path = os.path.join(parent_dir, file_name)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    for line in file:
                        initial_match = re.search(r'Initial Size \(bit\): (\d+)', line)
                        final_match = re.search(r'Final Size \(bit\).*?: (\d+)', line)

                        if initial_match:
                            total_initial_bits += int(initial_match.group(1))
                        if final_match:
                            total_final_bits += int(final_match.group(1))

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    clear_bits_file(output_dir)
    bits_file_path = os.path.join(parent_dir, 'bits.txt')
    try:
        with open(bits_file_path, 'a') as bits_file:
            bits_file.write(f"Total Initial Size (bit): {total_initial_bits}\n")
            bits_file.write(f"Total Final Size (bit): {total_final_bits}\n")
        print(f"Totals written to {bits_file_path}")
    except Exception as e:
        print(f"Error writing to {bits_file_path}: {e}")

    print(f"Total Initial Size (bit): {total_initial_bits}")
    print(f"Total Final Size (bit): {total_final_bits}")