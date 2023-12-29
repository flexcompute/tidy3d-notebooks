import os
import glob

def compile_rst_files(directory):
    rst_files = glob.glob(os.path.join(directory, '**', '*.rst'), recursive=True)
    rst_files.sort()  # Sort the files alphabetically

    with open('examples.rst', 'w') as output_file:
        output_file.write('Example Library\n================\n\n')

        for rst_file in rst_files:
            if rst_file != os.path.join(directory, 'index.rst'):
                with open(rst_file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content + '\n\n')

if __name__ == "__main__":
    compile_rst_files('.')