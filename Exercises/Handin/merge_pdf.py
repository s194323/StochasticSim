import os
from PyPDF2 import PdfMerger

def merge_pdfs(folder_path, output_path):
    # Initialize PdfFileMerger object
    pdf_merger = PdfMerger()

    # Get list of PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print((pdf_files))

    # Sort the files to merge them in order
    pdf_files.sort()

    # Loop through each PDF file and append it to the merger
    for pdf_file in pdf_files:
        pdf_merger.append(os.path.join(folder_path, pdf_file))

    # Write the merged PDF to the output file
    with open(output_path, 'wb') as output_file:
        pdf_merger.write(output_file)

    # Close the PdfFileMerger object
    pdf_merger.close()

    print(f'Merged {len(pdf_files)} PDFs into {output_path}')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
merge_pdfs('./', 'merged.pdf')